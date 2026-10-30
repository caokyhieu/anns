#pragma once
#include "common.hpp"
#include "PipNN/pipnn_utils.h"
#include <memory>
#include <set>
#include <map>

// Experimental mutable layer. The static paper benchmark uses upstream PiPNN,
// not this layer: its retained leaf/candidate state has a different memory cost.
struct MutablePiPNN {
  struct Config {
    size_t leaf_size=256, local_k=4, degree=32, capacity=80, bits=12;
    size_t leaders=128;
    std::vector<size_t> fanout{3,2};
    float alpha=1.2f;
    uint64_t seed=7;
  };
  struct Tree {
    std::vector<uint8_t> anchors;
    std::vector<std::unique_ptr<Tree>> children;
    size_t fanout=1, leaf=SIZE_MAX;
  };
  struct Leaf {
    std::vector<Id> ids, knn;
    std::vector<std::vector<Id>> sym;
  };
  struct Entry { Id id=NIL; float dist=0; uint16_t hash=0; };
  Data data;
  Config cfg;
  std::unique_ptr<Tree> root;
  std::vector<Leaf> leaves;
  std::vector<uint8_t> live;
  std::vector<float> planes, sketches;
  std::vector<Entry> reservoirs;
  std::vector<uint16_t> sizes;
  std::vector<std::vector<Id>> memberships, candidates, base, incoming, graph;
  size_t mutations=0;
  explicit MutablePiPNN(Data input, Config config):data(std::move(input)),cfg(config) {
    require(cfg.leaf_size>=2&&cfg.leaf_size<=2048&&cfg.local_k>0&&cfg.local_k<cfg.leaf_size,
            "invalid mutable leaf parameters");
    require(cfg.degree>0&&cfg.capacity>=cfg.degree&&cfg.capacity<=65535&&cfg.bits>0&&cfg.bits<=16,
            "invalid mutable reservoir parameters");
    require(cfg.leaders>=2&&cfg.leaders<=2048&&!cfg.fanout.empty(),"invalid mutable routing parameters");
    for(auto f:cfg.fanout) require(f>0&&f<cfg.leaders,"fanout must be smaller than leader cap");
    live.assign(data.n,1);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<float> normal;
    planes.resize(cfg.bits*data.dim);
    for(float& value:planes)value=normal(rng);
    rebuild();
  }
  static void unique(std::vector<Id>& ids) {
    std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
  }
  void sketch(Id p) {
    for(size_t h=0;h<cfg.bits;++h) {
      float value=0;
      for(size_t d=0;d<data.dim;++d)value+=planes[h*data.dim+d]*data.location(p)[d];
      sketches[size_t(p)*cfg.bits+h]=value;
    }
  }
  uint16_t hash(Id p,Id q) const {
    uint16_t value=0;
    for(size_t h=0;h<cfg.bits;++h)
      value|=uint16_t(sketches[size_t(q)*cfg.bits+h]>=sketches[size_t(p)*cfg.bits+h])<<h;
    return value;
  }
  Entry* table(Id p) {return reservoirs.data()+size_t(p)*cfg.capacity;}
  void stream(Id p,Id q) {
    Entry item{q,data.distance(p,q),hash(p,q)};
    auto key=[](const Entry& e){return Pair{e.dist,e.id};};
    Entry* entries=table(p);
    size_t n=sizes[p],worst=0;
    for(size_t i=0;i<n;++i) {
      if(entries[i].hash==item.hash) {if(key(item)<key(entries[i]))entries[i]=item;return;}
      if(key(entries[worst])<key(entries[i]))worst=i;
    }
    if(n<cfg.capacity){entries[n]=item;++sizes[p];}
    else if(key(item)<key(entries[worst]))entries[worst]=item;
  }
  std::vector<Id> prune(Id p,std::vector<Id> ids) const {
    unique(ids);
    std::vector<Pair> ranked; ranked.reserve(ids.size());
    for(Id q:ids)if(q!=p&&live[q])ranked.emplace_back(data.distance(p,q),q);
    std::sort(ranked.begin(),ranked.end());
    std::vector<Id> result;
    for(auto item:ranked) {
      bool keep=true;
      for(Id q:result)if(cfg.alpha*data.distance(q,item.second)<item.first){keep=false;break;}
      if(keep)result.push_back(item.second);
      if(result.size()==cfg.degree)break;
    }
    return result;
  }
  std::unique_ptr<Tree> partition(parlay::sequence<Id> ids,size_t depth,uint64_t seed) {
    auto tree=std::make_unique<Tree>();
    if(ids.size()>cfg.leaf_size && depth<20) {
      size_t f=depth<cfg.fanout.size()?cfg.fanout[depth]:1;
      size_t count=std::min({ids.size(),cfg.leaders,std::max(f+1,size_t(ids.size()*.005))});
      f=std::min(f,count-1);
      auto leader_ids=ids;
      std::mt19937_64 rng(seed);
      // Partial Fisher-Yates avoids shuffling every large partition.
      for(size_t i=0;i<count;++i)std::swap(leader_ids[i],leader_ids[i+rng()%(ids.size()-i)]);
      leader_ids.resize(count);
      parlay::sequence<std::pair<Id,Id>> assignments(ids.size()*f);
      parlayANN::assign_leaders(data,ids,leader_ids,f,assignments,5000);
      auto groups=parlay::group_by_index(assignments,count);
      size_t largest=0;
      for(auto& g:groups)largest=std::max(largest,g.size());
      if(largest<ids.size()) {
        tree->fanout=f;
        for(Id p:leader_ids)
          tree->anchors.insert(tree->anchors.end(),data.location(p),data.location(p)+data.dim);
        // Tree construction is serial across children; assignment itself is parallel.
        for(size_t i=0;i<count;++i)tree->children.push_back(partition(std::move(groups[i]),depth+1,seed*6364136223846793005ULL+i+1));
        return tree;
      }
    }
    // Duplicate/degenerate partitions are supported by a bounded-row scalar fallback.
    tree->leaf=leaves.size();
    Leaf leaf; leaf.ids.assign(ids.begin(),ids.end());std::sort(leaf.ids.begin(),leaf.ids.end());
    leaves.push_back(std::move(leaf));
    return tree;
  }
  void symmetrize(Leaf& leaf) {
    size_t n=leaf.ids.size();
    leaf.sym.assign(n,{});
    for(size_t i=0;i<n;++i)for(size_t j=0;j<cfg.local_k;++j) {
      Id q=leaf.knn[i*cfg.local_k+j];
      if(q==NIL)continue;
      size_t row=std::lower_bound(leaf.ids.begin(),leaf.ids.end(),q)-leaf.ids.begin();
      leaf.sym[i].push_back(q);leaf.sym[row].push_back(leaf.ids[i]);
    }
    for(auto& row:leaf.sym)unique(row);
  }
  void local_rebuild(Leaf& leaf) {
    size_t n=leaf.ids.size();
    leaf.knn.assign(n*cfg.local_k,NIL);
    if(n<2){symmetrize(leaf);return;}
    bool gemm=n<=2048;
    if(gemm)parlayANN::eigen_build_distmat(leaf.ids,data);
    float* matrix=static_cast<float*>(parlayANN::get_dist_mat());
    std::vector<Pair> ranked;ranked.reserve(n);
    for(size_t i=0;i<n;++i) {
      ranked.clear();
      for(size_t j=0;j<n;++j)if(i!=j) {
        float d=gemm?matrix[std::max(i,j)*n+std::min(i,j)]:data.distance(leaf.ids[i],leaf.ids[j]);
        ranked.emplace_back(d,leaf.ids[j]);
      }
      size_t k=std::min(cfg.local_k,ranked.size());
      std::partial_sort(ranked.begin(),ranked.begin()+k,ranked.end());
      for(size_t j=0;j<k;++j)leaf.knn[i*cfg.local_k+j]=ranked[j].second;
    }
    symmetrize(leaf);
  }
  void local_insert(Leaf& leaf,Id p) {
    auto old_ids=std::move(leaf.ids);auto old_knn=std::move(leaf.knn);
    leaf.ids=old_ids;leaf.ids.insert(std::lower_bound(leaf.ids.begin(),leaf.ids.end(),p),p);
    leaf.knn.assign(leaf.ids.size()*cfg.local_k,NIL);
    std::vector<Pair> ranked;
    for(size_t row=0,old=0;row<leaf.ids.size();++row) {
      Id q=leaf.ids[row];ranked.clear();
      if(q==p) {for(Id r:old_ids)ranked.emplace_back(data.distance(p,r),r);}
      else {
        for(size_t j=0;j<cfg.local_k;++j) {
          Id r=old_knn[old*cfg.local_k+j];if(r!=NIL)ranked.emplace_back(data.distance(q,r),r);
        }
        ranked.emplace_back(data.distance(q,p),p);++old;
      }
      size_t k=std::min(cfg.local_k,ranked.size());
      std::partial_sort(ranked.begin(),ranked.begin()+k,ranked.end());
      for(size_t j=0;j<k;++j)leaf.knn[row*cfg.local_k+j]=ranked[j].second;
    }
    symmetrize(leaf);
  }
  std::vector<Id> collect(Id p) const {
    std::vector<Id> result;
    for(Id l:memberships[p]) {
      const Leaf& leaf=leaves[l];
      auto row=std::lower_bound(leaf.ids.begin(),leaf.ids.end(),p)-leaf.ids.begin();
      require(row<leaf.ids.size()&&leaf.ids[row]==p,"broken membership");
      result.insert(result.end(),leaf.sym[row].begin(),leaf.sym[row].end());
    }
    unique(result);return result;
  }
  void resize() {
    size_t n=data.n;
    sketches.resize(n*cfg.bits);reservoirs.resize(n*cfg.capacity);sizes.resize(n);
    memberships.resize(n);candidates.resize(n);base.resize(n);incoming.resize(n);graph.resize(n);
  }
  void repair(std::vector<Id> affected) {
    unique(affected);
    auto changed=affected;
    for(Id p:affected) {
      auto next=collect(p);
      bool removed=!std::includes(next.begin(),next.end(),candidates[p].begin(),candidates[p].end());
      if(removed)sizes[p]=0;
      for(Id q:next)if(removed||!std::binary_search(candidates[p].begin(),candidates[p].end(),q))stream(p,q);
      candidates[p]=std::move(next);
      std::vector<Id> winners;
      for(size_t j=0;j<sizes[p];++j)winners.push_back(table(p)[j].id);
      auto next_base=prune(p,winners);
      for(Id q:base[p]) {
        auto& in=incoming[q];in.erase(std::remove(in.begin(),in.end(),p),in.end());changed.push_back(q);
      }
      for(Id q:next_base){incoming[q].push_back(p);changed.push_back(q);}
      base[p]=std::move(next_base);
    }
    unique(changed);
    for(Id p:changed) {
      auto all=base[p];all.insert(all.end(),incoming[p].begin(),incoming[p].end());
      graph[p]=live[p]?prune(p,std::move(all)):std::vector<Id>{};
    }
  }
  void rebuild() {
    leaves.clear();memberships.clear();candidates.clear();base.clear();incoming.clear();graph.clear();
    resize();std::fill(sizes.begin(),sizes.end(),0);
    parlay::parallel_for(0,data.n,[&](size_t p){sketch(p);});
    parlay::sequence<Id> ids;
    for(Id p=0;p<data.n;++p)if(live[p])ids.push_back(p);
    root=partition(std::move(ids),0,cfg.seed);
    parlay::parallel_for(0,leaves.size(),[&](size_t l){local_rebuild(leaves[l]);},1);
    for(Id l=0;l<leaves.size();++l)for(Id p:leaves[l].ids)memberships[p].push_back(l);
    parlay::parallel_for(0,data.n,[&](size_t p) {
      if(!live[p])return;
      candidates[p]=collect(p);
      for(Id q:candidates[p])stream(p,q);
      std::vector<Id> winners;
      for(size_t j=0;j<sizes[p];++j)winners.push_back(table(p)[j].id);
      base[p]=prune(p,winners);
    });
    for(Id p=0;p<data.n;++p)for(Id q:base[p])incoming[q].push_back(p);
    parlay::parallel_for(0,data.n,[&](size_t p) {
      auto all=base[p];all.insert(all.end(),incoming[p].begin(),incoming[p].end());
      graph[p]=prune(p,std::move(all));
    });
    mutations=0;
  }
  void route(const uint8_t* x,const Tree& tree,std::vector<Id>& result)const {
    if(tree.leaf!=SIZE_MAX){result.push_back(tree.leaf);return;}
    std::vector<Pair> ranked;
    for(Id i=0;i<tree.children.size();++i)
      ranked.emplace_back(parlayANN::euclidian_distance(x,tree.anchors.data()+size_t(i)*data.dim,data.dim),i);
    std::partial_sort(ranked.begin(),ranked.begin()+tree.fanout,ranked.end());
    for(size_t i=0;i<tree.fanout;++i)route(x,*tree.children[ranked[i].second],result);
  }
  void insert_at(Id p,const uint8_t* x) {
    std::memcpy(data.location(p),x,data.dim);live[p]=1;sketch(p);
    std::vector<Id> affected{p};
    route(data.location(p),*root,memberships[p]);
    for(Id l:memberships[p]) {
      local_insert(leaves[l],p);
      affected.insert(affected.end(),leaves[l].ids.begin(),leaves[l].ids.end());
    }
    repair(std::move(affected));++mutations;
  }
  Id add(const std::vector<uint8_t>& x) {
    require(x.size()==data.dim,"dimension mismatch");
    Id p=data.append(x.data());live.push_back(0);resize();insert_at(p,x.data());return p;
  }
  void erase(Id p) {
    require(p<data.n&&live[p],"unknown vector ID");
    std::vector<Id> affected{p};
    for(Id l:memberships[p]) {
      auto& leaf=leaves[l];
      leaf.ids.erase(std::lower_bound(leaf.ids.begin(),leaf.ids.end(),p));
      local_rebuild(leaf);
      affected.insert(affected.end(),leaf.ids.begin(),leaf.ids.end());
    }
    memberships[p].clear();live[p]=0;
    repair(std::move(affected));++mutations;
  }
  void update(Id p,const std::vector<uint8_t>& x) {
    require(x.size()==data.dim&&p<data.n&&live[p],"invalid update");
    erase(p);insert_at(p,x.data());--mutations;
  }
  std::vector<Id> search(const uint8_t* x,size_t k=10,size_t beam=64)const {
    require(beam>=k&&k>0,"invalid search parameters");
    std::vector<Id> routed,seeds;route(x,*root,routed);
    for(Id l:routed)if(!leaves[l].ids.empty())seeds.push_back(leaves[l].ids.front());
    auto first=std::find(live.begin(),live.end(),1);
    if(first==live.end())return {};
    seeds.push_back(first-live.begin());unique(seeds);
    return graph_search(data,x,beam,k,[&](Id p)->const std::vector<Id>& {return graph[p];},seeds);
  }
  double recall(const Data& queries,size_t beam=64)const {
    require(queries.dim==data.dim,"query dimension mismatch");
    size_t hits=0,total=0;
    for(Id i=0;i<queries.n;++i) {
      std::vector<Pair> ranked;
      for(Id p=0;p<data.n;++p)if(live[p])ranked.emplace_back(data[p].distance(queries[i]),p);
      size_t k=std::min<size_t>(10,ranked.size());
      if(!k)continue;
      std::partial_sort(ranked.begin(),ranked.begin()+k,ranked.end());
      auto found=search(queries.location(i),k,beam);
      for(size_t j=0;j<k;++j)hits+=std::find(found.begin(),found.end(),ranked[j].second)!=found.end();
      total+=k;
    }
    require(total>0,"recall requires live vectors and queries");return double(hits)/total;
  }
  bool monitor(const Data& queries,double threshold=.9,size_t period=1000,size_t beam=64) {
    if(recall(queries,beam)<threshold || mutations>=period){rebuild();return true;}return false;
  }
  void validate() {
    for(auto& leaf:leaves) {
      auto old=leaf.knn;auto sym=leaf.sym;local_rebuild(leaf);
      require(old==leaf.knn&&sym==leaf.sym,"local top-k/symmetric candidates mismatch");
      for(Id p:leaf.ids)require(live[p],"dead leaf member");
    }
    for(Id p=0;p<data.n;++p) {
      auto wanted=collect(p);require(wanted==candidates[p],"candidate mismatch");
      // Independent offline cone winners, followed by a global distance cap.
      std::map<uint16_t,Pair> winners;
      for(Id q:wanted) {
        Pair pair{data.distance(p,q),q};auto h=hash(p,q);
        auto it=winners.find(h);if(it==winners.end()||pair<it->second)winners[h]=pair;
      }
      std::vector<Pair> ranked;for(auto item:winners)ranked.push_back(item.second);
      std::sort(ranked.begin(),ranked.end());if(ranked.size()>cfg.capacity)ranked.resize(cfg.capacity);
      std::vector<Id> want,got;
      for(auto item:ranked)want.push_back(item.second);
      for(size_t j=0;j<sizes[p];++j)got.push_back(table(p)[j].id);
      unique(want);unique(got);require(want==got,"reservoir mismatch");
      require(base[p]==prune(p,want),"base graph mismatch");
      std::vector<Id> reverse;
      for(Id q=0;q<data.n;++q)if(std::find(base[q].begin(),base[q].end(),p)!=base[q].end())reverse.push_back(q);
      auto actual=incoming[p];unique(actual);require(actual==reverse,"reverse graph mismatch");
      auto all=base[p];all.insert(all.end(),reverse.begin(),reverse.end());
      require(graph[p]==prune(p,all),"final graph mismatch");
      for(Id q:graph[p])require(q!=p&&live[q],"stale final edge");
    }
  }
};
