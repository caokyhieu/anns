#include "common.hpp"
#include "PipNN/pipnn.h"
#include "hnswlib/hnswlib.h"
#include "fastpipnn.hpp"
#include <functional>
#include <map>
#include <memory>
#include <iomanip>

struct ByteSpace : hnswlib::SpaceInterface<float> {
  size_t dim;
  explicit ByteSpace(size_t d):dim(d){}
  static float distance(const void* a,const void* b,const void* p) {
    return parlayANN::euclidian_distance(static_cast<const uint8_t*>(a),
        static_cast<const uint8_t*>(b),*static_cast<const size_t*>(p));
  }
  size_t get_data_size() override {return dim;}
  hnswlib::DISTFUNC<float> get_dist_func() override {return distance;}
  void* get_dist_func_param() override {return &dim;}
};
struct Span { const Id* p; size_t n; const Id* begin()const{return p;} const Id* end()const{return p+n;} };
int main(int argc,char** argv) {
 try {
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2) {
    require(i+1<argc,"all arguments require values");
    args[argv[i]]=argv[i+1];
  }
  auto get=[&](std::string name,std::string fallback=""){return args.count(name)?args[name]:fallback;};
  auto number=[&](std::string name,int fallback){return std::stoi(get(name,std::to_string(fallback)));};
  std::string algorithm=get("--algorithm","pipnn"), output=get("--output","metrics.json");
  require(algorithm=="pipnn"||algorithm=="hnsw"||algorithm=="fastpipnn","algorithm must be pipnn, hnsw, or fastpipnn");
  size_t threads=number("--threads",32), query_threads=number("--query-threads",1);
  require(threads>0 && threads<=256 && query_threads>0 && query_threads<=threads,"invalid thread counts");
  auto thread_text=std::to_string(threads);
  setenv("PARLAY_NUM_THREADS",thread_text.c_str(),1);
  Eigen::setNbThreads(1);
  double loading=now();
  Data data(get("--base")), queries(get("--queries"));
  GroundTruth gt(get("--gt"));
  require(data.dim==queries.dim && gt.n==queries.n,"query/ground-truth mismatch");
  require(data.n>10,"need more than 10 base vectors");
  for(Id p:gt.ids) require(p<data.n,"ground truth references IDs outside this base subset");
  // Reject a ground-truth file for a different base prefix before timing builds.
  int checks=number("--gt-checks",3);
  require(checks>=0 && checks<=int(queries.n),"invalid ground truth check count");
  for(int check=0;check<checks;++check) {
    size_t qi=size_t(check)*queries.n/checks;
    auto exact=parlay::tabulate(data.n,[&](size_t p){return data[p].distance(queries[qi]);});
    std::nth_element(exact.begin(),exact.begin()+9,exact.end());
    std::sort(exact.begin(),exact.begin()+10);
    std::vector<float> supplied;
    std::set<Id> unique_ids;
    for(size_t j=0;j<10;++j) {
      Id id=gt.ids[qi*gt.k+j];unique_ids.insert(id);
      supplied.push_back(data[id].distance(queries[qi]));
    }
    std::sort(supplied.begin(),supplied.end());
    require(unique_ids.size()==10,"duplicate ground-truth IDs");
    for(size_t j=0;j<10;++j)require(supplied[j]==exact[j],"ground truth does not match this dataset prefix");
  }
  size_t nq=std::min<size_t>(queries.n,number("--query-count",queries.n));
  require(nq>0,"query count must be positive");
  int degree=number("--degree",64), leaf=number("--leaf-size",1024), local_k=number("--local-k",2);
  int reservoir=number("--reservoir",160), bits=number("--hash-bits",12);
  int top_leaders=number("--leaders",1000), m=number("--m",32), efc=number("--ef-construction",200);
  double alpha=std::stod(get("--alpha","1.2"));
  std::string fanout=get("--fanout","10,3,1");
  require(degree>0&&degree<=512&&leaf>=2&&leaf<=2048&&local_k>0&&local_k<leaf,"invalid graph parameters");
  require(bits>0&&bits<=16&&reservoir>=degree&&reservoir<=65535,"invalid reservoir parameters");
  require(top_leaders>=3&&top_leaders<=2048&&m>=2&&m<=256&&efc>=m,"invalid leaders/HNSW parameters");
  require(alpha>=1&&alpha<=4,"invalid alpha");
  double load_seconds=now()-loading;
  parlayANN::Graph<Id> graph;
  ByteSpace space(data.dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
  fastpipnn::Index<uint8_t> fast;
  double start=now();
  if(algorithm=="pipnn") {
    commandLine command(argc,argv);
    parlayANN::pipnn_index<Point,Data,Id> index;
    index.build_index(graph,command,data,1,leaf,local_k,true,true,alpha,degree,
                      top_leaders,1,0,.005,fanout,bits,reservoir,false,0);
  } else if(algorithm=="fastpipnn") {
    fastpipnn::Params fp;
    fp.leaf_size=leaf; fp.local_k=local_k; fp.degree=degree; fp.reservoir=reservoir;
    fp.hash_bits=bits; fp.alpha=alpha; fp.top_leaders=top_leaders; fp.threads=threads;
    fp.fanout.clear();
    std::istringstream levels(fanout);
    for(std::string f;std::getline(levels,f,',');) fp.fanout.push_back(std::stoul(f));
    fast.build(data.location(0),data.n,data.dim,data.stride,fp);
  } else {
    hnsw=std::make_unique<hnswlib::HierarchicalNSW<float>>(&space,data.n,m,efc,100);
    hnsw->addPoint(data.location(0),0);
    parlay::parallel_for(1,data.n,[&](size_t p){hnsw->addPoint(data.location(p),p);},1);
  }
  double build_seconds=now()-start, build_peak=peak_mib();
  std::cout<<"BUILD "<<algorithm<<" n="<<data.n<<" seconds="<<build_seconds<<" peak_mib="<<build_peak<<std::endl;
  if(algorithm=="pipnn") {
    parlay::parallel_for(0,data.n,[&](size_t p){
      auto ns=graph[p];
      require(ns.size()<=size_t(degree),"degree violation");
      for(size_t j=0;j<ns.size();++j) require(ns[j]<data.n&&ns[j]!=p,"invalid graph edge");
    });
  }
  if(algorithm=="fastpipnn") {
    auto& s=fast.stats;
    std::cout<<"PHASES sketch="<<s.sketch_seconds<<" partition="<<s.partition_seconds
             <<" leaves="<<s.leaf_seconds<<" prune="<<s.prune_seconds<<" leaf_count="<<s.leaves
             <<" leaf_points="<<s.leaf_points<<" depth="<<s.depth<<" mean_reservoir="<<s.mean_reservoir
             <<" mean_degree="<<s.mean_degree<<std::endl;
  }
  // Both PiPNN variants search from the same entry: the point nearest the mean (or 0).
  std::string entry_mode=get("--entry","medoid");
  require(entry_mode=="medoid"||entry_mode=="zero","entry must be medoid or zero");
  Id entry=0;
  if(entry_mode=="medoid") entry=fastpipnn::medoid(data.location(0),data.n,data.dim,data.stride,threads);
  std::string save=get("--save");
  if(!save.empty()) {
    require(algorithm!="fastpipnn","--save is not implemented for fastpipnn");
    if(algorithm=="pipnn") graph.save(save.data());
    else hnsw->saveIndex(save);
  }
  std::vector<int> beams;
  std::istringstream beam_stream(get("--beams","16,32,64,128,256,512"));
  std::string token;
  while(std::getline(beam_stream,token,',')) {
    int b=std::stoi(token); require(b>=10&&b<=100000,"invalid beam"); beams.push_back(b);
  }
  std::ofstream out(output);
  require(bool(out),"cannot write metrics");
  out<<std::setprecision(10)<<"{\n\"algorithm\":\""<<algorithm<<"\",\"n\":"<<data.n
     <<",\"dimension\":"<<data.dim<<",\"query_count\":"<<nq
     <<",\"threads\":"<<threads<<",\"query_threads\":"<<query_threads
     <<",\"load_seconds\":"<<load_seconds<<",\"build_seconds\":"<<build_seconds
     <<",\"build_peak_mib\":"<<build_peak
     <<",\"degree\":"<<degree<<",\"leaf_size\":"<<leaf<<",\"local_k\":"<<local_k
     <<",\"reservoir\":"<<reservoir<<",\"hash_bits\":"<<bits
     <<",\"alpha\":"<<alpha<<",\"fanout\":\""<<fanout<<"\",\"leaders\":"<<top_leaders
     <<",\"m\":"<<m<<",\"ef_construction\":"<<efc
     <<",\"entry\":\""<<entry_mode<<"\",\"distance\":\"squared_l2_uint8\"";
  if(algorithm=="fastpipnn") {
    auto& s=fast.stats;
    out<<",\"phases\":{\"sketch_seconds\":"<<s.sketch_seconds<<",\"partition_seconds\":"<<s.partition_seconds
       <<",\"leaf_seconds\":"<<s.leaf_seconds<<",\"prune_seconds\":"<<s.prune_seconds
       <<",\"leaves\":"<<s.leaves<<",\"leaf_points\":"<<s.leaf_points<<",\"depth\":"<<s.depth
       <<",\"random_splits\":"<<s.random_splits<<",\"mean_reservoir\":"<<s.mean_reservoir
       <<",\"mean_degree\":"<<s.mean_degree<<"}";
  }
  out<<",\"queries\":[\n";
  for(size_t bi=0;bi<beams.size();++bi) {
    int beam=beams[bi];
    if(hnsw) hnsw->setEf(beam);
    auto search=[&](size_t i) {
      if(hnsw) {
        auto result=hnsw->searchKnn(queries.location(i),10);
        std::vector<Id> ids;
        while(!result.empty()){ids.push_back(result.top().second);result.pop();}
        return ids;
      }
      if(algorithm=="fastpipnn")
        return graph_search(data,queries.location(i),beam,10,[&](Id p){return fast.neighbors(p);},{entry});
      return graph_search(data,queries.location(i),beam,10,[&](Id p){
        auto ns=graph[p]; return Span{ns.begin(),ns.size()};
      },{entry});
    };
    for(size_t i=0;i<std::min<size_t>(nq,100);++i) search(i);
    std::vector<double> times;
    std::vector<std::vector<Id>> answers(nq);
    for(int repeat=0;repeat<3;++repeat) {
      double t=now();
      if(query_threads==1) {
        for(size_t i=0;i<nq;++i) answers[i]=search(i);
      } else {
        parlay::parallel_for(0,query_threads,[&](size_t worker){
          for(size_t i=worker;i<nq;i+=query_threads) answers[i]=search(i);
        },1);
      }
      times.push_back(now()-t);
    }
    size_t correct=0;
    for(size_t i=0;i<nq;++i) {
      std::sort(answers[i].begin(),answers[i].end());
      answers[i].erase(std::unique(answers[i].begin(),answers[i].end()),answers[i].end());
      for(Id id:answers[i])
        for(size_t j=0;j<10;++j) if(id==gt.ids[i*gt.k+j]) {++correct;break;}
    }
    std::sort(times.begin(),times.end());
    double recall=correct/double(nq*10), qps=nq/times[1];
    if(bi)out<<",\n";
    out<<"{\"beam\":"<<beam<<",\"recall_at_10\":"<<recall<<",\"qps\":"<<qps
       <<",\"median_seconds\":"<<times[1]<<"}";
    std::cout<<"QUERY beam="<<beam<<" recall="<<recall<<" qps="<<qps<<std::endl;
  }
  out<<"\n],\"peak_mib\":"<<peak_mib()<<"}\n";
  require(bool(out),"failed writing metrics");
 } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
