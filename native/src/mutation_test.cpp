#include <map>
#include "mutable.hpp"
int main() {
 try {
  setenv("PARLAY_NUM_THREADS","4",1);
  std::mt19937 rng(7);
  Data input(300,16), queries(20,16);
  for(auto* d:{&input,&queries})for(auto& v:d->bytes)v=rng()%256;
  MutablePiPNN::Config cfg;cfg.leaf_size=32;cfg.local_k=4;cfg.degree=16;cfg.capacity=32;
  MutablePiPNN index(std::move(input),cfg);
  index.validate();
  auto root=index.root.get();
  double start=now();
  for(size_t i=0;i<45;++i) {
    std::vector<uint8_t> x(16);for(auto&v:x)v=rng()%256;
    Id p=rng()%index.data.n;while(!index.live[p])p=(p+1)%index.data.n;
    if(i%3==0)index.add(x);
    else if(i%3==1)index.erase(p);
    else index.update(p,x);
    require(index.root.get()==root,"unexpected rebuild during local mutation");
    index.validate();
  }
  std::cout<<"45 mixed mutations and full invariants passed; seconds="<<now()-start
           <<" recall@10="<<index.recall(queries,64)<<std::endl;
  bool rebuilt=index.monitor(queries,.0,1);
  require(rebuilt&&index.mutations==0,"periodic monitor failed");
  index.validate();
  for(auto& neighbors:index.graph)neighbors.clear();
  require(index.monitor(queries,.99,1000),"recall monitor failed");
  index.validate();
  auto original=index.data.n;
  for(Id p=0;p<original;++p)if(index.live[p])index.erase(p);
  index.validate();
  require(index.search(queries.location(0)).empty(),"empty index query");
  index.rebuild();
  Id id=index.add(std::vector<uint8_t>(16,5));
  require(id==original,"IDs were reused");
  require(index.search(queries.location(0))==std::vector<Id>{id},"singleton query");
  index.validate();
  Data duplicate(40,16);
  MutablePiPNN duplicates(std::move(duplicate),cfg);
  duplicates.validate();duplicates.erase(0);duplicates.validate();
  // Deletion must recover a previously pruned candidate in the same hash cone.
  Data line(5,1);for(Id i=0;i<5;++i)*line.location(i)=i;
  MutablePiPNN::Config small;small.leaf_size=8;small.local_k=4;small.bits=1;
  small.degree=2;small.capacity=2;
  MutablePiPNN cones(std::move(line),small);
  cones.erase(1);cones.validate();
  bool recovered=false;
  for(size_t i=0;i<cones.sizes[0];++i)recovered|=cones.table(0)[i].id==2;
  require(recovered,"deleted cone winner was not replaced");
  bool invalid=false;
  try{cones.update(0,{1,2});}catch(const std::exception&){invalid=true;}
  require(invalid&&cones.live[0],"invalid update mutated state");
  std::cout<<"PASS: mutation, deletion recovery, duplicate, empty, stable-ID, validation and monitor checks"<<std::endl;
 } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
