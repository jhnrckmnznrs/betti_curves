// C++17 native sparse filtered Morse cancellation, using contiguous indexed
// generators and on-disk deferred contractions. F2 arithmetic throughout.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <queue>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <limits>

struct Stats {
  uint64_t remaining,pairs,operations,trace_bytes,nonfree,tile_pairs,hit_budget,adaptive_stopped,adaptive_windows;
};
using U=uint32_t;
using V=std::vector<U>;

static bool contains(const V& v,U i) {return std::binary_search(v.begin(),v.end(),i);}
static void erase_val(V& v,U i) {auto it=std::lower_bound(v.begin(),v.end(),i);if(it!=v.end()&&*it==i)v.erase(it);}
static void add_val(V& v,U i) {auto it=std::lower_bound(v.begin(),v.end(),i);if(it==v.end()||*it!=i)v.insert(it,i);}
static V symdiff(const V& a,const V& b) {
  V out;out.reserve(a.size()+b.size());
  std::set_symmetric_difference(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(out));
  return out;
}
struct Candidate {
  uint64_t cost; U length;int32_t level;uint8_t dim;U a,b;
};
struct Greater {bool operator()(const Candidate&x,const Candidate&y)const{
  return std::tie(x.cost,x.length,x.level,x.dim,x.a,x.b)>
         std::tie(y.cost,y.length,y.level,y.dim,y.a,y.b);
}};
static void write_u32(std::ofstream& out,U v) {out.write(reinterpret_cast<const char*>(&v),sizeof(v));}

// A single live candidate per lower cell, with in-place updates. Unlike a
// duplicate-rich priority queue its storage never exceeds O(number of cells).
struct IndexedCandidates {
 std::vector<Candidate> items;
 std::vector<int32_t> location;
 Greater greater;
 explicit IndexedCandidates(size_t n):location(n,-1){}
 bool less(const Candidate& a,const Candidate& b) const {return greater(b,a);}
 void swap_at(size_t i,size_t j){
   std::swap(items[i],items[j]); location[items[i].a]=static_cast<int32_t>(i);
   location[items[j].a]=static_cast<int32_t>(j);
 }
 void up(size_t i){while(i){size_t j=(i-1)/2;if(!less(items[i],items[j]))break;swap_at(i,j);i=j;}}
 void down(size_t i){size_t n=items.size();while(2*i+1<n){size_t j=2*i+1;if(j+1<n&&less(items[j+1],items[j]))j++;
   if(!less(items[j],items[i]))break;swap_at(i,j);i=j;}}
 void erase(U a){int32_t p=location[a];if(p<0)return;size_t i=static_cast<size_t>(p);
   location[a]=-1;size_t last=items.size()-1;
   if(i!=last){items[i]=items[last];location[items[i].a]=static_cast<int32_t>(i);}
   items.pop_back();if(i<items.size()){U moved=items[i].a;up(i);down(static_cast<size_t>(location[moved]));}}
 void update(const Candidate& c){int32_t p=location[c.a];if(p<0){location[c.a]=static_cast<int32_t>(items.size());items.push_back(c);up(items.size()-1);}
   else{size_t i=static_cast<size_t>(p);items[i]=c;up(i);down(static_cast<size_t>(location[c.a]));}}
 Candidate pop(){Candidate c=items.front();erase(c.a);return c;}
 bool empty()const{return items.empty();}
 size_t size()const{return items.size();}
};


extern "C" int compact_morse_run(
  const U* ids,const int32_t* levels,const uint8_t* dims,const uint64_t* offsets,
  const U* data,uint64_t n,int32_t ny,int32_t nz,int32_t block_voxels,
  int local_first,int indexed_scheduler,uint64_t max_operations,uint64_t max_fill,uint64_t max_heap,
  uint64_t max_trace_bytes,uint64_t max_pairs,double max_seconds,
  double adaptive_cohomology_ns_per_cell,uint64_t adaptive_window,uint64_t adaptive_patience,
  const char* output_dir,Stats* stats,char*err,uint64_t err_size) {
 try {
  using Clock=std::chrono::steady_clock;
  auto start=Clock::now();
  if(n>std::numeric_limits<U>::max())throw std::runtime_error("Too many cells for uint32 indices");
  std::vector<V> cols(n),rows(n);
  std::vector<uint8_t> alive(n,1);
  for(uint64_t i=0;i<n;i++){
    if(offsets[i+1]<offsets[i])throw std::runtime_error("Bad CSR offsets");
    cols[i].assign(data+offsets[i],data+offsets[i+1]);
    if(!std::is_sorted(cols[i].begin(),cols[i].end()))throw std::runtime_error("CSR must be sorted");
    for(U f:cols[i]){
      if(f>=n||dims[f]+1!=dims[i]||levels[f]>levels[i])throw std::runtime_error("Invalid filtered boundary");
      rows[f].push_back(static_cast<U>(i));
    }
  }
  for(V& row:rows)std::sort(row.begin(),row.end());
  if(adaptive_cohomology_ns_per_cell<0.0 ||
     (adaptive_cohomology_ns_per_cell>0.0 && (!indexed_scheduler || !adaptive_window || !adaptive_patience)))
    throw std::runtime_error("Invalid adaptive stopping configuration");
  const std::string base(output_dir);
  std::ofstream tr(base+"/trace.bin",std::ios::binary|std::ios::trunc);
  std::ofstream idx(base+"/trace_offsets.bin",std::ios::binary|std::ios::trunc);
  if(!tr||!idx)throw std::runtime_error("Cannot create trace files");
  uint64_t trace_bytes=0;
  uint64_t pairs=0,ops=0,nonfree=0,tile_pairs=0;
  bool hit=false,adaptive_stopped=false;uint64_t adaptive_windows=0;
  auto tile=[&](U i){
    uint64_t p=ids[i],sz=2ull*nz+1,sy=2ull*ny+1;
    return std::make_tuple(p/(sy*sz)/(2*block_voxels),((p/sz)%sy)/(2*block_voxels),(p%sz)/(2*block_voxels));
  };
  auto phase=[&](bool local){
    std::priority_queue<Candidate,std::vector<Candidate>,Greater> heap;
    auto push=[&](U a,U b){
      if(!alive[a]||!alive[b]||levels[a]!=levels[b])return;
      if(local&&tile(a)!=tile(b))return;
      if(!contains(cols[b],a))return;
      uint64_t cost=(rows[a].size()-1ull)*(cols[b].size()-1ull);
      if(cost>max_fill)return;
      heap.push(Candidate{cost,static_cast<U>(cols[b].size()),levels[a],dims[a],a,b});
    };
    for(U a=0;a<n;a++)if(alive[a])for(U b:rows[a]){
      push(a,b);
      if(heap.size()>max_heap){hit=true;return;}
    }
    uint64_t attempts=0;
    while(!heap.empty()&&pairs<max_pairs){
      auto item=heap.top();heap.pop();
      if((++attempts&8191)==0){
        if(std::chrono::duration<double>(Clock::now()-start).count()>max_seconds){hit=true;return;}
      }
      U a=item.a,b=item.b;
      if(!alive[a]||!alive[b]||!contains(cols[b],a))continue;
      if(local&&tile(a)!=tile(b))continue;
      uint64_t cost=(rows[a].size()-1ull)*(cols[b].size()-1ull);
      if(cost!=item.cost||cols[b].size()!=item.length){
        if(cost<=max_fill)push(a,b);
        continue;
      }
      if(ops+cost>max_operations){hit=true;return;}
      const V boundary=cols[b];
      V image=boundary;erase_val(image,a);
      V affected=rows[a];erase_val(affected,b);
      V upper_rows=rows[b];
      uint64_t size=16+4ull*(image.size()+affected.size());
      if(trace_bytes+size>max_trace_bytes){hit=true;return;}
      uint64_t off=trace_bytes;
      idx.write(reinterpret_cast<const char*>(&off),8);
      write_u32(tr,a);write_u32(tr,b);
      write_u32(tr,static_cast<U>(image.size()));write_u32(tr,static_cast<U>(affected.size()));
      if(!image.empty())tr.write(reinterpret_cast<const char*>(image.data()),image.size()*4);
      if(!affected.empty())tr.write(reinterpret_cast<const char*>(affected.data()),affected.size()*4);
      trace_bytes+=size;
      if(!affected.empty())nonfree++;
      if(local)tile_pairs++;
      V touched=boundary;
      for(U x:affected){
        const V old=cols[x];
        V neu=symdiff(old,boundary);
        V toggled=symdiff(old,neu);
        for(U f:toggled){
          if(contains(neu,f))add_val(rows[f],x);
          else erase_val(rows[f],x);
          touched.push_back(f);
        }
        cols[x]=std::move(neu);
      }
      for(U y:upper_rows){erase_val(cols[y],b);}
      for(U f:cols[b])erase_val(rows[f],b);
      for(U f:cols[a])erase_val(rows[f],a);
      cols[a].clear();cols[b].clear();rows[a].clear();rows[b].clear();
      alive[a]=0;alive[b]=0;
      ops+=cost;pairs++;
      std::sort(touched.begin(),touched.end());
      touched.erase(std::unique(touched.begin(),touched.end()),touched.end());
      for(U f:touched)if(alive[f])for(U x:rows[f]){
        push(f,x);
        if(heap.size()>max_heap){hit=true;return;}
      }
    }
  };

  auto indexed_phase=[&](bool local){
    IndexedCandidates heap(n);
    // A per-input forecast, not a theorem about cohomology cost.  A completed
    // window of cancellations removes exactly two critical cells per pair.
    // Stop if the measured time to remove them persistently exceeds the user's
    // estimated time those cells would cost downstream.  No partial pivot is
    // ever executed: every finished cancellation is a filtered chain homotopy.
    auto checkpoint_time=Clock::now();
    uint64_t checkpoint_pairs=pairs, unprofitable_windows=0;
    auto checkpoint=[&](){
      if(adaptive_cohomology_ns_per_cell<=0.0||pairs-checkpoint_pairs<adaptive_window)return false;
      auto now=Clock::now();
      double observed_ns=std::chrono::duration<double,std::nano>(now-checkpoint_time).count();
      double saved_ns=2.0*static_cast<double>(pairs-checkpoint_pairs)*adaptive_cohomology_ns_per_cell;
      adaptive_windows++;
      if(observed_ns>saved_ns)unprofitable_windows++;
      else unprofitable_windows=0;
      checkpoint_time=now;checkpoint_pairs=pairs;
      if(unprofitable_windows>=adaptive_patience){adaptive_stopped=true;return true;}
      return false;
    };
    bool skipped=false;
    auto pick=[&](U a){
      heap.erase(a);
      if(!alive[a])return;
      bool found=false; Candidate best{};
      for(U b:rows[a]){
        if(!alive[b]||levels[a]!=levels[b])continue;
        if(local&&tile(a)!=tile(b))continue;
        uint64_t cost=(rows[a].size()-1ull)*(cols[b].size()-1ull);
        if(cost>max_fill)continue;
        Candidate c{cost,static_cast<U>(cols[b].size()),levels[a],dims[a],a,b};
        if(!found||heap.less(c,best)){best=c;found=true;}
      }
      if(found){
        if(heap.location[a]>=0||heap.size()<max_heap)heap.update(best);
        else skipped=true; // Exact reduction still valid; this cancellation is deferred.
      }
    };
    // Populate a bounded candidate window incrementally, rather than quitting
    // before the first cancellation when the filtration exceeds max_heap.
    // Eligible cells not admitted because of capacity may remain uncollapsed;
    // this affects compression, never the persistence of the retained complex.
    uint64_t cursor=0;
    auto refill=[&](){
      while(cursor<n&&heap.size()<max_heap){
        U a=static_cast<U>(cursor++);
        if(alive[a])pick(a);
      }
    };
    refill();
    uint64_t attempts=0;
    while((!heap.empty()||cursor<n)&&pairs<max_pairs){
      if(heap.empty()){refill();if(heap.empty())break;}
      if((++attempts&4095)==0&&std::chrono::duration<double>(Clock::now()-start).count()>max_seconds){hit=true;return;}
      Candidate item=heap.pop();U a=item.a,b=item.b;
      if(!alive[a])continue;
      // Revalidate every cost against the current differential and row support.
      // If another incident coface has become cheaper, update rather than
      // executing the stale candidate. Both choices are filtered cancellations.
      pick(a);
      if(heap.location[a]<0)continue;
      Candidate best=heap.items[static_cast<size_t>(heap.location[a])];
      if(best.b!=b||best.cost!=item.cost||best.length!=item.length)continue;
      heap.erase(a);
      if(ops+best.cost>max_operations){hit=true;return;}
      const V boundary=cols[b];V image=boundary;erase_val(image,a);
      V affected=rows[a];erase_val(affected,b);V upper_rows=rows[b];
      uint64_t size=16+4ull*(image.size()+affected.size());
      if(trace_bytes+size>max_trace_bytes){hit=true;return;}
      uint64_t off=trace_bytes;idx.write(reinterpret_cast<const char*>(&off),8);
      write_u32(tr,a);write_u32(tr,b);
      write_u32(tr,static_cast<U>(image.size()));write_u32(tr,static_cast<U>(affected.size()));
      if(!image.empty())tr.write(reinterpret_cast<const char*>(image.data()),image.size()*4);
      if(!affected.empty())tr.write(reinterpret_cast<const char*>(affected.data()),affected.size()*4);
      trace_bytes+=size;if(!affected.empty())nonfree++;if(local)tile_pairs++;
      V touched=boundary;
      for(U x:affected){
        const V old=cols[x];V neu=symdiff(old,boundary);V toggled=symdiff(old,neu);
        for(U f:toggled){if(contains(neu,f))add_val(rows[f],x);else erase_val(rows[f],x);touched.push_back(f);}
        cols[x]=std::move(neu);
      }
      for(U y:upper_rows)erase_val(cols[y],b);
      for(U f:cols[b])erase_val(rows[f],b);
      for(U f:cols[a])erase_val(rows[f],a);
      cols[a].clear();cols[b].clear();rows[a].clear();rows[b].clear();alive[a]=0;alive[b]=0;
      ops+=best.cost;pairs++;
      if(checkpoint())return;
      std::sort(touched.begin(),touched.end());touched.erase(std::unique(touched.begin(),touched.end()),touched.end());
      for(U f:touched)if(alive[f])pick(f);
      refill();
    }
    if(skipped)hit=true;
  };
  if(indexed_scheduler){if(local_first){indexed_phase(true);if(!hit&&!adaptive_stopped&&pairs<max_pairs)indexed_phase(false);}
    else indexed_phase(false);}
  else {if(local_first){phase(true);if(!hit&&pairs<max_pairs)phase(false);}
    else phase(false);}

  tr.close();idx.close();
  if(!tr||!idx)throw std::runtime_error("Trace output write failure");
  std::ofstream surv(base+"/critical_index.bin",std::ios::binary|std::ios::trunc),
                off(base+"/critical_offsets.bin",std::ios::binary|std::ios::trunc),
                dat(base+"/critical_data.bin",std::ios::binary|std::ios::trunc);
  if(!surv||!off||!dat)throw std::runtime_error("Cannot create critical CSR");
  uint64_t offset=0,remaining=0;
  off.write(reinterpret_cast<const char*>(&offset),8);
  for(U i=0;i<n;i++)if(alive[i]){
    surv.write(reinterpret_cast<const char*>(&i),4);
    if(!cols[i].empty())dat.write(reinterpret_cast<const char*>(cols[i].data()),cols[i].size()*4);
    offset+=cols[i].size();off.write(reinterpret_cast<const char*>(&offset),8);
    remaining++;
  }
  surv.close();off.close();dat.close();
  if(!surv||!off||!dat)throw std::runtime_error("Critical CSR write failure");
  *stats=Stats{remaining,pairs,ops,trace_bytes,nonfree,tile_pairs,hit?1ull:0ull,adaptive_stopped?1ull:0ull,adaptive_windows};
  return 0;
 }catch(const std::exception& e){
   if(err_size){std::strncpy(err,e.what(),err_size-1);err[err_size-1]='\0';}
   return 1;
 }
}

// Sparse-annotation persistent H1 cohomology on the *critical* filtered
// algebraic complex. No Python set/dict is used in the large filtration pass.
// Critical boundary indices refer to the full precursor index space.
extern "C" int compact_h1_run(
  const U* cell_ids,const int32_t* all_levels,const uint8_t* all_dims,
  uint64_t precursor_n,const U* critical,const uint64_t* offsets,
  const U* data,uint64_t n,uint64_t max_operations,uint64_t max_annotation_entries,uint64_t max_output_pairs,double max_seconds,
  const char* pair_path,uint64_t* pair_count,uint64_t* op_count,uint64_t* peak_entries,
  char*err,uint64_t err_size) {
 try{
  using Clock=std::chrono::steady_clock;
  auto start=Clock::now();
  if(n>std::numeric_limits<U>::max())throw std::runtime_error("Critical complex too large for uint32");
  std::vector<int32_t> lookup(precursor_n,-1);
  for(U i=0;i<n;i++){
    if(critical[i]>=precursor_n)throw std::runtime_error("Critical index out of range");
    lookup[critical[i]]=i;
  }
  std::vector<U> order(n),rank(n);std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](U a,U b){
    U x=critical[a],y=critical[b];
    return std::tie(all_levels[x],all_dims[x],cell_ids[x])<
           std::tie(all_levels[y],all_dims[y],cell_ids[y]);
  });
  for(U i=0;i<n;i++)rank[order[i]]=i;
  std::vector<U> parent(n);std::iota(parent.begin(),parent.end(),0);
  auto root=[&](U p){while(parent[p]!=p){parent[p]=parent[parent[p]];p=parent[p];}return p;};
  std::vector<V> annotation(n),reverse(n);
  std::ofstream out(pair_path,std::ios::binary|std::ios::trunc);
  if(!out)throw std::runtime_error("Cannot open persistence pair catalog");
  uint64_t count=0,entries=0,max_seen=0,ops=0;
  auto check_entries=[&](){max_seen=std::max(max_seen,entries);if(entries>max_annotation_entries)throw std::runtime_error("Cohomology annotation budget exceeded");};
  for(U step=0;step<n;step++){
    if((step&8191)==0&&std::chrono::duration<double>(Clock::now()-start).count()>max_seconds)
       throw std::runtime_error("Cohomology time budget exceeded");
    U c=order[step],old=critical[c];
    int d=all_dims[old];
    if(d==1){
      uint64_t lo=offsets[c],hi=offsets[c+1];
      if(hi-lo==2){
        int32_t aa=lookup[data[lo]],bb=lookup[data[lo+1]];
        if(aa<0||bb<0)throw std::runtime_error("Edge has missing vertex");
        U a=root(static_cast<U>(aa)),b=root(static_cast<U>(bb));
        if(a!=b){parent[a]=b;continue;}
      }else if(hi!=lo)throw std::runtime_error("Reduced edge has invalid 0-boundary");
      annotation[c]={c};reverse[c]={c};entries++;check_entries();
    }else if(d==2){
      V pivot_set;
      for(uint64_t j=offsets[c];j<offsets[c+1];j++){
        int32_t edge=lookup[data[j]];
        if(edge<0)throw std::runtime_error("Face has missing edge");
        pivot_set=symdiff(pivot_set,annotation[edge]);
      }
      if(pivot_set.empty())continue;
      U youngest=pivot_set[0];
      for(U item:pivot_set)if(rank[item]>rank[youngest])youngest=item;
      if(all_levels[critical[youngest]]<all_levels[old]){
        if(count>=max_output_pairs)throw std::runtime_error("H1 barcode output budget exceeded");
        U birth_id=critical[youngest];U death_id=old;
        int32_t birth=all_levels[birth_id],death=all_levels[death_id];
        write_u32(out,birth_id);write_u32(out,death_id);
        out.write(reinterpret_cast<const char*>(&birth),4);
        out.write(reinterpret_cast<const char*>(&death),4);
        count++;
      }
      V affected=reverse[youngest];
      for(U edge:affected){
        V prev=annotation[edge];V next=symdiff(prev,pivot_set);
        V touched=symdiff(prev,next);
        for(U gen:touched){
          if(contains(next,gen))add_val(reverse[gen],edge);
          else erase_val(reverse[gen],edge);
        }
        entries=entries-prev.size()+next.size();check_entries();
        annotation[edge]=std::move(next);
        ops+=prev.size()+pivot_set.size();
        if(ops>max_operations)throw std::runtime_error("Cohomology operation budget exceeded");
      }
      if(!reverse[youngest].empty())throw std::runtime_error("Annotation elimination failure");
    }
  }
  out.close();if(!out)throw std::runtime_error("Cannot write persistence pairs");
  *pair_count=count;*op_count=ops;*peak_entries=max_seen;
  return 0;
 }catch(const std::exception&e){
   if(err_size){std::strncpy(err,e.what(),err_size-1);err[err_size-1]='\0';}
   return 1;
 }
}

// On-demand *primal* reduction for ONE selected H1 interval. Keeps reduced
// columns and compact face-dependency lists, NOT dense change-of-basis chains.
// Replays only dependencies reachable from the requested destroyer face.
extern "C" int compact_h1_replay(
  const U* cell_ids,const int32_t* all_levels,const uint8_t* all_dims,
  uint64_t precursor_n,const U* critical,const uint64_t* offsets,
  const U* data,uint64_t n,U creator,U destroyer,uint64_t max_operations,
  uint64_t max_output_entries,double max_seconds,const char* cycle_path,
  const char* fill_path,uint64_t* op_count,char*err,uint64_t err_size) {
 try{
  using Clock=std::chrono::steady_clock;
  auto start=Clock::now();
  std::vector<int32_t> lookup(precursor_n,-1);
  for(U i=0;i<n;i++)lookup[critical[i]]=i;
  if(creator>=precursor_n||destroyer>=precursor_n||lookup[creator]<0||lookup[destroyer]<0)
    throw std::runtime_error("Pair creator/destroyer not critical cells");
  U target=static_cast<U>(lookup[destroyer]),wanted=static_cast<U>(lookup[creator]);
  std::vector<U> order(n),rank(n);std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](U a,U b){
    U x=critical[a],y=critical[b];
    return std::tie(all_levels[x],all_dims[x],cell_ids[x])<
           std::tie(all_levels[y],all_dims[y],cell_ids[y]);
  });
  for(U i=0;i<n;i++)rank[order[i]]=i;
  std::vector<int32_t> owner(n,-1);
  std::vector<V> reduced(n),dependencies(n);
  V selected;
  uint64_t ops=0;
  for(U step=0;step<n;step++){
    if((step&1023)==0&&std::chrono::duration<double>(Clock::now()-start).count()>max_seconds)
      throw std::runtime_error("Primal replay time budget exceeded");
    U face=order[step];
    if(all_dims[critical[face]]!=2)continue;
    V col;
    for(uint64_t t=offsets[face];t<offsets[face+1];t++){
      int32_t edge=lookup[data[t]];
      if(edge<0)throw std::runtime_error("Missing critical edge");
      col.push_back(static_cast<U>(edge));
    }
    std::sort(col.begin(),col.end());
    auto& deps=dependencies[face];
    while(!col.empty()){
      U pivot=*std::max_element(col.begin(),col.end(),[&](U a,U b){return rank[a]<rank[b];});
      int32_t prev=owner[pivot];
      if(prev<0)break;
      deps.push_back(static_cast<U>(prev));
      ops+=reduced[prev].size()+col.size();
      if(ops>max_operations)throw std::runtime_error("Primal replay operation budget exceeded");
      col=symdiff(col,reduced[prev]);
    }
    if(!col.empty()){
      U pivot=*std::max_element(col.begin(),col.end(),[&](U a,U b){return rank[a]<rank[b];});
      if(owner[pivot]>=0)throw std::runtime_error("Pivot already owned");
      owner[pivot]=face;reduced[face]=col;
    }
    if(face==target){
      if(col.empty())throw std::runtime_error("Target does not create an H1 death");
      U pivot=*std::max_element(col.begin(),col.end(),[&](U a,U b){return rank[a]<rank[b];});
      if(pivot!=wanted)throw std::runtime_error("Cohomology pair does not match primal pivot");
      selected=std::move(col);break;
    }
  }
  if(selected.empty())throw std::runtime_error("Target face not reached");
  // Expand only reachable change-of-basis dependencies. They always precede
  // their dependent face in filtration order, so topological evaluation is safe.
  std::vector<uint8_t> reachable(n,0);
  std::vector<U> queue{target};
  while(!queue.empty()){
    U current=queue.back();queue.pop_back();
    if(reachable[current])continue;
    reachable[current]=1;
    for(U dep:dependencies[current]){
      if(rank[dep]>=rank[current])throw std::runtime_error("Forward primal dependency");
      queue.push_back(dep);
      ops++;
      if(ops>max_operations)throw std::runtime_error("Primal dependency budget exceeded");
    }
  }
  std::vector<V> expanded(n);
  uint64_t stored_entries=0;
  for(U step=0;step<=rank[target];step++){
    U face=order[step];if(!reachable[face])continue;
    V fill{face};
    for(U dep:dependencies[face]){
      fill=symdiff(fill,expanded[dep]);
      ops+=fill.size()+expanded[dep].size();
      if(ops>max_operations)throw std::runtime_error("Primal fill expansion budget exceeded");
    }
    stored_entries+=fill.size();
    if(stored_entries>max_output_entries)throw std::runtime_error("Primal fill memory budget exceeded");
    expanded[face]=std::move(fill);
  }
  V fill=std::move(expanded[target]);
  std::ofstream cs(cycle_path,std::ios::binary|std::ios::trunc),fs(fill_path,std::ios::binary|std::ios::trunc);
  if(!cs||!fs)throw std::runtime_error("Cannot open replay output");
  for(U x:selected)write_u32(cs,critical[x]);
  for(U x:fill)write_u32(fs,critical[x]);
  cs.close();fs.close();
  if(!cs||!fs)throw std::runtime_error("Cannot write replay output");
  *op_count=ops;
  return 0;
 }catch(const std::exception&e){
   if(err_size){std::strncpy(err,e.what(),err_size-1);err[err_size-1]='\0';}
   return 1;
 }
}
