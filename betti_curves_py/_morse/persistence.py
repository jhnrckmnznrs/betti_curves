"""Sparse H1 persistent reduction on a filtered algebraic Morse complex.

Works on arbitrary F2 Morse boundary columns, not merely grid face boundaries.
Selective primal replay and explicit original-grid lift preserve class identity.
"""
from dataclasses import dataclass
def xor_into(target, other):
    target.symmetric_difference_update(other)

@dataclass(frozen=True)
class H1Pair:
    pair_id:str
    creator:int
    destroyer:int
    birth:int
    death:int


class MorsePersistenceH1:
    def __init__(self, morse, *, max_operations=2_000_000, max_columns=200_000):
        self.morse=morse
        self.ids=sorted(morse.columns,key=lambda c:(morse.level[c],morse.dim[c],c))
        if len(self.ids)>max_columns:raise MemoryError('Morse reduction columns budget')
        self.rank={cid:r for r,cid in enumerate(self.ids)}
        self.columns={cid:frozenset(morse.columns[cid]) for cid in self.ids}
        self.reduced={}; self.pivot_owner={}; self.pairs=[]
        self.operations=0
        # H1 finite pairs are precisely pivots of 2-cell boundaries in full reduction.
        for b in self.ids:
            if morse.dim[b]!=2:continue
            col=set(self.columns[b])
            while col:
                pivot=max(col,key=self.rank.__getitem__)
                owner=self.pivot_owner.get(pivot)
                if owner is None:break
                xor_into(col,self.reduced[owner]);self.operations+=len(self.reduced[owner])
                if self.operations>max_operations:raise MemoryError('Persistent reduction budget')
            if col:
                a=max(col,key=self.rank.__getitem__)
                self.pivot_owner[a]=b;self.reduced[b]=frozenset(col)
                if morse.level[a]<morse.level[b]:
                    self.pairs.append(H1Pair(f'{a}:{b}',a,b,morse.level[a],morse.level[b]))
            else:self.reduced[b]=frozenset()

    def select(self, rank=0):
        return sorted(self.pairs,key=lambda p:(p.death-p.birth,p.pair_id),reverse=True)[rank]

    def replay(self,pair:H1Pair, *,max_operations=2_000_000):
        """Return a chain certificate in reduced and original-grid coordinates."""
        if pair not in self.pairs:raise ValueError('Pair does not belong to this engine')
        col=set(self.columns[pair.destroyer]);fill={pair.destroyer};work={"operations":0};cache={}
        while col:
            pivot=max(col,key=self.rank.__getitem__)
            if pivot==pair.creator:break
            owner=self.pivot_owner.get(pivot)
            if owner is None or self.rank[owner]>=self.rank[pair.destroyer]:
                raise AssertionError('Incorrect pivot ownership for replay')
            # On-demand replay dependencies; recursive elimination if necessary.
            rc,rf=self._replay_earlier(owner,cache,max_operations,work)
            xor_into(col,rc);xor_into(fill,rf)
            work['operations']+=len(rc)+len(rf)
            if work['operations']>max_operations:raise MemoryError('Replay budget')
        if not col or max(col,key=self.rank.__getitem__)!=pair.creator:
            raise AssertionError('Pair-specific cycle not recovered')
        if self.morse.boundary(fill)!=col:raise AssertionError('Morse fill boundary incorrect')
        lift_cycle=self.morse.lift(col);lift_fill=self.morse.lift(fill)
        if self.morse.boundary(lift_fill,original=True)!=lift_cycle:
            raise AssertionError('Original geometric fill boundary incorrect')
        if self.morse.boundary(lift_cycle,original=True):
            raise AssertionError('Lifted loop not closed')
        if max((self.morse.initial_levels[i] for i in lift_cycle),default=-1)>pair.birth:
            raise AssertionError('Loop not yet born at claimed birth')
        if max((self.morse.initial_levels[i] for i in lift_fill),default=-1)>pair.death:
            raise AssertionError('Filling chain not available at death')
        if self.is_boundary_at_birth(col,pair.birth):
            raise AssertionError('Selected loop already bounds at birth')
        return {'pair_id':pair.pair_id,'birth':pair.birth,'death':pair.death,
                'morse_cycle':frozenset(col),'morse_filling':frozenset(fill),
                'cycle_cells':frozenset(lift_cycle),'filling_cells':frozenset(lift_fill),
                'verified':True,'replay_operations':work['operations']}

    def _replay_earlier(self, face,cache,max_operations,work):
        if face in cache:return cache[face]
        if len(cache)>max_operations:raise MemoryError('Replay cache budget')
        col=set(self.columns[face]);fill={face}
        while col:
            p=max(col,key=self.rank.__getitem__)
            owner=self.pivot_owner.get(p)
            if owner is None or owner==face:break
            if self.rank[owner]>=self.rank[face]:raise AssertionError('Forward dependence')
            reduced,filling=self._replay_earlier(owner,cache,max_operations,work)
            xor_into(col,reduced);xor_into(fill,filling)
            work['operations']+=len(reduced)+len(filling)
            if work['operations']>max_operations:raise MemoryError('Replay recursion budget')
        cache[face]=(frozenset(col),frozenset(fill))
        return cache[face]

    def is_boundary_at_birth(self,cycle,threshold):
        """Rank test in the Morse complex at threshold (equivalent to original)."""
        base={}
        for face in self.ids:
            if self.morse.level[face]>threshold:break
            if self.morse.dim[face]!=2:continue
            col=set(self.columns[face])
            while col:
                p=max(col,key=self.rank.__getitem__)
                if p not in base:
                    base[p]=frozenset(col);break
                xor_into(col,base[p])
        col=set(cycle)
        while col:
            p=max(col,key=self.rank.__getitem__)
            if p not in base:return False
            xor_into(col,base[p])
        return True


class MorseCohomologyH1:
    """Independent sparse-annotation cohomology on an arbitrary Morse differential."""
    def __init__(self,morse,*,max_operations=5_000_000):
        ids=sorted(morse.columns,key=lambda c:(morse.level[c],morse.dim[c],c))
        parents={cid:cid for cid in ids if morse.dim[cid]==0}
        def root(x):
            while parents[x]!=x:
                parents[x]=parents[parents[x]];x=parents[x]
            return x
        annotations={};incidence={};self.pairs=[];ops=0
        for cid in ids:
            d=morse.dim[cid]
            if d==1:
                faces=morse.columns[cid]
                if len(faces)==2:
                    a,b=tuple(faces);x=root(a);y=root(b)
                    if x!=y:
                        parents[x]=y;continue
                elif faces:raise AssertionError('An edge has invalid 0-boundary')
                annotations[cid]={cid};incidence[cid]={cid}
            elif d==2:
                pivot_set=set()
                for edge in morse.columns[cid]:
                    xor_into(pivot_set,annotations.get(edge,()))
                if not pivot_set:continue
                youngest=max(pivot_set,key=lambda x:(morse.level[x], x))
                if morse.level[youngest]<morse.level[cid]:
                    self.pairs.append(H1Pair(f'{youngest}:{cid}',youngest,cid,morse.level[youngest],morse.level[cid]))
                affected=tuple(incidence[youngest])
                for edge in affected:
                    old=annotations[edge];new=old.symmetric_difference(pivot_set)
                    for g in old-new:incidence[g].remove(edge)
                    for g in new-old:incidence[g].add(edge)
                    if new:annotations[edge]=new
                    else:del annotations[edge]
                    ops+=len(old)+len(pivot_set)
                    if ops>max_operations:raise MemoryError('Cohomology budget exceeded')
                if incidence[youngest]:raise AssertionError('Cohomology elimination failure')
                del incidence[youngest]
        self.operations=ops
