"""Native sparse algebra with on-disk deferred chain contractions.

C++17 is compiled into a *user-cache* shared object only when the native
backend is invoked. Source wheels remain portable; no binaries are bundled.
Compile failure is explicit, never silently replaced by a different algorithm.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import json
import numpy as np
from ._csr import prepare_complex

class _Stats(ctypes.Structure):
    _fields_=[(key,ctypes.c_uint64) for key in ('remaining','pairs','operations','trace_bytes','nonfree','tile_pairs','hit_budget')]

_build_lock=threading.Lock()

def build_native(*,cache_dir=None):
    source=Path(__file__).parent/'native'/'morse.cpp'
    if not source.exists():raise FileNotFoundError(f'Native C++ source missing: {source}')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache=Path(cache_dir or Path(tempfile.gettempdir())/'compact_morse_native')
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'morse_{digest}.so'
    with _build_lock:
        if not path.exists():
            compiler=shutil.which(os.environ.get('CXX','g++'))
            if not compiler:raise RuntimeError('C++17 compiler required for native sparse reduction')
            tmp=cache/(path.name+'.tmp')
            result=subprocess.run([compiler,'-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(tmp)],capture_output=True,text=True,timeout=180)
            if result.returncode:
                raise RuntimeError(f'C++ compilation failed: {result.stderr[:2000]}')
            os.replace(tmp,path)
    dll=ctypes.CDLL(str(path))
    fn=dll.compact_morse_run
    fn.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint8),
                 ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint64,
                 ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,
                 ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,
                 ctypes.c_double,ctypes.c_char_p,ctypes.POINTER(_Stats),ctypes.c_char_p,ctypes.c_uint64]
    fn.restype=ctypes.c_int
    fn._owner_dll=dll
    return fn

class CompactMorse:
    def __init__(self,volume,output_dir,*,pre_collapse=True,block_voxels=8,
                 max_cells=3_000_000,max_operations=8_000_000,max_fill=128,
                 max_heap_entries=8_000_000,max_trace_bytes=300_000_000,
                 max_pairs=10_000_000,max_seconds=90.,local_first=False,
                 indexed_scheduler=False):
        import time
        t=time.perf_counter()
        self.path=Path(output_dir).resolve();self.path.mkdir(parents=True,exist_ok=True)
        for name,v in [('max_operations',max_operations),('max_fill',max_fill),('max_heap_entries',max_heap_entries),('max_trace_bytes',max_trace_bytes),('max_pairs',max_pairs)]:
            if not isinstance(v,int) or v<=0:raise ValueError(f'{name} must be a positive integer')
        if max_seconds<=0:raise ValueError('max_seconds must be positive')
        prep=prepare_complex(volume,pre_collapse=pre_collapse,block_voxels=block_voxels,max_cells=max_cells)
        self.shape=prep['shape'];self.pre_stats=prep['stats']
        for name in ('ids','levels','dims','offsets','data'):
            np.save(self.path/(name+'.npy'),prep[name],allow_pickle=False)
        stats=_Stats();err=ctypes.create_string_buffer(512)
        fn=build_native()
        def ptr(key,kind):return prep[key].ctypes.data_as(ctypes.POINTER(kind))
        code=fn(ptr('ids',ctypes.c_uint32),ptr('levels',ctypes.c_int32),ptr('dims',ctypes.c_uint8),
                ptr('offsets',ctypes.c_uint64),ptr('data',ctypes.c_uint32),len(prep['ids']),
                self.shape[1],self.shape[2],block_voxels,int(local_first),int(indexed_scheduler),max_operations,
                max_fill,max_heap_entries,max_trace_bytes,max_pairs,float(max_seconds),
                str(self.path).encode(),ctypes.byref(stats),err,512)
        if code:raise RuntimeError('Native Morse: '+err.value.decode())
        self.stats={key:int(getattr(stats,key)) for key,_ in _Stats._fields_}
        self.stats.update(self.pre_stats);self.stats['seconds']=time.perf_counter()-t
        self.stats['indexed_scheduler']=bool(indexed_scheduler)
        (self.path/'metadata.json').write_text(json.dumps({'shape': self.shape, 'stats': self.stats}), encoding='utf8')
        self._init_views()

    @classmethod
    def open(cls,output_dir):
        obj=cls.__new__(cls);obj.path=Path(output_dir).resolve()
        meta=json.loads((obj.path/'metadata.json').read_text(encoding='utf8'))
        obj.shape=tuple(meta['shape']);obj.stats=meta['stats']
        obj._init_views()
        return obj

    def _init_views(self):
        self.ids=np.load(self.path/'ids.npy',mmap_mode='r')
        self.levels=np.load(self.path/'levels.npy',mmap_mode='r')
        self.dims=np.load(self.path/'dims.npy',mmap_mode='r')
        self.offsets=np.load(self.path/'offsets.npy',mmap_mode='r')
        self.data=np.load(self.path/'data.npy',mmap_mode='r')
        self.critical=np.fromfile(self.path/'critical_index.bin',dtype='<u4')
        self.critical_offsets=np.fromfile(self.path/'critical_offsets.bin',dtype='<u8')
        self.critical_data=np.memmap(self.path/'critical_data.bin',dtype='<u4',mode='r') if (self.path/'critical_data.bin').stat().st_size else np.empty(0,dtype=np.uint32)
        self.trace_offsets=np.fromfile(self.path/'trace_offsets.bin',dtype='<u8')
        self.trace=np.memmap(self.path/'trace.bin',dtype='<u4',mode='r') if len(self.trace_offsets) else np.empty(0,dtype=np.uint32)
        self._critical_set=None

    @property
    def critical_packed_ids(self):return self.ids[self.critical]

    def columns(self):
        """Materialize ONLY the critical complex for ordinary persistence solvers."""
        out={}
        for j,c in enumerate(self.critical):
            out[int(self.ids[c])]=set(int(self.ids[t]) for t in self.critical_data[int(self.critical_offsets[j]):int(self.critical_offsets[j+1])])
        return out

    def _index_chain(self,chain):
        raw=sorted(int(x) for x in chain)
        if any(x<0 or x>np.iinfo(np.uint32).max for x in raw):raise ValueError('Cell ID must be a nonnegative uint32')
        ids=np.asarray(raw,dtype=np.uint32)
        positions=np.searchsorted(self.ids,ids)
        if np.any(positions>=len(self.ids)) or not np.array_equal(self.ids[positions],ids):
            raise ValueError('Chain contains cell outside precursor complex')
        return set(int(x) for x in positions)

    def lift(self,chain):
        """Deferred filtered inclusion from reduced cells to precursor cells.

        Input and output are ORIGINAL packed cubical cell IDs. A replay scans
        the binary cancellation log in reverse; no dense chain map is stored.
        """
        if self._critical_set is None:self._critical_set=set(map(int,self.critical))
        current=self._index_chain(chain)
        if not current.issubset(self._critical_set):raise ValueError('Input must use only critical cells')
        for off in self.trace_offsets[::-1]:
            k=int(off)//4;a=int(self.trace[k]);b=int(self.trace[k+1]);n=int(self.trace[k+2]);m=int(self.trace[k+3]);
            sources=self.trace[k+4+n:k+4+n+m]
            parity=0
            for s in sources:parity^=(int(s) in current)
            if parity:
                if b in current:current.remove(b)
                else:current.add(b)
        return set(int(self.ids[j]) for j in current)

    def project(self,chain):
        current=self._index_chain(chain)
        for off in self.trace_offsets:
            k=int(off)//4;a=int(self.trace[k]);b=int(self.trace[k+1]);n=int(self.trace[k+2]);
            if a in current:
                current.remove(a)
                for j in self.trace[k+4:k+4+n]:
                    x=int(j)
                    if x in current:current.remove(x)
                    else:current.add(x)
            current.discard(b)
        return set(int(self.ids[j]) for j in current)

    def boundary(self,chain,*,initial=False):
        """Boundary in precursor or Morse complex, using sparse CSR on disk."""
        if initial:
            current=self._index_chain(chain)
            offsets,data=self.offsets,self.data
            lookup=None
        else:
            pids=self.critical_packed_ids
            positions=np.searchsorted(pids,np.asarray(sorted(chain),dtype=np.uint32))
            vals=np.asarray(sorted(chain),dtype=np.uint32)
            if np.any(positions>=len(pids)) or not np.array_equal(pids[positions],vals):raise ValueError('Noncritical chain')
            current=set(int(self.critical[pos]) for pos in positions)
            offsets,data=self.critical_offsets,self.critical_data
            lookup={int(c):i for i,c in enumerate(self.critical)}
        out=set()
        for c in current:
            index=c if initial else lookup[c]
            for j in data[int(offsets[index]):int(offsets[index+1])]:
                packed=int(self.ids[int(j)])
                if packed in out:out.remove(packed)
                else:out.add(packed)
        return out

    def validate_chain_map(self,*,sample=30):
        """Verify critical d²=0 and lifting commutes with boundary."""
        for c in self.critical_packed_ids[:sample]:
            cell={int(c)}
            assert not self.boundary(self.boundary(cell))
            assert self.boundary(self.lift(cell),initial=True)==self.lift(self.boundary(cell))
            assert self.project(self.lift(cell))==cell
            v=int(c);ix=int(np.searchsorted(self.ids,v))
            assert all(self.levels[int(np.searchsorted(self.ids,x))]<=self.levels[ix] for x in self.lift(cell))
        return True

    _H1_RECORD_DTYPE = np.dtype([
        ('creator', '<u4'), ('destroyer', '<u4'),
        ('birth', '<i4'), ('death', '<i4'),
    ])
    _H1_INDEX_DTYPE = np.dtype([('creator', '<u4'), ('row', '<u4')])
    _H1_INPUT_FILES = ('ids.npy', 'levels.npy', 'dims.npy',
                       'critical_index.bin', 'critical_offsets.bin',
                       'critical_data.bin')

    def _h1_inputs_signature(self):
        """Detect changed catalog inputs before reusing saved pair records.

        Source files belong to this engine's immutable catalog directory. The
        size/mtime signature guards accidental edits, but is not a cryptographic
        content hash or a concurrency/hostile-mutation protocol.
        """
        return [[name, (self.path / name).stat().st_size,
                 (self.path / name).stat().st_mtime_ns]
                for name in self._H1_INPUT_FILES]

    def _cached_h1_records(self):
        """Return read-only mapped pairs iff a *completed* matching run exists."""
        meta_path = self.path / 'h1_pairs_metadata.json'
        records_path = self.path / 'h1_pairs.bin'
        if not meta_path.is_file() or not records_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            if meta.get('format') != 1 or meta.get('signature') != self._h1_inputs_signature():
                return None
            if records_path.stat().st_size != int(meta['count']) * self._H1_RECORD_DTYPE.itemsize:
                return None
            if not meta['count']:
                return np.empty(0, dtype=self._H1_RECORD_DTYPE)
            return np.memmap(records_path, dtype=self._H1_RECORD_DTYPE, mode='r')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _make_h1_index(self, records):
        """Index finite H1 creators in O(B log B) preprocessing, O(B) disk.

        Records themselves remain in a mmap; exact pair reconstruction then
        needs no full Python tuple list or repeat cohomology reduction.
        """
        index_path = self.path / 'h1_creators.npy'
        if index_path.exists():
            try:
                index = np.load(index_path, mmap_mode='r', allow_pickle=False)
                # Constant-cost validation on reopen. The index is atomically
                # created by this engine and removed whenever the H1 catalog
                # is regenerated; we deliberately avoid scanning B rows for
                # every selected-pair lookup.
                if index.dtype == self._H1_INDEX_DTYPE and len(index) == len(records):
                    if len(index):
                        positions = sorted(set((0, len(index)//2, len(index)-1)))
                        keys = index['creator'][positions]
                        rows = index['row'][positions]
                        if (np.any(rows >= len(records)) or
                            np.any(keys[1:] <= keys[:-1]) or
                            not np.array_equal(self.ids[records['creator'][rows]], keys)):
                            raise ValueError('H1 index failed integrity spot-check')
                    return index
            except (OSError, ValueError):
                pass
        if len(records) > np.iinfo(np.uint32).max:
            raise OverflowError('Too many H1 pairs for uint32 row IDs')
        order = np.argsort(self.ids[records['creator']], kind='stable')
        creators = self.ids[records['creator'][order]]
        if len(creators) > 1 and np.any(creators[1:] == creators[:-1]):
            raise AssertionError('A persistence creator is paired more than once')
        index = np.empty(len(records), dtype=self._H1_INDEX_DTYPE)
        index['creator'] = creators
        index['row'] = order.astype(np.uint32)
        # Same-directory atomic replacement prevents partially written indexes.
        temp_path = self.path / 'h1_creators.build.npy'
        np.save(temp_path, index, allow_pickle=False)
        os.replace(temp_path, index_path)
        return np.load(index_path, mmap_mode='r', allow_pickle=False)

    def h1_pair(self, pair_id, *, max_operations=100_000_000,
                max_seconds=120.):
        """Find one previously computed pair in O(log B) indexed operations.

        Returned tuple is (pair_id, birth, death, creator_packed,
        destroyer_packed). Reuses a durable H1 catalog if present; otherwise
        computes it once. No list of all B Python pair objects is needed.
        """
        if not isinstance(pair_id, str):
            raise ValueError('Expected a string H1 pair ID')
        parts = pair_id.split(':')
        if len(parts) != 2 or not all(x.isdecimal() for x in parts):
            raise ValueError('Malformed H1 pair ID')
        creator_packed, destroyer_packed = map(int, parts)
        if creator_packed > np.iinfo(np.uint32).max or destroyer_packed > np.iinfo(np.uint32).max:
            raise ValueError('H1 pair ID outside uint32 cell range')
        records = self._cached_h1_records()
        if records is None:
            self.h1_pairs(max_operations=max_operations, max_seconds=max_seconds)
            records = self._cached_h1_records()
        if records is None:
            raise RuntimeError('H1 pair catalog was not written correctly')
        index = self._make_h1_index(records)
        creator_pos = int(np.searchsorted(index['creator'], creator_packed))
        if creator_pos >= len(index) or int(index['creator'][creator_pos]) != creator_packed:
            raise ValueError('Requested pair ID not in native H1 catalog')
        row = records[int(index['row'][creator_pos])]
        c_idx, d_idx = int(row['creator']), int(row['destroyer'])
        if (int(self.ids[c_idx]) != creator_packed or
                int(self.ids[d_idx]) != destroyer_packed):
            raise ValueError('Requested pair ID not in native H1 catalog')
        return (pair_id, int(row['birth']), int(row['death']),
                creator_packed, destroyer_packed)

    def h1_records(self, *, max_operations=100_000_000, max_seconds=120.,
                   max_annotation_entries=50_000_000,
                   max_output_pairs=10_000_000,
                   use_cache=True):
        """Native H1 barcode, cached durably for later class-aware replay.

        The work/time budgets apply on a fresh computation; a complete cached
        result needs no recomputation. `use_cache=False` forces an exact rerun.
        """
        if (not isinstance(max_annotation_entries, int) or max_annotation_entries <= 0 or
                not isinstance(max_output_pairs, int) or max_output_pairs <= 0):
            raise ValueError('Positive cohomology annotation and barcode budgets required')
        records = self._cached_h1_records() if use_cache else None
        if records is None:
            exported = build_native()
            native = exported._owner_dll
            run = native.compact_h1_run
            run.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32),
                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
                            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64),
                            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64,
                            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_char_p,
                            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                            ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_uint64]
            run.restype = ctypes.c_int
            def ptr(arr, type_):
                return np.asarray(arr).ctypes.data_as(ctypes.POINTER(type_))
            count = ctypes.c_uint64()
            operations = ctypes.c_uint64()
            peak_entries = ctypes.c_uint64()
            err = ctypes.create_string_buffer(512)
            # Only the complete metadata establishes validity. An interrupted
            # run cannot accidentally turn a truncated output into a cache hit.
            meta_path = self.path / 'h1_pairs_metadata.json'
            meta_path.unlink(missing_ok=True)
            target = self.path / 'h1_pairs.bin'
            code = run(ptr(self.ids, ctypes.c_uint32),
                       ptr(self.levels, ctypes.c_int32),
                       ptr(self.dims, ctypes.c_uint8), len(self.ids),
                       ptr(self.critical, ctypes.c_uint32),
                       ptr(self.critical_offsets, ctypes.c_uint64),
                       ptr(self.critical_data, ctypes.c_uint32),
                       len(self.critical), int(max_operations), int(max_annotation_entries), int(max_output_pairs), float(max_seconds),
                       str(target).encode(), ctypes.byref(count),
                       ctypes.byref(operations), ctypes.byref(peak_entries), err, 512)
            if code:
                raise RuntimeError('Native H1: ' + err.value.decode())
            if target.stat().st_size != count.value * self._H1_RECORD_DTYPE.itemsize:
                raise RuntimeError('Native H1 output length mismatch')
            metadata = dict(format=1, count=int(count.value),
                            signature=self._h1_inputs_signature(),
                            operations=int(operations.value),
                            peak_annotation_entries=int(peak_entries.value))
            temp_meta = self.path / 'h1_pairs_metadata.build.json'
            temp_meta.write_text(json.dumps(metadata), encoding='utf8')
            os.replace(temp_meta, meta_path)
            (self.path / 'h1_creators.npy').unlink(missing_ok=True)
            records = self._cached_h1_records()
            if records is None:
                raise RuntimeError('Completed H1 catalog failed verification')
        # Build the O(B)-disk lookup index lazily: barcode-only consumers
        # never pay the sorting/memory cost of representative selection.
        return records

    def h1_pair_at(self, index):
        """Return a pair by catalog row without building all Python IDs."""
        records = self.h1_records()
        if not 0 <= index < len(records):
            raise IndexError('H1 pair catalog row out of range')
        r = records[index]
        creator = int(self.ids[r['creator']])
        destroyer = int(self.ids[r['destroyer']])
        return (f'{creator}:{destroyer}', int(r['birth']), int(r['death']),
                creator, destroyer)

    def h1_pairs(self, *, max_operations=100_000_000, max_seconds=120.,
                 use_cache=True):
        """Compatibility API: materialize Python tuples only if requested."""
        records = self.h1_records(max_operations=max_operations,
                                  max_seconds=max_seconds, use_cache=use_cache)
        return [(f'{int(self.ids[r["creator"]])}:{int(self.ids[r["destroyer"]])}',
                 int(r['birth']), int(r['death']),
                 int(self.ids[r['creator']]), int(self.ids[r['destroyer']]))
                for r in records]


class _IndexLevels:
    """Lazy lookup of precursor filtration values; avoids a huge Python dict."""
    def __init__(self,engine):self.engine=engine
    def __getitem__(self,key):
        k=int(np.searchsorted(self.engine.ids,key))
        if k>=len(self.engine.ids) or int(self.engine.ids[k])!=key:raise KeyError(key)
        return int(self.engine.levels[k])

class _PrimalAdapter:
    """Materialize only critical columns for rare selected-loop primal replay."""
    def __init__(self,engine):
        self.engine=engine
        self.columns=engine.columns()
        self.level={int(engine.ids[t]):int(engine.levels[t]) for t in engine.critical}
        self.dim={int(engine.ids[t]):int(engine.dims[t]) for t in engine.critical}
        self.initial_levels=_IndexLevels(engine)
    def boundary(self,chain,original=False):
        return self.engine.boundary(chain,initial=original)
    def lift(self,chain):return self.engine.lift(chain)


def replay_selected_h1(engine,pair_id,*,max_critical_cells=200_000,max_operations=2_000_000):
    """Certify a selected pair: primal loop, death chain, original coordinates.

    The native H1 catalog is authoritative; reverse the native cancellation
    trace only for a selected representative. Materialization is restricted to
    the smaller critical complex and guarded by max_critical_cells.
    """
    from .persistence import MorsePersistenceH1
    if len(engine.critical)>max_critical_cells:
        raise MemoryError('Critical complex too large for selective primal replay')
    all_ids={item[0] for item in engine.h1_pairs()}
    if pair_id not in all_ids:raise ValueError('Requested H1 pair is not in native catalog')
    p=MorsePersistenceH1(_PrimalAdapter(engine),max_operations=max_operations,
                        max_columns=max_critical_cells)
    pair=next((x for x in p.pairs if x.pair_id==pair_id),None)
    if pair is None:raise AssertionError('Native cohomology/primal pairing mismatch')
    return p.replay(pair,max_operations=max_operations)

CompactMorse.replay_h1=replay_selected_h1

def _deferred_homotopy(engine,chain,*,max_steps=30_000,max_work=2_000_000):
    """On-demand chain homotopy of one precursor chain.

    For large traces this may be expensive; no dense homotopy is cached.
    This is primarily a proof/certificate tool, not the default explanation path.
    """
    if len(engine.trace_offsets)>max_steps:raise MemoryError('Homotopy trace length exceeds budget')
    current=engine._index_chain(chain)
    out=set();operations=0
    for si,off in enumerate(engine.trace_offsets):
        k=int(off)//4;a=int(engine.trace[k]);b=int(engine.trace[k+1]);n=int(engine.trace[k+2]);
        if a in current:
            lifted={b}
            for earlier_off in engine.trace_offsets[:si][::-1]:
                j=int(earlier_off)//4;upper=int(engine.trace[j+1]);n0=int(engine.trace[j+2]);m0=int(engine.trace[j+3]);
                if sum(int(x) in lifted for x in engine.trace[j+4+n0:j+4+n0+m0])%2:
                    lifted.symmetric_difference_update({upper})
                operations+=1
                if operations>max_work:raise MemoryError('Homotopy work budget')
            out.symmetric_difference_update(lifted)
            current.remove(a)
            for index in engine.trace[k+4:k+4+n]:
                x=int(index)
                if x in current:current.remove(x)
                else:current.add(x)
        current.discard(b)
    return {int(engine.ids[x]) for x in out}

CompactMorse.homotopy=_deferred_homotopy

def replay_h1_native(engine,pair_id,*,max_operations=100_000_000,
                     max_output_entries=2_000_000,max_seconds=120.,
                     validate=True):
    """Primal target-only reduction, sparse dependency replay, disk trace lift.

    If validate, independently checks the lifted cycle and death filling chain
    against the original cubical boundary and filtration. Birth nonbounding
    follows from the verified filtration-ordered reduction pivot; the optional
    Python replay is available for a separate birth-time rank test.
    """
    if max_operations<=0 or max_output_entries<=0 or max_seconds<=0:
        raise ValueError('Positive replay work, output, and time budgets required')
    _,birth,death,creator,destroyer=engine.h1_pair(
        pair_id, max_operations=max_operations, max_seconds=max_seconds)
    exported=build_native();native=exported._owner_dll
    fn=native.compact_h1_replay
    fn.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint8),
                 ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint64),
                 ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
                 ctypes.c_uint64,ctypes.c_uint64,ctypes.c_double,ctypes.c_char_p,ctypes.c_char_p,
                 ctypes.POINTER(ctypes.c_uint64),ctypes.c_char_p,ctypes.c_uint64]
    fn.restype=ctypes.c_int
    def ptr(a,t):return np.asarray(a).ctypes.data_as(ctypes.POINTER(t))
    c_idx=int(np.searchsorted(engine.ids,creator));d_idx=int(np.searchsorted(engine.ids,destroyer))
    with tempfile.TemporaryDirectory(prefix='replay_',dir=engine.path) as directory:
        cycle_path=str(Path(directory)/'cycle.bin');fill_path=str(Path(directory)/'fill.bin')
        operations=ctypes.c_uint64();err=ctypes.create_string_buffer(512)
        code=fn(ptr(engine.ids,ctypes.c_uint32),ptr(engine.levels,ctypes.c_int32),
                ptr(engine.dims,ctypes.c_uint8),len(engine.ids),
                ptr(engine.critical,ctypes.c_uint32),ptr(engine.critical_offsets,ctypes.c_uint64),
                ptr(engine.critical_data,ctypes.c_uint32),len(engine.critical),
                c_idx,d_idx,int(max_operations),int(max_output_entries),float(max_seconds),
                cycle_path.encode(),fill_path.encode(),ctypes.byref(operations),err,512)
        if code:raise RuntimeError('Native H1 replay: '+err.value.decode())
        cycle=set(map(int,engine.ids[np.fromfile(cycle_path,dtype='<u4')]))
        filling=set(map(int,engine.ids[np.fromfile(fill_path,dtype='<u4')]))
    if validate:
        assert engine.boundary(filling)==cycle,'Morse chain filling does not match cycle'
        spatial_cycle=engine.lift(cycle)
        spatial_filling=engine.lift(filling)
        assert not engine.boundary(spatial_cycle,initial=True),'Lifted cycle not closed'
        assert engine.boundary(spatial_filling,initial=True)==spatial_cycle,'Lifted death filling invalid'
        assert all(int(engine.levels[np.searchsorted(engine.ids,x)])<=birth for x in spatial_cycle)
        assert all(int(engine.levels[np.searchsorted(engine.ids,x)])<=death for x in spatial_filling)
    else:
        spatial_cycle=engine.lift(cycle);spatial_filling=engine.lift(filling)
    return {'pair_id':pair_id,'birth':birth,'death':death,
            'morse_cycle':frozenset(cycle),'morse_filling':frozenset(filling),
            'cycle_cells':frozenset(spatial_cycle),'filling_cells':frozenset(spatial_filling),
            'verified':bool(validate),'birth_nonboundary_by_pivot':True,
            'replay_operations':int(operations.value)}

CompactMorse.replay_h1_native=replay_h1_native
