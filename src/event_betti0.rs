use anyhow::Result;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::atomic_output::AtomicOutput;
use crate::connectivity::Connectivity;
use crate::interface_sparsify::{InterfaceFiltration, sparsify_cross_interface};
use crate::io::{Block, TiffStackReader};
use crate::local_pruning::NeighborhoodComponentPruner;
use crate::slab_interface::{
    NO_INTERFACE_REP, face_node_id, interface_node_count as slab_interface_node_count,
    local_boundary_node_id,
};

const NUM_U16_VALUES: usize = 65_536;

#[derive(Debug, Clone, Copy)]
pub struct Betti0Event {
    pub value: u16,
    pub delta: i64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InterfaceMergeEvent {
    pub(crate) value: u16,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

#[derive(Debug)]
pub(crate) struct BoundaryFaceNodes {
    pub(crate) width: usize,
    pub(crate) height: usize,

    /// One interface node id per face voxel.
    pub(crate) node_ids: Vec<u32>,

    /// Original grayscale value of the face voxel.
    pub(crate) values: Vec<u16>,
}

#[derive(Debug)]
pub(crate) struct SlabEventSummary {
    pub(crate) slab_id: usize,

    /// Local Betti-0 changes inside this slab.
    pub(crate) local_betti0_events: Vec<Betti0Event>,

    /// Number of interface nodes owned by this slab.
    pub(crate) interface_node_count: u32,

    /// Local equivalence events between interface nodes.
    ///
    /// These do NOT change Betti-0 globally by themselves.
    /// They only tell the global reducer that two boundary nodes
    /// became connected inside the slab at this filtration value.
    pub(crate) interface_merge_events: Vec<InterfaceMergeEvent>,

    pub(crate) z_min_face: BoundaryFaceNodes,
    pub(crate) z_max_face: BoundaryFaceNodes,
}

#[derive(Debug)]
struct VoxelBuckets {
    /// `offsets[v]..offsets[v + 1]` gives voxel indices with value `v`.
    offsets: Vec<usize>,

    /// Voxel indices sorted by voxel value.
    indices: Vec<u32>,
}

fn build_voxel_buckets_u16(values: &[u16]) -> VoxelBuckets {
    let mut counts = vec![0usize; NUM_U16_VALUES];

    for &v in values {
        counts[v as usize] += 1;
    }

    let mut offsets = vec![0usize; NUM_U16_VALUES + 1];

    for v in 0..NUM_U16_VALUES {
        offsets[v + 1] = offsets[v] + counts[v];
    }

    let mut cursor = offsets.clone();
    let mut indices = vec![0u32; values.len()];

    for (idx, &v) in values.iter().enumerate() {
        let pos = cursor[v as usize];
        indices[pos] = idx as u32;
        cursor[v as usize] += 1;
    }

    VoxelBuckets { offsets, indices }
}

/// Union-find for local slab processing.
///
/// It also stores one representative interface node per local component.
/// When two components merge and both have interface representatives,
/// we emit an interface merge event.
#[derive(Debug)]
struct LocalEventUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,

    /// For each root, an interface representative or NO_INTERFACE_REP.
    interface_rep: Vec<u32>,
}

impl LocalEventUnionFind {
    fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "LocalEventUnionFind uses u32 indices; slab has too many voxels"
        );

        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];
        let interface_rep = vec![NO_INTERFACE_REP; n];

        Self {
            parent,
            rank,
            interface_rep,
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = p;
        }
        x
    }

    fn set_interface_rep(&mut self, x: u32, rep: u32) {
        let root = self.find(x);
        self.interface_rep[root as usize] = rep;
    }

    /// Returns:
    /// - `None` if already in same component
    /// - `Some((new_root, maybe_interface_merge))` if merged
    fn union_with_interface_event(
        &mut self,
        a: u32,
        b: u32,
        value: u16,
    ) -> Option<(u32, Option<InterfaceMergeEvent>)> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return None;
        }

        let rep_a = self.interface_rep[ra as usize];
        let rep_b = self.interface_rep[rb as usize];

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        let merged_rep = match (rep_a != NO_INTERFACE_REP, rep_b != NO_INTERFACE_REP) {
            (true, true) => {
                self.interface_rep[ra as usize] = rep_a;

                if rep_a != rep_b {
                    Some(InterfaceMergeEvent {
                        value,
                        a: rep_a,
                        b: rep_b,
                    })
                } else {
                    None
                }
            }
            (true, false) => {
                self.interface_rep[ra as usize] = rep_a;
                None
            }
            (false, true) => {
                self.interface_rep[ra as usize] = rep_b;
                None
            }
            (false, false) => {
                self.interface_rep[ra as usize] = NO_INTERFACE_REP;
                None
            }
        };

        Some((ra, merged_rep))
    }
}

fn union_active_neighbor_event(
    uf: &mut LocalEventUnionFind,
    active: &[u8],
    idx_u32: u32,
    neighbor: usize,
    value: u16,
    delta: &mut i64,
    interface_merge_events: &mut Vec<InterfaceMergeEvent>,
) {
    if active[neighbor] != 0
        && let Some((_new_root, maybe_interface_event)) =
            uf.union_with_interface_event(idx_u32, neighbor as u32, value)
    {
        *delta -= 1;

        if let Some(event) = maybe_interface_event {
            interface_merge_events.push(event);
        }
    }
}

/// Union-find used by the global reducer over interface nodes.
#[derive(Debug)]
pub(crate) struct GlobalInterfaceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl GlobalInterfaceUnionFind {
    pub(crate) fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "GlobalInterfaceUnionFind uses u32 indices; too many interface nodes"
        );

        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];

        Self { parent, rank }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = p;
        }
        x
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return false;
        }

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        true
    }
}

/// Process one slab once, event-based.
///
/// This computes local Betti-0 events and boundary interface events.
/// It does not yet do global slab reconciliation.
pub(crate) fn process_slab_event_based_betti0(
    slab_id: usize,
    block: &Block,
    connectivity: Connectivity,
) -> SlabEventSummary {
    let n = block.voxel_count();

    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalEventUnionFind::new(n);
    let mut active = vec![0u8; n];

    let buckets = build_voxel_buckets_u16(&block.values);
    let mut local_pruner = NeighborhoodComponentPruner::new(connectivity);

    let interface_node_count = u32::try_from(slab_interface_node_count(slice_size, depth))
        .expect("slab interface node count exceeds u32");

    let mut local_betti0_events: Vec<Betti0Event> = Vec::new();
    let mut interface_merge_events: Vec<InterfaceMergeEvent> = Vec::new();

    for value in 0..NUM_U16_VALUES {
        let start = buckets.offsets[value];
        let end = buckets.offsets[value + 1];

        if start == end {
            continue;
        }

        let value_u16 = value as u16;
        let mut delta = 0i64;

        for pos in start..end {
            let idx_u32 = buckets.indices[pos];
            let idx = idx_u32 as usize;

            active[idx] = 1;
            delta += 1;

            let face_index = idx % slice_size;
            let z = idx / slice_size;
            if let Some(interface_node) = local_boundary_node_id(z, depth, face_index, slice_size) {
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let x = idx % width;
            let y = (idx / width) % height;

            local_pruner.for_each_representative_neighbor(
                &active,
                x,
                y,
                z,
                width,
                height,
                depth,
                |neighbor| {
                    union_active_neighbor_event(
                        &mut uf,
                        &active,
                        idx_u32,
                        neighbor,
                        value_u16,
                        &mut delta,
                        &mut interface_merge_events,
                    );
                },
            );
        }

        if delta != 0 {
            local_betti0_events.push(Betti0Event {
                value: value_u16,
                delta,
            });
        }
    }

    let z_min_face = extract_boundary_face_nodes(block, 0);
    let z_max_face = extract_boundary_face_nodes(block, depth - 1);

    SlabEventSummary {
        slab_id,
        local_betti0_events,
        interface_node_count,
        interface_merge_events,
        z_min_face,
        z_max_face,
    }
}

fn extract_boundary_face_nodes(block: &Block, z_local: usize) -> BoundaryFaceNodes {
    let width = block.shape[0];
    let height = block.shape[1];
    let slice_size = width * height;

    let mut node_ids = vec![0u32; width * height];
    let mut values = vec![0u16; width * height];

    for y in 0..height {
        for x in 0..width {
            let face_idx = y * width + x;
            let idx = z_local * slice_size + y * width + x;

            node_ids[face_idx] = face_node_id(z_local, block.shape[2], face_idx, slice_size);
            values[face_idx] = block.values[idx];
        }
    }

    BoundaryFaceNodes {
        width,
        height,
        node_ids,
        values,
    }
}

fn make_slab_ranges(depth: usize, slab_depth: usize) -> Vec<(usize, usize, usize)> {
    let mut ranges = Vec::new();

    let mut slab_id = 0usize;
    let mut z0 = 0usize;

    while z0 < depth {
        let z1 = z0.saturating_add(slab_depth).min(depth);
        ranges.push((slab_id, z0, z1));

        z0 = z1;
        slab_id += 1;
    }

    ranges
}

fn process_all_slabs_event_based_betti0(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<SlabEventSummary>> {
    let ranges = make_slab_ranges(volume.depth, slab_depth);

    println!(
        "Processing {} Betti-0 slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabEventSummary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;
            Ok(process_slab_event_based_betti0(
                slab_id,
                &block,
                connectivity,
            ))
        })
        .collect();

    let mut summaries = parallel_results
        .into_iter()
        .collect::<Result<Vec<SlabEventSummary>>>()?;

    summaries.sort_by_key(|summary| summary.slab_id);
    Ok(summaries)
}

fn build_interface_offsets(summaries: &[SlabEventSummary]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(summaries.len() + 1);
    let mut current = 0u32;
    offsets.push(current);

    for summary in summaries {
        current = current
            .checked_add(summary.interface_node_count)
            .expect("too many interface nodes for u32 global ids");
        offsets.push(current);
    }

    offsets
}

fn global_node_id(interface_offsets: &[u32], slab_id: usize, local_node_id: u32) -> u32 {
    interface_offsets[slab_id] + local_node_id
}

/// Reduce slab summaries into the global sparse Betti-0 curve.
///
/// Local Betti-0 events contribute directly to the Betti count.
/// Local interface merge events only update the global interface UF.
/// Cross-slab edges update the UF and contribute -1 to Betti-0 if successful.
fn reduce_event_based_betti0(
    summaries: &[SlabEventSummary],
    connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let interface_offsets = build_interface_offsets(summaries);
    let total_interface_nodes = *interface_offsets.last().unwrap_or(&0) as usize;

    let mut global_uf = GlobalInterfaceUnionFind::new(total_interface_nodes);
    let mut local_delta_by_value = vec![0i64; NUM_U16_VALUES];
    let mut interface_merges_by_value: Vec<Vec<(u32, u32)>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();
    let mut cross_edges_by_value: Vec<Vec<(u32, u32)>> =
        (0..NUM_U16_VALUES).map(|_| Vec::new()).collect();

    for summary in summaries {
        for event in &summary.local_betti0_events {
            local_delta_by_value[event.value as usize] += event.delta;
        }

        for event in &summary.interface_merge_events {
            let a = global_node_id(&interface_offsets, summary.slab_id, event.a);
            let b = global_node_id(&interface_offsets, summary.slab_id, event.b);
            interface_merges_by_value[event.value as usize].push((a, b));
        }
    }

    if summaries.len() >= 2 {
        for pair_id in 0..summaries.len() - 1 {
            let left = &summaries[pair_id];
            let right = &summaries[pair_id + 1];
            let left_face = &left.z_max_face;
            let right_face = &right.z_min_face;

            assert_eq!(left_face.width, right_face.width);
            assert_eq!(left_face.height, right_face.height);

            let stats = sparsify_cross_interface(
                &left_face.values,
                &right_face.values,
                left_face.width,
                left_face.height,
                connectivity,
                InterfaceFiltration::SublevelMax,
                |edge| {
                    let left_index = edge.left_face_index as usize;
                    let right_index = edge.right_face_index as usize;

                    let a = global_node_id(
                        &interface_offsets,
                        left.slab_id,
                        left_face.node_ids[left_index],
                    );
                    let b = global_node_id(
                        &interface_offsets,
                        right.slab_id,
                        right_face.node_ids[right_index],
                    );

                    cross_edges_by_value[edge.value as usize].push((a, b));
                    Ok(())
                },
            )?;

            println!(
                "Betti-0 interface {pair_id}: retained {} of {} cross edges",
                stats.retained_edges, stats.candidate_edges
            );
        }
    }

    let mut beta0 = 0i64;
    let mut sparse_curve = Vec::new();
    let mut previous: Option<i64> = None;

    for value in 0..NUM_U16_VALUES {
        beta0 += local_delta_by_value[value];

        for &(a, b) in &interface_merges_by_value[value] {
            if !global_uf.union(a, b) {
                beta0 += 1;
            }
        }

        for &(a, b) in &cross_edges_by_value[value] {
            if global_uf.union(a, b) {
                beta0 -= 1;
            }
        }

        debug_assert!(
            beta0 >= 0,
            "Betti-0 became negative at threshold {}: beta0 = {}",
            value,
            beta0
        );

        if previous != Some(beta0) {
            sparse_curve.push((value as u16, beta0));
            previous = Some(beta0);
        }
    }

    Ok(sparse_curve)
}

pub fn compute_event_based_betti0_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let start = Instant::now();

    println!("Processing slabs with the in-memory event-based Betti-0 reducer...");

    let summaries = process_all_slabs_event_based_betti0(volume, slab_depth, connectivity)?;

    println!("Processed {} slab summaries", summaries.len());

    let curve = reduce_event_based_betti0(&summaries, connectivity)?;

    println!(
        "In-memory event-based Betti-0 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(curve)
}

pub fn write_event_betti0_csv(path: &Path, curve: &[(u16, i64)]) -> Result<()> {
    let mut file = AtomicOutput::create(path)?;
    writeln!(file, "threshold,betti0")?;

    for &(threshold, beta0) in curve {
        writeln!(file, "{threshold},{beta0}")?;
    }

    file.commit()
}
