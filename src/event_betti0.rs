use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};

const NUM_U16_VALUES: usize = 65_536;

#[derive(Debug, Clone, Copy)]
pub struct Betti0Event {
    pub value: u16,
    pub delta: i64,
}

#[derive(Debug, Clone, Copy)]
struct InterfaceMergeEvent {
    value: u16,
    a: u32,
    b: u32,
}

#[derive(Debug)]
struct BoundaryFaceNodes {
    width: usize,
    height: usize,

    /// One interface node id per face voxel.
    node_ids: Vec<u32>,

    /// Original grayscale value of the face voxel.
    values: Vec<u16>,
}

#[derive(Debug)]
struct SlabEventSummary {
    slab_id: usize,

    /// Local Betti-0 changes inside this slab.
    local_betti0_events: Vec<Betti0Event>,

    /// Number of interface nodes owned by this slab.
    interface_node_count: u32,

    /// Local equivalence events between interface nodes.
    ///
    /// These do NOT change Betti-0 globally by themselves.
    /// They only tell the global reducer that two boundary nodes
    /// became connected inside the slab at this filtration value.
    interface_merge_events: Vec<InterfaceMergeEvent>,

    z_min_face: BoundaryFaceNodes,
    z_max_face: BoundaryFaceNodes,
}

#[derive(Debug)]
struct VoxelBuckets {
    /// offsets[v]..offsets[v + 1] gives voxel indices with value v.
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

    /// For each root, an optional interface node representative.
    interface_rep: Vec<Option<u32>>,
}

impl LocalEventUnionFind {
    fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "LocalEventUnionFind uses u32 indices; slab has too many voxels"
        );

        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];
        let interface_rep = vec![None; n];

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
        self.interface_rep[root as usize] = Some(rep);
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

        let merged_rep = match (rep_a, rep_b) {
            (Some(a_rep), Some(b_rep)) => {
                self.interface_rep[ra as usize] = Some(a_rep);

                if a_rep != b_rep {
                    Some(InterfaceMergeEvent {
                        value,
                        a: a_rep,
                        b: b_rep,
                    })
                } else {
                    None
                }
            }
            (Some(a_rep), None) => {
                self.interface_rep[ra as usize] = Some(a_rep);
                None
            }
            (None, Some(b_rep)) => {
                self.interface_rep[ra as usize] = Some(b_rep);
                None
            }
            (None, None) => {
                self.interface_rep[ra as usize] = None;
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
struct GlobalInterfaceUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl GlobalInterfaceUnionFind {
    fn new(n: usize) -> Self {
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

    fn union(&mut self, a: u32, b: u32) -> bool {
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

#[derive(Debug, Clone, Copy)]
struct CrossSlabEdge {
    value: u16,
    a_global: u32,
    b_global: u32,
}

/// Process one slab once, event-based.
///
/// This computes local Betti-0 events and boundary interface events.
/// It does not yet do global slab reconciliation.
fn process_slab_event_based_betti0(
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

    // Create one interface node for each unique voxel on z_min or z_max.
    // If depth == 1, the same voxel is both z_min and z_max and gets one node.
    let mut boundary_node_for_voxel: Vec<Option<u32>> = vec![None; n];
    let mut interface_node_count = 0u32;

    for z in [0usize, depth - 1] {
        for y in 0..height {
            for x in 0..width {
                let idx = z * slice_size + y * width + x;

                if boundary_node_for_voxel[idx].is_none() {
                    boundary_node_for_voxel[idx] = Some(interface_node_count);
                    interface_node_count += 1;
                }
            }
        }
    }

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

            if let Some(interface_node) = boundary_node_for_voxel[idx] {
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let x = idx % width;
            let y = (idx / width) % height;
            let z = idx / slice_size;

            match connectivity {
                Connectivity::Six => {
                    // Event-based activation must check all active neighbors,
                    // not just backward neighbors.
                    //
                    // -x
                    if x > 0 {
                        let nb = idx - 1;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }

                    // +x
                    if x + 1 < width {
                        let nb = idx + 1;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }

                    // -y
                    if y > 0 {
                        let nb = idx - width;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }

                    // +y
                    if y + 1 < height {
                        let nb = idx + width;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }

                    // -z
                    if z > 0 {
                        let nb = idx - slice_size;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }

                    // +z
                    if z + 1 < depth {
                        let nb = idx + slice_size;
                        union_active_neighbor_event(
                            &mut uf,
                            &active,
                            idx_u32,
                            nb,
                            value_u16,
                            &mut delta,
                            &mut interface_merge_events,
                        );
                    }
                }

                Connectivity::TwentySix => {
                    // Event-based activation must check all 26 active neighbors.
                    for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }

                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                let nz = z as isize + dz;

                                if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || nx >= width as isize
                                    || ny >= height as isize
                                    || nz >= depth as isize
                                {
                                    continue;
                                }

                                let nb =
                                    nz as usize * slice_size + ny as usize * width + nx as usize;

                                union_active_neighbor_event(
                                    &mut uf,
                                    &active,
                                    idx_u32,
                                    nb,
                                    value_u16,
                                    &mut delta,
                                    &mut interface_merge_events,
                                );
                            }
                        }
                    }
                }
            }
        }

        if delta != 0 {
            local_betti0_events.push(Betti0Event {
                value: value_u16,
                delta,
            });
        }
    }

    let z_min_face = extract_boundary_face_nodes(block, &boundary_node_for_voxel, 0);
    let z_max_face = extract_boundary_face_nodes(block, &boundary_node_for_voxel, depth - 1);

    SlabEventSummary {
        slab_id,
        local_betti0_events,
        interface_node_count,
        interface_merge_events,
        z_min_face,
        z_max_face,
    }
}

fn extract_boundary_face_nodes(
    block: &Block,
    boundary_node_for_voxel: &[Option<u32>],
    z_local: usize,
) -> BoundaryFaceNodes {
    let width = block.shape[0];
    let height = block.shape[1];
    let slice_size = width * height;

    let mut node_ids = vec![0u32; width * height];
    let mut values = vec![0u16; width * height];

    for y in 0..height {
        for x in 0..width {
            let face_idx = y * width + x;
            let idx = z_local * slice_size + y * width + x;

            node_ids[face_idx] = boundary_node_for_voxel[idx]
                .expect("every z-face voxel must have an interface node");
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
        let z1 = usize::min(z0 + slab_depth, depth);
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
            let summary = process_slab_event_based_betti0(slab_id, &block, connectivity);
            Ok(summary)
        })
        .collect();

    let mut summaries: Vec<SlabEventSummary> =
        parallel_results.into_iter().collect::<Result<Vec<_>>>()?;

    summaries.sort_by_key(|s| s.slab_id);

    Ok(summaries)
}

// /// Read all z-slabs and process each slab once.
// fn process_all_slabs_event_based_betti0(
//     volume: &TiffStackReader,
//     slab_depth: usize,
//     connectivity: Connectivity,
// ) -> Result<Vec<SlabEventSummary>> {
//     let mut summaries = Vec::new();

//     let mut slab_id = 0usize;
//     let mut z0 = 0usize;

//     while z0 < volume.depth {
//         let z1 = usize::min(z0 + slab_depth, volume.depth);

//         let block = volume.read_z_slab(z0, z1)?;
//         let summary = process_slab_event_based_betti0(slab_id, &block, connectivity);

//         summaries.push(summary);

//         z0 = z1;
//         slab_id += 1;
//     }

//     Ok(summaries)
// }

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

fn generate_cross_slab_edges(
    summaries: &[SlabEventSummary],
    interface_offsets: &[u32],
    connectivity: Connectivity,
) -> Vec<CrossSlabEdge> {
    let mut edges = Vec::new();

    if summaries.len() < 2 {
        return edges;
    }

    for k in 0..summaries.len() - 1 {
        let left = &summaries[k];
        let right = &summaries[k + 1];

        let left_face = &left.z_max_face;
        let right_face = &right.z_min_face;

        assert_eq!(left_face.width, right_face.width);
        assert_eq!(left_face.height, right_face.height);

        let width = left_face.width;
        let height = left_face.height;

        for y in 0..height {
            for x in 0..width {
                let i_left = y * width + x;

                let left_local_node = left_face.node_ids[i_left];
                let left_value = left_face.values[i_left];

                match connectivity {
                    Connectivity::Six => {
                        let i_right = y * width + x;

                        let right_local_node = right_face.node_ids[i_right];
                        let right_value = right_face.values[i_right];

                        let value = left_value.max(right_value);

                        edges.push(CrossSlabEdge {
                            value,
                            a_global: global_node_id(
                                interface_offsets,
                                left.slab_id,
                                left_local_node,
                            ),
                            b_global: global_node_id(
                                interface_offsets,
                                right.slab_id,
                                right_local_node,
                            ),
                        });
                    }

                    Connectivity::TwentySix => {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;

                                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize
                                {
                                    continue;
                                }

                                let i_right = ny as usize * width + nx as usize;

                                let right_local_node = right_face.node_ids[i_right];
                                let right_value = right_face.values[i_right];

                                let value = left_value.max(right_value);

                                edges.push(CrossSlabEdge {
                                    value,
                                    a_global: global_node_id(
                                        interface_offsets,
                                        left.slab_id,
                                        left_local_node,
                                    ),
                                    b_global: global_node_id(
                                        interface_offsets,
                                        right.slab_id,
                                        right_local_node,
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    edges
}

/// Reduce slab summaries into the global sparse Betti-0 curve.
///
/// Local Betti-0 events contribute directly to the Betti count.
/// Local interface merge events only update the global interface UF.
/// Cross-slab edges update the UF and contribute -1 to Betti-0 if successful.
fn reduce_event_based_betti0(
    summaries: &[SlabEventSummary],
    connectivity: Connectivity,
) -> Vec<(u16, i64)> {
    let interface_offsets = build_interface_offsets(summaries);
    let total_interface_nodes = *interface_offsets.last().unwrap_or(&0) as usize;

    let mut global_uf = GlobalInterfaceUnionFind::new(total_interface_nodes);

    let mut local_delta_by_value = vec![0i64; NUM_U16_VALUES];

    let mut interface_merges_by_value: Vec<Vec<(u32, u32)>> = vec![Vec::new(); NUM_U16_VALUES];
    let mut cross_edges_by_value: Vec<Vec<(u32, u32)>> = vec![Vec::new(); NUM_U16_VALUES];

    for summary in summaries {
        for event in &summary.local_betti0_events {
            local_delta_by_value[event.value as usize] += event.delta;
        }

        for event in &summary.interface_merge_events {
            let a_global = global_node_id(&interface_offsets, summary.slab_id, event.a);
            let b_global = global_node_id(&interface_offsets, summary.slab_id, event.b);

            interface_merges_by_value[event.value as usize].push((a_global, b_global));
        }
    }

    let cross_edges = generate_cross_slab_edges(summaries, &interface_offsets, connectivity);

    for edge in cross_edges {
        cross_edges_by_value[edge.value as usize].push((edge.a_global, edge.b_global));
    }

    let mut beta0 = 0i64;
    let mut sparse_curve: Vec<(u16, i64)> = Vec::new();
    let mut previous_recorded: Option<i64> = None;

    for value in 0..NUM_U16_VALUES {
        let mut changed = false;
        let mut redundant_interface_merges = 0i64;
        let mut successful_cross_merges = 0i64;

        // 1. Apply local Betti-0 changes from all slabs.
        //
        // This includes:
        //   +1 for local births
        //   -1 for local successful unions
        let local_delta = local_delta_by_value[value];

        if local_delta != 0 {
            beta0 += local_delta;
            changed = true;
        }

        // 2. Process local interface merge events.
        //
        // These represent local slab merges between boundary-reaching components.
        //
        // Important:
        // The local merge has already contributed -1 through local_delta.
        //
        // If the global interface UF says the two interface nodes are already
        // connected, that local -1 was redundant globally, so we must add +1 back.
        for &(a, b) in &interface_merges_by_value[value] {
            if !global_uf.union(a, b) {
                beta0 += 1;
                redundant_interface_merges += 1;
                changed = true;
            }
        }

        // 3. Process cross-slab edges.
        //
        // A successful cross-slab union reduces global Betti-0 by 1.
        for &(a, b) in &cross_edges_by_value[value] {
            if global_uf.union(a, b) {
                beta0 -= 1;
                successful_cross_merges += 1;
                changed = true;
            }
        }

        if changed {
            println!(
                "t={} local_delta={} redundant_interface_merges={} successful_cross_merges={} beta0={}",
                value, local_delta, redundant_interface_merges, successful_cross_merges, beta0
            );
        }

        // Betti-0 can never be negative.
        // If this triggers, we still have an accounting bug.
        debug_assert!(
            beta0 >= 0,
            "Betti-0 became negative at threshold {}: beta0 = {}",
            value,
            beta0
        );

        if changed && previous_recorded != Some(beta0) {
            sparse_curve.push((value as u16, beta0));
            previous_recorded = Some(beta0);
        }
    }

    if sparse_curve.first().is_none_or(|&(t, _)| t != 0) {
        sparse_curve.insert(0, (0, 0));
    }

    sparse_curve
}

/// Public entry point.
///
/// Computes global Betti-0 of the full TIFF volume using event-based
/// z-slab processing.
pub fn compute_event_based_betti0_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let start = Instant::now();

    println!("Processing slabs event-based for Betti-0...");

    let summaries = process_all_slabs_event_based_betti0(volume, slab_depth, connectivity)?;

    println!("Processed {} slab summaries", summaries.len());

    let curve = reduce_event_based_betti0(&summaries, connectivity);

    println!(
        "Event-based Betti-0 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(curve)
}

pub fn write_event_betti0_csv(path: &Path, curve: &[(u16, i64)]) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "threshold,betti0")?;

    for &(threshold, beta0) in curve {
        writeln!(file, "{},{}", threshold, beta0)?;
    }

    Ok(())
}
