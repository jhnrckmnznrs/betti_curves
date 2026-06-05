use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};

const NUM_U16_VALUES: usize = 65_536;

#[derive(Debug)]
struct VoxelBuckets {
    offsets: Vec<usize>,
    indices: Vec<u32>,
}

struct LocalBackgroundEventBuffers<'a> {
    delta: &'a mut i64,
    interface_merge_events: &'a mut Vec<InterfaceMergeEvent>,
    interface_outside_events: &'a mut Vec<InterfaceOutsideEvent>,
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

#[derive(Debug, Clone, Copy)]
struct LocalDeltaEvent {
    value: u16,
    delta: i64,
}

#[derive(Debug, Clone, Copy)]
struct InterfaceMergeEvent {
    value: u16,
    a: u32,
    b: u32,

    /// Local effect already included in local_delta_events.
    ///
    /// For background non-outside component count:
    ///     nonoutside + nonoutside -> -1
    ///     outside + nonoutside    -> -1
    ///     outside + outside       ->  0
    local_delta: i64,
}

#[derive(Debug, Clone, Copy)]
struct InterfaceOutsideEvent {
    value: u16,
    node: u32,

    /// Local effect already included in local_delta_events.
    ///
    /// Usually:
    ///     0  for an outside boundary birth
    ///    -1  for a nonoutside interface component locally merging into outside
    local_delta: i64,
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
struct SlabBetti2Summary {
    slab_id: usize,

    interface_node_count: u32,

    /// Local changes to the number of background components not touching outside.
    local_delta_events: Vec<LocalDeltaEvent>,

    /// Local equivalences between boundary interface nodes.
    interface_merge_events: Vec<InterfaceMergeEvent>,

    /// Events saying an interface component becomes outside-connected locally.
    interface_outside_events: Vec<InterfaceOutsideEvent>,

    z_min_face: BoundaryFaceNodes,
    z_max_face: BoundaryFaceNodes,
}

#[derive(Debug)]
struct LocalBackgroundUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,

    /// For each local component root, one representative interface node if present.
    interface_rep: Vec<Option<u32>>,

    /// Whether this local component touches the global image boundary.
    touches_outside: Vec<bool>,
}

#[derive(Debug)]
struct LocalUnionOutcome {
    local_delta: i64,
    interface_merge: Option<InterfaceMergeEvent>,
    outside_event: Option<InterfaceOutsideEvent>,
}

impl LocalBackgroundUnionFind {
    fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "LocalBackgroundUnionFind uses u32 indices; slab has too many voxels"
        );

        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0u8; n],
            interface_rep: vec![None; n],
            touches_outside: vec![false; n],
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

    fn mark_outside(&mut self, x: u32) {
        let root = self.find(x);
        self.touches_outside[root as usize] = true;
    }

    fn union_with_events(&mut self, a: u32, b: u32, value: u16) -> Option<LocalUnionOutcome> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return None;
        }

        let rep_a = self.interface_rep[ra as usize];
        let rep_b = self.interface_rep[rb as usize];

        let out_a = self.touches_outside[ra as usize];
        let out_b = self.touches_outside[rb as usize];

        let local_delta = if out_a && out_b { 0 } else { -1 };

        let interface_merge = match (rep_a, rep_b) {
            (Some(a_rep), Some(b_rep)) if a_rep != b_rep => Some(InterfaceMergeEvent {
                value,
                a: a_rep,
                b: b_rep,
                local_delta,
            }),
            _ => None,
        };

        let outside_event = match (rep_a, rep_b) {
            (Some(rep), None) => {
                if !out_a && out_b {
                    Some(InterfaceOutsideEvent {
                        value,
                        node: rep,
                        local_delta,
                    })
                } else {
                    None
                }
            }
            (None, Some(rep)) => {
                if out_a && !out_b {
                    Some(InterfaceOutsideEvent {
                        value,
                        node: rep,
                        local_delta,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        self.interface_rep[ra as usize] = rep_a.or(rep_b);
        self.touches_outside[ra as usize] = out_a || out_b;

        Some(LocalUnionOutcome {
            local_delta,
            interface_merge,
            outside_event,
        })
    }
}

#[derive(Debug)]
struct GlobalOutsideUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
    touches_outside: Vec<bool>,
}

impl GlobalOutsideUnionFind {
    fn new(n: usize) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "GlobalOutsideUnionFind uses u32 indices; too many interface nodes"
        );

        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0u8; n],
            touches_outside: vec![false; n],
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

    fn is_outside(&mut self, x: u32) -> bool {
        let root = self.find(x);
        self.touches_outside[root as usize]
    }

    fn mark_outside(&mut self, x: u32) {
        let root = self.find(x);
        self.touches_outside[root as usize] = true;
    }

    /// Returns None if already same component.
    /// Otherwise returns Some((a_outside, b_outside)).
    fn union(&mut self, a: u32, b: u32) -> Option<(bool, bool)> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        if ra == rb {
            return None;
        }

        let out_a = self.touches_outside[ra as usize];
        let out_b = self.touches_outside[rb as usize];

        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];

        if rank_a < rank_b {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb as usize] = ra;

        if rank_a == rank_b {
            self.rank[ra as usize] += 1;
        }

        self.touches_outside[ra as usize] = out_a || out_b;

        Some((out_a, out_b))
    }
}

fn voxel_touches_global_boundary(
    block: &Block,
    x: usize,
    y: usize,
    z_local: usize,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
) -> bool {
    let z_global = block.z0 + z_local;

    x == 0
        || x + 1 == global_width
        || y == 0
        || y + 1 == global_height
        || z_global == 0
        || z_global + 1 == global_depth
}

fn process_local_union_outcome(
    outcome: LocalUnionOutcome,
    delta: &mut i64,
    interface_merge_events: &mut Vec<InterfaceMergeEvent>,
    interface_outside_events: &mut Vec<InterfaceOutsideEvent>,
) {
    *delta += outcome.local_delta;

    if let Some(event) = outcome.interface_merge {
        interface_merge_events.push(event);
    }

    if let Some(event) = outcome.outside_event {
        interface_outside_events.push(event);
    }
}

fn union_active_neighbor_local_background(
    uf: &mut LocalBackgroundUnionFind,
    active: &[u8],
    idx_u32: u32,
    neighbor: usize,
    value: u16,
    events: &mut LocalBackgroundEventBuffers<'_>,
) {
    assert!(
        neighbor < active.len(),
        "invalid neighbor: current={}, neighbor={}, active_len={}, value={}",
        idx_u32,
        neighbor,
        active.len(),
        value
    );

    if active[neighbor] == 0 {
        return;
    }

    if let Some(outcome) = uf.union_with_events(idx_u32, neighbor as u32, value) {
        process_local_union_outcome(
            outcome,
            &mut *events.delta,
            &mut *events.interface_merge_events,
            &mut *events.interface_outside_events,
        );
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

fn process_slab_event_based_betti2(
    slab_id: usize,
    block: &Block,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
) -> SlabBetti2Summary {
    let n = block.voxel_count();

    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = LocalBackgroundUnionFind::new(n);
    let mut active = vec![0u8; n];

    let buckets = build_voxel_buckets_u16(&block.values);

    // One interface node for every voxel on z_min or z_max.
    // If depth == 1, the same voxel gets only one node.
    let mut boundary_node_for_voxel = vec![None; n];
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

    let mut local_delta_events = Vec::new();
    let mut interface_merge_events = Vec::new();
    let mut interface_outside_events = Vec::new();

    // Background superlevel: process intensities descending.
    for value in (0..NUM_U16_VALUES).rev() {
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

            let x = idx % width;
            let y = (idx / width) % height;
            let z = idx / slice_size;

            if let Some(interface_node) = boundary_node_for_voxel[idx] {
                uf.set_interface_rep(idx_u32, interface_node);
            }

            let touches_outside = voxel_touches_global_boundary(
                block,
                x,
                y,
                z,
                global_width,
                global_height,
                global_depth,
            );

            if touches_outside {
                uf.mark_outside(idx_u32);

                if let Some(interface_node) = boundary_node_for_voxel[idx] {
                    interface_outside_events.push(InterfaceOutsideEvent {
                        value: value_u16,
                        node: interface_node,
                        local_delta: 0,
                    });
                }
            } else {
                // A new background component not touching outside contributes +1.
                delta += 1;
            }

            let mut event_buffers = LocalBackgroundEventBuffers {
                delta: &mut delta,
                interface_merge_events: &mut interface_merge_events,
                interface_outside_events: &mut interface_outside_events,
            };

            match background_connectivity {
                Connectivity::Six => {
                    // -x
                    if x > 0 {
                        let neighbor = idx - 1;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }

                    // +x
                    if x + 1 < width {
                        let neighbor = idx + 1;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }

                    // -y
                    if y > 0 {
                        let neighbor = idx - width;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }

                    // +y
                    if y + 1 < height {
                        let neighbor = idx + width;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }

                    // -z
                    if z > 0 {
                        let neighbor = idx - slice_size;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }

                    // +z
                    if z + 1 < depth {
                        let neighbor = idx + slice_size;

                        union_active_neighbor_local_background(
                            &mut uf,
                            &active,
                            idx_u32,
                            neighbor,
                            value_u16,
                            &mut event_buffers,
                        );
                    }
                }

                Connectivity::TwentySix => {
                    for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }

                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                let nz = z as isize + dz;

                                // Perform this check before converting to usize.
                                if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || nx >= width as isize
                                    || ny >= height as isize
                                    || nz >= depth as isize
                                {
                                    continue;
                                }

                                let nx = nx as usize;
                                let ny = ny as usize;
                                let nz = nz as usize;

                                let neighbor = nz * slice_size + ny * width + nx;

                                union_active_neighbor_local_background(
                                    &mut uf,
                                    &active,
                                    idx_u32,
                                    neighbor,
                                    value_u16,
                                    &mut event_buffers,
                                );
                            }
                        }
                    }
                }
            }
        }

        if delta != 0 {
            local_delta_events.push(LocalDeltaEvent {
                value: value_u16,
                delta,
            });
        }
    }

    let z_min_face = extract_boundary_face_nodes(block, &boundary_node_for_voxel, 0);
    let z_max_face = extract_boundary_face_nodes(block, &boundary_node_for_voxel, depth - 1);

    SlabBetti2Summary {
        slab_id,
        interface_node_count,
        local_delta_events,
        interface_merge_events,
        interface_outside_events,
        z_min_face,
        z_max_face,
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

fn process_all_slabs_event_based_betti2(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<SlabBetti2Summary>> {
    let [global_width, global_height, global_depth] = volume.shape();

    let ranges = make_slab_ranges(volume.depth, slab_depth);

    println!(
        "Processing {} Betti-2 slab summaries in parallel...",
        ranges.len()
    );

    let parallel_results: Vec<Result<SlabBetti2Summary>> = ranges
        .par_iter()
        .map(|&(slab_id, z0, z1)| {
            let block = volume.read_z_slab(z0, z1)?;

            let summary = process_slab_event_based_betti2(
                slab_id,
                &block,
                global_width,
                global_height,
                global_depth,
                background_connectivity,
            );

            Ok(summary)
        })
        .collect();

    let mut summaries: Vec<SlabBetti2Summary> =
        parallel_results.into_iter().collect::<Result<Vec<_>>>()?;

    summaries.sort_by_key(|s| s.slab_id);

    Ok(summaries)
}

// fn process_all_slabs_event_based_betti2(
//     volume: &TiffStackReader,
//     slab_depth: usize,
//     background_connectivity: Connectivity,
// ) -> Result<Vec<SlabBetti2Summary>> {
//     let [global_width, global_height, global_depth] = volume.shape();

//     let mut summaries = Vec::new();

//     let mut slab_id = 0usize;
//     let mut z0 = 0usize;

//     while z0 < volume.depth {
//         let z1 = usize::min(z0 + slab_depth, volume.depth);

//         let block = volume.read_z_slab(z0, z1)?;

//         let summary = process_slab_event_based_betti2(
//             slab_id,
//             &block,
//             global_width,
//             global_height,
//             global_depth,
//             background_connectivity,
//         );

//         summaries.push(summary);

//         z0 = z1;
//         slab_id += 1;
//     }

//     Ok(summaries)
// }

fn build_interface_offsets(summaries: &[SlabBetti2Summary]) -> Vec<u32> {
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

#[derive(Debug, Clone, Copy)]
struct CrossSlabEdge {
    value: u16,
    a_global: u32,
    b_global: u32,
}

fn generate_cross_slab_edges(
    summaries: &[SlabBetti2Summary],
    interface_offsets: &[u32],
    background_connectivity: Connectivity,
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

                match background_connectivity {
                    Connectivity::Six => {
                        let i_right = y * width + x;

                        let right_local_node = right_face.node_ids[i_right];
                        let right_value = right_face.values[i_right];

                        // Background superlevel edge appears at min().
                        let value = left_value.min(right_value);

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

                                let value = left_value.min(right_value);

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

fn reduce_event_based_betti2(
    summaries: &[SlabBetti2Summary],
    background_connectivity: Connectivity,
    max_value: u16,
) -> Vec<(u16, i64)> {
    let interface_offsets = build_interface_offsets(summaries);
    let total_interface_nodes = *interface_offsets.last().unwrap_or(&0) as usize;

    let mut global_uf = GlobalOutsideUnionFind::new(total_interface_nodes);

    let mut local_delta_by_value = vec![0i64; NUM_U16_VALUES];

    let mut outside_events_by_value: Vec<Vec<(u32, i64)>> = vec![Vec::new(); NUM_U16_VALUES];
    let mut interface_merges_by_value: Vec<Vec<(u32, u32, i64)>> = vec![Vec::new(); NUM_U16_VALUES];
    let mut cross_edges_by_value: Vec<Vec<(u32, u32)>> = vec![Vec::new(); NUM_U16_VALUES];

    for summary in summaries {
        for event in &summary.local_delta_events {
            local_delta_by_value[event.value as usize] += event.delta;
        }

        for event in &summary.interface_outside_events {
            let node = global_node_id(&interface_offsets, summary.slab_id, event.node);

            outside_events_by_value[event.value as usize].push((node, event.local_delta));
        }

        for event in &summary.interface_merge_events {
            let a = global_node_id(&interface_offsets, summary.slab_id, event.a);
            let b = global_node_id(&interface_offsets, summary.slab_id, event.b);

            interface_merges_by_value[event.value as usize].push((a, b, event.local_delta));
        }
    }

    let cross_edges =
        generate_cross_slab_edges(summaries, &interface_offsets, background_connectivity);

    for edge in cross_edges {
        cross_edges_by_value[edge.value as usize].push((edge.a_global, edge.b_global));
    }

    // dense_betti2[t] = beta2 for foreground threshold t.
    let max_value_usize = max_value as usize;
    let mut dense_betti2 = vec![0i64; max_value_usize + 1];

    let mut beta2 = 0i64;

    // Process background superlevel threshold s descending.
    //
    // After processing value s, we know B_s = { I >= s },
    // which corresponds to foreground threshold t = s - 1.
    for s in (1..=max_value_usize).rev() {
        // 1. Apply local slab delta events.
        beta2 += local_delta_by_value[s];

        // 2. Process local outside events before local/cross merges.
        for &(node, local_delta) in &outside_events_by_value[s] {
            match local_delta {
                0 => {
                    // Outside birth: local delta was already 0.
                    // Mark outside but do not change beta2.
                    global_uf.mark_outside(node);
                }
                -1 => {
                    // Locally, a nonoutside interface component became outside,
                    // so local_delta_by_value already subtracted 1.
                    //
                    // If globally it was already outside, that subtraction was
                    // redundant and must be added back.
                    if global_uf.is_outside(node) {
                        beta2 += 1;
                    } else {
                        global_uf.mark_outside(node);
                    }
                }
                _ => panic!("unexpected outside event local_delta: {}", local_delta),
            }
        }

        // 3. Process local interface merge events.
        //
        // The local effect is already included in local_delta_by_value.
        // We apply correction = actual_delta - local_delta.
        for &(a, b, local_delta) in &interface_merges_by_value[s] {
            let actual_delta = match global_uf.union(a, b) {
                Some((a_out, b_out)) => {
                    if a_out && b_out {
                        0
                    } else {
                        -1
                    }
                }
                None => 0,
            };

            beta2 += actual_delta - local_delta;
        }

        // 4. Process cross-slab edges.
        //
        // These were not included locally, so apply actual delta directly.
        for &(a, b) in &cross_edges_by_value[s] {
            if let Some((a_out, b_out)) = global_uf.union(a, b)
                && !(a_out && b_out)
            {
                beta2 -= 1;
            }
        }

        let foreground_threshold = s - 1;
        dense_betti2[foreground_threshold] = beta2;
    }

    // At t = max_value, background { I > max_value } is empty.
    dense_betti2[max_value_usize] = 0;

    // Compress in ascending foreground-threshold order.
    let mut sparse_curve = Vec::new();
    let mut previous: Option<i64> = None;

    for (t, &b2) in dense_betti2.iter().enumerate() {
        if previous != Some(b2) {
            sparse_curve.push((t as u16, b2));
            previous = Some(b2);
        }
    }

    sparse_curve
}

pub fn compute_event_based_betti2_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    background_connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let start = Instant::now();

    println!("Processing slabs event-based for Betti-2...");

    let summaries =
        process_all_slabs_event_based_betti2(volume, slab_depth, background_connectivity)?;

    println!("Processed {} slab summaries", summaries.len());

    let max_value = summaries
        .iter()
        .flat_map(|s| s.z_min_face.values.iter().chain(s.z_max_face.values.iter()))
        .copied()
        .max()
        .unwrap_or(0);

    // The face-only max is not always the true image max.
    // So scan volume once cheaply for the actual max.
    let mut actual_max = max_value;

    let mut z0 = 0usize;
    while z0 < volume.depth {
        let z1 = usize::min(z0 + slab_depth, volume.depth);
        let block = volume.read_z_slab(z0, z1)?;

        if let Some(block_max) = block.values.iter().copied().max() {
            actual_max = actual_max.max(block_max);
        }

        z0 = z1;
    }

    let curve = reduce_event_based_betti2(&summaries, background_connectivity, actual_max);

    println!(
        "Event-based slabwise Betti-2 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(curve)
}

pub fn write_event_betti2_csv(path: &Path, curve: &[(u16, i64)]) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "threshold,betti2")?;

    for &(threshold, beta2) in curve {
        writeln!(file, "{},{}", threshold, beta2)?;
    }

    Ok(())
}
