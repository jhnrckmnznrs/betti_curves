use crate::betti0::extract_z_face_labels;
use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::union_find::{
    DynamicOutsideUnionFind, FaceLabels, INACTIVE_ROOT, LocalRootKey, UnionFind,
    get_or_create_global_outside_id,
};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug)]
struct BackgroundSlabThresholdSummary {
    slab_id: usize,
    local_components: i64,
    local_outside_components: i64,
    outside_roots: HashSet<u32>,
    z_min_face: FaceLabels,
    z_max_face: FaceLabels,
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

fn process_background_slab_at_threshold(
    slab_id: usize,
    block: &Block,
    foreground_threshold: u16,
    global_width: usize,
    global_height: usize,
    global_depth: usize,
    background_connectivity: Connectivity,
) -> BackgroundSlabThresholdSummary {
    let n = block.voxel_count();

    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = UnionFind::new(n);
    let mut active = vec![0u8; n];
    let mut local_components: i64 = 0;

    for (idx, is_active) in active.iter_mut().enumerate() {
        if block.values[idx] > foreground_threshold {
            *is_active = 1;
            local_components += 1;
        }
    }

    for idx in 0..n {
        if active[idx] == 0 {
            continue;
        }

        let x = idx % width;
        let y = (idx / width) % height;
        let z = idx / slice_size;

        let idx_u32 = idx as u32;

        match background_connectivity {
            Connectivity::Six => {
                if x > 0 {
                    let nb = idx - 1;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_components -= 1;
                    }
                }

                if y > 0 {
                    let nb = idx - width;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_components -= 1;
                    }
                }

                if z > 0 {
                    let nb = idx - slice_size;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_components -= 1;
                    }
                }
            }
            Connectivity::TwentySix => {
                for dz in -1isize..=0 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dz == 0 && dy == 0 && dx >= 0 {
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

                            let nb = nz as usize * slice_size + ny as usize * width + nx as usize;

                            if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                                local_components -= 1;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut outside_roots: HashSet<u32> = HashSet::new();

    for (idx, &is_active) in active.iter().enumerate() {
        if is_active == 0 {
            continue;
        }

        let x = idx % width;
        let y = (idx / width) % height;
        let z = idx / slice_size;

        if voxel_touches_global_boundary(block, x, y, z, global_width, global_height, global_depth)
        {
            let root = uf.find(idx as u32);
            outside_roots.insert(root);
        }
    }

    let local_outside_components = outside_roots.len() as i64;

    let z_min_face = extract_z_face_labels(block, &mut uf, &active, 0);
    let z_max_face = extract_z_face_labels(block, &mut uf, &active, depth - 1);

    BackgroundSlabThresholdSummary {
        slab_id,
        local_components,
        local_outside_components,
        outside_roots,
        z_min_face,
        z_max_face,
    }
}

fn reconcile_background_adjacent_z_faces(
    left: &BackgroundSlabThresholdSummary,
    right: &BackgroundSlabThresholdSummary,
    global_uf: &mut DynamicOutsideUnionFind,
    id_map: &mut HashMap<LocalRootKey, u32>,
    background_connectivity: Connectivity,
) -> (i64, i64) {
    let left_z_max = &left.z_max_face;
    let right_z_min = &right.z_min_face;

    assert_eq!(left_z_max.width, right_z_min.width);
    assert_eq!(left_z_max.height, right_z_min.height);

    let width = left_z_max.width;
    let height = left_z_max.height;

    let mut successful_cross_merges = 0i64;
    let mut outside_outside_merges = 0i64;

    for y in 0..height {
        for x in 0..width {
            let i_left = y * width + x;
            let root_a = left_z_max.roots[i_left];

            if root_a == INACTIVE_ROOT {
                continue;
            }

            match background_connectivity {
                Connectivity::Six => {
                    let i_right = y * width + x;
                    let root_b = right_z_min.roots[i_right];

                    if root_b == INACTIVE_ROOT {
                        continue;
                    }

                    let (a_out, b_out, merged) =
                        union_background_roots(left, right, root_a, root_b, global_uf, id_map);

                    if merged {
                        successful_cross_merges += 1;
                        if a_out && b_out {
                            outside_outside_merges += 1;
                        }
                    }
                }
                Connectivity::TwentySix => {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;

                            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                                continue;
                            }

                            let i_right = ny as usize * width + nx as usize;
                            let root_b = right_z_min.roots[i_right];

                            if root_b == INACTIVE_ROOT {
                                continue;
                            }

                            let (a_out, b_out, merged) = union_background_roots(
                                left, right, root_a, root_b, global_uf, id_map,
                            );

                            if merged {
                                successful_cross_merges += 1;
                                if a_out && b_out {
                                    outside_outside_merges += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (successful_cross_merges, outside_outside_merges)
}

fn union_background_roots(
    left: &BackgroundSlabThresholdSummary,
    right: &BackgroundSlabThresholdSummary,
    root_a: u32,
    root_b: u32,
    global_uf: &mut DynamicOutsideUnionFind,
    id_map: &mut HashMap<LocalRootKey, u32>,
) -> (bool, bool, bool) {
    let key_a = LocalRootKey {
        slab_id: left.slab_id,
        root: root_a,
    };

    let key_b = LocalRootKey {
        slab_id: right.slab_id,
        root: root_b,
    };

    let a_touches_outside = left.outside_roots.contains(&root_a);
    let b_touches_outside = right.outside_roots.contains(&root_b);

    let id_a = get_or_create_global_outside_id(key_a, a_touches_outside, id_map, global_uf);
    let id_b = get_or_create_global_outside_id(key_b, b_touches_outside, id_map, global_uf);

    if let Some((a_out, b_out)) = global_uf.union(id_a, id_b) {
        (a_out, b_out, true)
    } else {
        (false, false, false)
    }
}

fn compute_betti2_threshold_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    foreground_threshold: u16,
    background_connectivity: Connectivity,
    verbose: bool,
) -> Result<i64> {
    let [global_width, global_height, global_depth] = volume.shape();

    let mut total_background_components = 0i64;
    let mut outside_background_components = 0i64;

    let mut global_uf = DynamicOutsideUnionFind::new();
    let mut id_map: HashMap<LocalRootKey, u32> = HashMap::new();

    let mut prev_summary: Option<BackgroundSlabThresholdSummary> = None;

    let mut slab_id = 0usize;
    let mut z0 = 0usize;

    while z0 < volume.depth {
        let z1 = usize::min(z0 + slab_depth, volume.depth);

        if verbose {
            println!(
                "  foreground threshold {}, reading background slab {}: z={}..{}",
                foreground_threshold, slab_id, z0, z1
            );
        }

        let block = volume.read_z_slab(z0, z1)?;

        let summary = process_background_slab_at_threshold(
            slab_id,
            &block,
            foreground_threshold,
            global_width,
            global_height,
            global_depth,
            background_connectivity,
        );

        total_background_components += summary.local_components;
        outside_background_components += summary.local_outside_components;

        if let Some(left_summary) = prev_summary.as_ref() {
            let (cross_merges, outside_outside_merges) = reconcile_background_adjacent_z_faces(
                left_summary,
                &summary,
                &mut global_uf,
                &mut id_map,
                background_connectivity,
            );

            total_background_components -= cross_merges;
            outside_background_components -= outside_outside_merges;
        }

        prev_summary = Some(summary);

        z0 = z1;
        slab_id += 1;
    }

    let beta2 = total_background_components - outside_background_components;

    if verbose {
        println!(
            "  foreground threshold {}: background components = {}, outside background components = {}, beta2 = {}, boundary UF nodes = {}",
            foreground_threshold,
            total_background_components,
            outside_background_components,
            beta2,
            global_uf.len()
        );
    }

    Ok(beta2)
}

pub fn compute_sparse_betti2_curve_parallel(
    volume: &TiffStackReader,
    slab_depth: usize,
    unique_values: &[u16],
    background_connectivity: Connectivity,
) -> Result<Vec<(u16, i64)>> {
    let mut thresholds = unique_values.to_vec();

    if !thresholds.contains(&0) {
        thresholds.push(0);
    }

    if !thresholds.contains(&u16::MAX) {
        thresholds.push(u16::MAX);
    }

    thresholds.sort();
    thresholds.dedup();

    println!(
        "Computing Betti-2 at {} thresholds in parallel...",
        thresholds.len()
    );

    let start = Instant::now();

    let parallel_results: Vec<Result<(u16, i64)>> = thresholds
        .par_iter()
        .map(|&t| {
            let beta2 = compute_betti2_threshold_zslabs(
                volume,
                slab_depth,
                t,
                background_connectivity,
                false,
            )?;

            Ok((t, beta2))
        })
        .collect();

    let mut results: Vec<(u16, i64)> = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;

    results.sort_by_key(|&(t, _)| t);
    results.dedup_by_key(|(t, _)| *t);

    println!(
        "Parallel Betti-2 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(results)
}
