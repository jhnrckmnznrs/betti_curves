use crate::connectivity::Connectivity;
use crate::io::{Block, TiffStackReader};
use crate::union_find::{
    DynamicUnionFind, FaceLabels, INACTIVE_ROOT, LocalRootKey, UnionFind, get_or_create_global_id,
};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug)]
struct SlabThresholdSummary {
    slab_id: usize,
    local_beta0: i64,
    z_min_face: FaceLabels,
    z_max_face: FaceLabels,
}

fn process_slab_at_threshold(
    slab_id: usize,
    block: &Block,
    threshold: u16,
    connectivity: Connectivity,
) -> SlabThresholdSummary {
    let n = block.voxel_count();

    let width = block.shape[0];
    let height = block.shape[1];
    let depth = block.shape[2];
    let slice_size = width * height;

    let mut uf = UnionFind::new(n);
    let mut active = vec![0u8; n];

    let mut local_beta0: i64 = 0;

    for (idx, is_active) in active.iter_mut().enumerate() {
        if block.values[idx] <= threshold {
            *is_active = 1;
            local_beta0 += 1;
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

        match connectivity {
            Connectivity::Six => {
                if x > 0 {
                    let nb = idx - 1;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_beta0 -= 1;
                    }
                }

                if y > 0 {
                    let nb = idx - width;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_beta0 -= 1;
                    }
                }

                if z > 0 {
                    let nb = idx - slice_size;
                    if active[nb] != 0 && uf.union(idx_u32, nb as u32) {
                        local_beta0 -= 1;
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
                                local_beta0 -= 1;
                            }
                        }
                    }
                }
            }
        }
    }

    let z_min_face = extract_z_face_labels(block, &mut uf, &active, 0);
    let z_max_face = extract_z_face_labels(block, &mut uf, &active, depth - 1);

    SlabThresholdSummary {
        slab_id,
        local_beta0,
        z_min_face,
        z_max_face,
    }
}

pub(crate) fn extract_z_face_labels(
    block: &Block,
    uf: &mut UnionFind,
    active: &[u8],
    z_local: usize,
) -> FaceLabels {
    let width = block.shape[0];
    let height = block.shape[1];
    let slice_size = width * height;

    let mut roots = vec![INACTIVE_ROOT; width * height];

    for y in 0..height {
        for x in 0..width {
            let face_idx = y * width + x;
            let idx = z_local * slice_size + y * width + x;

            if active[idx] != 0 {
                roots[face_idx] = uf.find(idx as u32);
            }
        }
    }

    FaceLabels {
        width,
        height,
        roots,
    }
}

fn reconcile_adjacent_z_faces(
    left_slab_id: usize,
    left_z_max: &FaceLabels,
    right_slab_id: usize,
    right_z_min: &FaceLabels,
    global_uf: &mut DynamicUnionFind,
    id_map: &mut HashMap<LocalRootKey, u32>,
    connectivity: Connectivity,
) -> i64 {
    assert_eq!(left_z_max.width, right_z_min.width);
    assert_eq!(left_z_max.height, right_z_min.height);

    let width = left_z_max.width;
    let height = left_z_max.height;

    let mut successful_cross_merges = 0i64;

    for y in 0..height {
        for x in 0..width {
            let i_left = y * width + x;
            let root_a = left_z_max.roots[i_left];

            if root_a == INACTIVE_ROOT {
                continue;
            }

            match connectivity {
                Connectivity::Six => {
                    let i_right = y * width + x;
                    let root_b = right_z_min.roots[i_right];

                    if root_b == INACTIVE_ROOT {
                        continue;
                    }

                    let key_a = LocalRootKey {
                        slab_id: left_slab_id,
                        root: root_a,
                    };
                    let key_b = LocalRootKey {
                        slab_id: right_slab_id,
                        root: root_b,
                    };

                    let id_a = get_or_create_global_id(key_a, id_map, global_uf);
                    let id_b = get_or_create_global_id(key_b, id_map, global_uf);

                    if global_uf.union(id_a, id_b) {
                        successful_cross_merges += 1;
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

                            let key_a = LocalRootKey {
                                slab_id: left_slab_id,
                                root: root_a,
                            };
                            let key_b = LocalRootKey {
                                slab_id: right_slab_id,
                                root: root_b,
                            };

                            let id_a = get_or_create_global_id(key_a, id_map, global_uf);
                            let id_b = get_or_create_global_id(key_b, id_map, global_uf);

                            if global_uf.union(id_a, id_b) {
                                successful_cross_merges += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    successful_cross_merges
}

fn compute_global_betti0_threshold_zslabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    threshold: u16,
    connectivity: Connectivity,
    verbose: bool,
) -> Result<i64> {
    let depth = volume.depth;

    let mut local_beta0_sum = 0i64;
    let mut cross_merges = 0i64;

    let mut global_uf = DynamicUnionFind::new();
    let mut id_map: HashMap<LocalRootKey, u32> = HashMap::new();

    let mut prev_slab_id: Option<usize> = None;
    let mut prev_z_max_face: Option<FaceLabels> = None;

    let mut slab_id = 0usize;
    let mut z0 = 0usize;

    while z0 < depth {
        let z1 = usize::min(z0 + slab_depth, depth);

        if verbose {
            println!(
                "  threshold {}, reading slab {}: z={}..{}",
                threshold, slab_id, z0, z1
            );
        }

        let block = volume.read_z_slab(z0, z1)?;
        let summary = process_slab_at_threshold(slab_id, &block, threshold, connectivity);

        local_beta0_sum += summary.local_beta0;

        if let (Some(left_id), Some(left_face)) = (prev_slab_id, prev_z_max_face.as_ref()) {
            let merges = reconcile_adjacent_z_faces(
                left_id,
                left_face,
                summary.slab_id,
                &summary.z_min_face,
                &mut global_uf,
                &mut id_map,
                connectivity,
            );

            cross_merges += merges;
        }

        prev_slab_id = Some(summary.slab_id);
        prev_z_max_face = Some(summary.z_max_face);

        z0 = z1;
        slab_id += 1;
    }

    let global_beta0 = local_beta0_sum - cross_merges;

    if verbose {
        println!(
            "  threshold {}: local sum = {}, cross merges = {}, global beta0 = {}, boundary UF nodes = {}, connectivity = {:?}",
            threshold,
            local_beta0_sum,
            cross_merges,
            global_beta0,
            global_uf.len(),
            connectivity
        );
    }

    Ok(global_beta0)
}

pub fn compute_sparse_global_betti0_parallel(
    volume: &TiffStackReader,
    slab_depth: usize,
    connectivity: Connectivity,
    unique_values: &[u16],
) -> Result<Vec<(u16, i64)>> {
    let mut thresholds = unique_values.to_vec();

    if !thresholds.contains(&0) {
        thresholds.push(0);
    }

    thresholds.sort();
    thresholds.dedup();

    println!(
        "Computing Betti-0 at {} thresholds in parallel...",
        thresholds.len()
    );

    let start = Instant::now();

    let parallel_results: Vec<Result<(u16, i64)>> = thresholds
        .par_iter()
        .map(|&t| {
            let beta0 =
                compute_global_betti0_threshold_zslabs(volume, slab_depth, t, connectivity, false)?;

            Ok((t, beta0))
        })
        .collect();

    let mut results: Vec<(u16, i64)> = parallel_results.into_iter().collect::<Result<Vec<_>>>()?;

    results.sort_by_key(|&(t, _)| t);
    results.dedup_by_key(|(t, _)| *t);

    println!(
        "Parallel Betti-0 computation took {:.3} seconds",
        start.elapsed().as_secs_f64()
    );

    Ok(results)
}
