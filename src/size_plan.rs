//! Checked preflight sizing for slabwise computations.
//!
//! This module does not claim an exact peak-RSS prediction: the different
//! reducers retain different event and branch records.  It does compute the
//! dimension products and identifier counts that every current reducer must
//! be able to represent.  Keeping these checks in one place turns otherwise
//! late allocation panics into early, actionable errors.

use anyhow::{Result, bail};

use crate::connectivity::Connectivity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SizePlan {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
    pub(crate) slab_depth: usize,
    pub(crate) voxel_count: usize,
    pub(crate) face_area: usize,
    pub(crate) slab_count: usize,
    pub(crate) maximum_slab_voxels: usize,
    pub(crate) cross_candidate_count: usize,
    pub(crate) cross_vertex_count: usize,
    pub(crate) global_interface_nodes: usize,
    pub(crate) hierarchical_pair_interface_nodes: usize,
}

impl SizePlan {
    pub(crate) fn new(
        width: usize,
        height: usize,
        depth: usize,
        slab_depth: usize,
        cross_connectivity: Connectivity,
    ) -> Result<Self> {
        Self::new_inner(width, height, depth, slab_depth, cross_connectivity, false)
    }

    pub(crate) fn new_hierarchical(
        width: usize,
        height: usize,
        depth: usize,
        slab_depth: usize,
        cross_connectivity: Connectivity,
    ) -> Result<Self> {
        Self::new_inner(width, height, depth, slab_depth, cross_connectivity, true)
    }

    fn new_inner(
        width: usize,
        height: usize,
        depth: usize,
        slab_depth: usize,
        cross_connectivity: Connectivity,
        hierarchical: bool,
    ) -> Result<Self> {
        if width == 0 || height == 0 || depth == 0 {
            bail!("volume dimensions must all be positive");
        }
        if slab_depth == 0 {
            bail!("slab depth must be positive");
        }

        let face_area = width
            .checked_mul(height)
            .ok_or_else(|| anyhow::anyhow!("width times height overflows usize"))?;
        let voxel_count = face_area
            .checked_mul(depth)
            .ok_or_else(|| anyhow::anyhow!("voxel count overflows usize"))?;
        let slab_count = depth.div_ceil(slab_depth);
        let maximum_slab_voxels = face_area
            .checked_mul(depth.min(slab_depth))
            .ok_or_else(|| anyhow::anyhow!("maximum slab voxel count overflows usize"))?;

        if maximum_slab_voxels > u32::MAX as usize {
            bail!(
                "one slab contains {maximum_slab_voxels} voxels, but local union-find IDs are u32; reduce slab_depth"
            );
        }

        let global_interface_nodes = global_interface_node_count(face_area, depth, slab_depth)?;
        if !hierarchical && global_interface_nodes > u32::MAX as usize {
            bail!(
                "the flat reducer needs {global_interface_nodes} global interface IDs, but IDs are u32; increase slab_depth or use h0-scalar-hierarchical"
            );
        }

        // A hierarchical composition combines at most two summaries, each of
        // which exposes no more than two z-faces.  Unlike the flat reducer,
        // its u32 identifier requirement therefore depends on at most four
        // faces, not on the total number of slabs.
        let hierarchical_pair_interface_nodes = if slab_count > 1 {
            face_area.checked_mul(4).ok_or_else(|| {
                anyhow::anyhow!("hierarchical pair interface count overflows usize")
            })?
        } else {
            global_interface_nodes
        };
        if hierarchical && hierarchical_pair_interface_nodes > u32::MAX as usize {
            bail!(
                "one hierarchical H0 pair can expose {hierarchical_pair_interface_nodes} interface nodes, but pairwise union-find IDs are u32; reduce the face dimensions or use a 64-bit hierarchical backend"
            );
        }

        let cross_vertex_count = if slab_count > 1 {
            face_area
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("cross-interface vertex count overflows usize"))?
        } else {
            0
        };

        if slab_count > 1
            && matches!(cross_connectivity, Connectivity::TwentySix)
            && cross_vertex_count > i32::MAX as usize
        {
            bail!(
                "a 26-connected cross interface has {cross_vertex_count} vertices, but its compact union-find supports at most {}; reduce the face dimensions or use a 64-bit interface backend",
                i32::MAX
            );
        }

        let cross_candidate_count = match cross_connectivity {
            Connectivity::Six => face_area,
            Connectivity::TwentySix => {
                let horizontal = width
                    .checked_mul(3)
                    .and_then(|value| value.checked_sub(2))
                    .ok_or_else(|| {
                        anyhow::anyhow!("26-neighbor interface width overflows usize")
                    })?;
                let vertical = height
                    .checked_mul(3)
                    .and_then(|value| value.checked_sub(2))
                    .ok_or_else(|| {
                        anyhow::anyhow!("26-neighbor interface height overflows usize")
                    })?;
                horizontal.checked_mul(vertical).ok_or_else(|| {
                    anyhow::anyhow!("26-neighbor cross-edge candidate count overflows usize")
                })?
            }
        };

        Ok(Self {
            width,
            height,
            depth,
            slab_depth,
            voxel_count,
            face_area,
            slab_count,
            maximum_slab_voxels,
            cross_candidate_count,
            cross_vertex_count,
            global_interface_nodes,
            hierarchical_pair_interface_nodes,
        })
    }

    pub(crate) fn print(&self) {
        println!("=== Checked size plan ===");
        println!(
            "volume: {} x {} x {} = {} voxels",
            self.width, self.height, self.depth, self.voxel_count
        );
        println!("face area A: {}", self.face_area);
        println!(
            "slabs: {} (requested depth {}, largest slab {} voxels)",
            self.slab_count, self.slab_depth, self.maximum_slab_voxels
        );
        if self.slab_count > 1 {
            println!(
                "one cross face: {} possible adjacency visits, {} activated vertices",
                self.cross_candidate_count, self.cross_vertex_count
            );
        }
        println!(
            "current global reconciliation state: {} interface nodes",
            self.global_interface_nodes
        );
        println!(
            "note: disk-backed modes bound run storage, but current global topology state still grows with slab count"
        );
        println!();
    }

    pub(crate) fn print_hierarchical(&self) {
        println!("=== Checked hierarchical size plan ===");
        println!(
            "volume: {} x {} x {} = {} voxels",
            self.width, self.height, self.depth, self.voxel_count
        );
        println!("face area A: {}", self.face_area);
        println!(
            "slabs: {} (requested depth {}, largest slab {} voxels)",
            self.slab_count, self.slab_depth, self.maximum_slab_voxels
        );
        if self.slab_count > 1 {
            println!(
                "one cross face: {} possible adjacency visits, {} activated vertices",
                self.cross_candidate_count, self.cross_vertex_count
            );
        }
        println!(
            "flat cumulative interface nodes (informational only): {}",
            self.global_interface_nodes
        );
        println!(
            "hierarchical pairwise frontier bound: {} interface nodes",
            self.hierarchical_pair_interface_nodes
        );
        println!(
            "note: hierarchical H0 validates u32 IDs against the pairwise frontier, not the cumulative slab count"
        );
        println!();
    }
}

fn global_interface_node_count(face_area: usize, depth: usize, slab_depth: usize) -> Result<usize> {
    let mut count = 0usize;
    let mut z0 = 0usize;
    while z0 < depth {
        let z1 = z0.saturating_add(slab_depth).min(depth);
        let local_depth = z1 - z0;
        let slab_nodes = if local_depth == 1 {
            face_area
        } else {
            face_area
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("slab interface count overflows usize"))?
        };
        count = count
            .checked_add(slab_nodes)
            .ok_or_else(|| anyhow::anyhow!("global interface count overflows usize"))?;
        z0 = z1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_one_slice_and_two_face_slabs_exactly() {
        let plan = SizePlan::new(3, 5, 7, 3, Connectivity::TwentySix).unwrap();
        assert_eq!(plan.face_area, 15);
        assert_eq!(plan.slab_count, 3);
        assert_eq!(plan.maximum_slab_voxels, 45);
        assert_eq!(plan.global_interface_nodes, 75);
        assert_eq!(plan.hierarchical_pair_interface_nodes, 60);
        assert_eq!(plan.cross_candidate_count, 91);
        assert_eq!(plan.cross_vertex_count, 30);
    }

    #[test]
    fn one_full_depth_slab_has_no_cross_vertices() {
        let plan = SizePlan::new(3, 5, 7, 20, Connectivity::TwentySix).unwrap();
        assert_eq!(plan.slab_count, 1);
        assert_eq!(plan.global_interface_nodes, 30);
        assert_eq!(plan.hierarchical_pair_interface_nodes, 30);
        assert_eq!(plan.cross_vertex_count, 0);
    }

    #[test]
    fn rejects_a_slab_that_exceeds_local_u32_identifiers() {
        let error = SizePlan::new(65_536, 65_536, 1, 1, Connectivity::Six).unwrap_err();
        assert!(error.to_string().contains("local union-find IDs are u32"));
    }

    #[test]
    fn rejects_global_interface_state_that_exceeds_u32_identifiers() {
        let error = SizePlan::new(65_536, 32_768, 2, 1, Connectivity::Six).unwrap_err();
        assert!(error.to_string().contains("flat reducer needs"));
    }

    #[test]
    fn hierarchical_plan_allows_cumulative_interface_count_above_u32() {
        let plan = SizePlan::new_hierarchical(50_000, 10_000, 10, 1, Connectivity::Six).unwrap();
        assert!(plan.global_interface_nodes > u32::MAX as usize);
        assert_eq!(plan.hierarchical_pair_interface_nodes, 2_000_000_000);
    }

    #[test]
    fn hierarchical_plan_rejects_pair_frontier_above_u32() {
        let error =
            SizePlan::new_hierarchical(65_536, 16_384, 8, 1, Connectivity::Six).unwrap_err();
        assert!(error.to_string().contains("hierarchical H0 pair"));
    }

    #[test]
    fn rejects_a_twenty_six_connected_face_that_exceeds_compact_ids() {
        let error = SizePlan::new(32_768, 32_768, 2, 1, Connectivity::TwentySix).unwrap_err();
        assert!(error.to_string().contains("compact union-find"));
    }

    #[test]
    fn reports_dimension_product_overflow_without_allocating() {
        let error = SizePlan::new(usize::MAX, 2, 1, 1, Connectivity::Six).unwrap_err();
        assert!(error.to_string().contains("width times height overflows"));
    }
}
