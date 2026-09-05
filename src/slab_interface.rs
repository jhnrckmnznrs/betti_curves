//! Deterministic numbering for the two boundary faces of one z-slab.
//!
//! The lower face uses IDs from zero to face_size minus one. If a slab has
//! more than one slice, the upper face uses the next face_size IDs; for a
//! one-slice slab the two faces are the same voxels and therefore share IDs.
//! Computing these IDs removes a full-slab optional-ID array from each local
//! reducer.

pub(crate) const NO_INTERFACE_REP: u32 = u32::MAX;

pub(crate) fn interface_node_count(face_size: usize, depth: usize) -> usize {
    if depth == 1 {
        face_size
    } else {
        face_size
            .checked_mul(2)
            .expect("slab interface node count overflow")
    }
}

pub(crate) fn local_boundary_node_id(
    z: usize,
    depth: usize,
    face_index: usize,
    face_size: usize,
) -> Option<u32> {
    if z == 0 {
        Some(u32::try_from(face_index).expect("face index exceeds u32"))
    } else if z + 1 == depth {
        Some(u32::try_from(face_size + face_index).expect("upper-face interface index exceeds u32"))
    } else {
        None
    }
}

pub(crate) fn face_node_id(z: usize, depth: usize, face_index: usize, face_size: usize) -> u32 {
    local_boundary_node_id(z, depth, face_index, face_size)
        .expect("requested slice is not a slab boundary face")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slice_faces_share_ids() {
        assert_eq!(interface_node_count(12, 1), 12);
        assert_eq!(face_node_id(0, 1, 7, 12), 7);
    }

    #[test]
    fn two_distinct_faces_have_disjoint_ids() {
        assert_eq!(interface_node_count(12, 3), 24);
        assert_eq!(face_node_id(0, 3, 7, 12), 7);
        assert_eq!(face_node_id(2, 3, 7, 12), 19);
        assert_eq!(local_boundary_node_id(1, 3, 7, 12), None);
    }
}
