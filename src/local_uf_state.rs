use crate::scalar_stream_tuning::{ActiveStateStrategy, UnionFindLayoutStrategy};

const PACKED_ROOT_TAG: u32 = 1 << 31;
const PACKED_RANK_MASK: u32 = 0xff;
const INACTIVE_WORD: u32 = u32::MAX;

#[derive(Debug)]
pub(crate) struct LocalUnionFindState {
    parent: Vec<u32>,
    rank: Option<Vec<u8>>,
    layout: UnionFindLayoutStrategy,
    active_state: ActiveStateStrategy,
}

impl LocalUnionFindState {
    pub(crate) fn new(
        len: usize,
        active_state: ActiveStateStrategy,
        layout: UnionFindLayoutStrategy,
    ) -> Self {
        match layout {
            UnionFindLayoutStrategy::ParentRank => {
                assert!(
                    len <= u32::MAX as usize,
                    "parent-rank local union-find reserves u32::MAX as the inactive sentinel"
                );
                let parent = match active_state {
                    ActiveStateStrategy::Separate => (0..len as u32).collect(),
                    ActiveStateStrategy::ParentSentinel => vec![INACTIVE_WORD; len],
                };
                Self {
                    parent,
                    rank: Some(vec![0; len]),
                    layout,
                    active_state,
                }
            }
            UnionFindLayoutStrategy::Packed => {
                assert!(
                    len <= PACKED_ROOT_TAG as usize,
                    "packed local union-find supports at most 2^31 voxels per slab"
                );
                let initial = match active_state {
                    ActiveStateStrategy::Separate => Self::packed_root_word(0),
                    ActiveStateStrategy::ParentSentinel => INACTIVE_WORD,
                };
                Self {
                    parent: vec![initial; len],
                    rank: None,
                    layout,
                    active_state,
                }
            }
        }
    }

    #[inline]
    const fn packed_root_word(rank: u8) -> u32 {
        PACKED_ROOT_TAG | rank as u32
    }

    #[inline]
    fn packed_word_is_root(word: u32) -> bool {
        word != INACTIVE_WORD && (word & PACKED_ROOT_TAG) != 0
    }

    #[inline]
    pub(crate) fn activate(&mut self, node: u32) {
        if matches!(self.active_state, ActiveStateStrategy::ParentSentinel) {
            let slot = &mut self.parent[node as usize];
            debug_assert_eq!(*slot, INACTIVE_WORD);
            *slot = match self.layout {
                UnionFindLayoutStrategy::ParentRank => node,
                UnionFindLayoutStrategy::Packed => Self::packed_root_word(0),
            };
        }
    }

    #[inline]
    pub(crate) fn is_active(&self, node: usize) -> bool {
        debug_assert!(matches!(
            self.active_state,
            ActiveStateStrategy::ParentSentinel
        ));
        self.parent[node] != INACTIVE_WORD
    }

    #[inline]
    pub(crate) fn is_root(&self, node: u32) -> bool {
        let word = self.parent[node as usize];
        match self.layout {
            UnionFindLayoutStrategy::ParentRank => word == node,
            UnionFindLayoutStrategy::Packed => Self::packed_word_is_root(word),
        }
    }

    #[inline]
    pub(crate) fn direct_parent_is(&self, node: u32, root: u32) -> bool {
        node != root && !self.is_root(node) && self.parent[node as usize] == root
    }

    /// Return the already-stored parent for a non-root node.
    ///
    /// This is used by the cached-find H2 hot path so the parent word loaded for
    /// the direct-parent check can be reused by path halving instead of being
    /// fetched again inside `find`. `None` means that `node` is already a root.
    #[inline]
    pub(crate) fn parent_if_nonroot(&self, node: u32) -> Option<u32> {
        let word = self.parent[node as usize];
        debug_assert_ne!(word, INACTIVE_WORD);
        match self.layout {
            UnionFindLayoutStrategy::ParentRank => (word != node).then_some(word),
            UnionFindLayoutStrategy::Packed => {
                if Self::packed_word_is_root(word) {
                    None
                } else {
                    Some(word)
                }
            }
        }
    }

    /// Returns true only when `root` is exactly two parent edges above `node`.
    ///
    /// This is a read-only hot-path shortcut: it never performs path compression
    /// and is valid for both parent-rank and packed layouts. Packed roots store a
    /// tagged rank word in their own parent slot, so we must not index through a
    /// parent that is already a root.
    #[inline]
    pub(crate) fn grandparent_is(&self, node: u32, root: u32) -> bool {
        if node == root || self.is_root(node) {
            return false;
        }
        let parent = self.parent[node as usize];
        if parent == root || self.is_root(parent) {
            return false;
        }
        self.parent[parent as usize] == root
    }

    /// Continue path-halving find after the caller has already loaded the first
    /// parent of a known non-root node. This preserves the same path-halving
    /// updates and step count as `find(node)` while avoiding the duplicate
    /// initial root/parent load in the caller+find sequence.
    #[inline]
    pub(crate) fn find_from_known_parent(&mut self, mut node: u32, mut parent: u32) -> (u32, u64) {
        debug_assert_eq!(self.parent_if_nonroot(node), Some(parent));
        let mut steps = 1u64;
        loop {
            if self.is_root(parent) {
                return (parent, steps);
            }
            let grandparent = self.parent[parent as usize];
            debug_assert!(
                grandparent < PACKED_ROOT_TAG
                    || matches!(self.layout, UnionFindLayoutStrategy::ParentRank)
            );
            self.parent[node as usize] = grandparent;
            node = parent;
            parent = grandparent;
            steps += 1;
        }
    }

    /// Find with path halving. Returns `(root, parent_edges_traversed)`.
    pub(crate) fn find(&mut self, mut node: u32) -> (u32, u64) {
        debug_assert!(self.parent[node as usize] != INACTIVE_WORD);
        let mut steps = 0u64;
        while !self.is_root(node) {
            steps += 1;
            let parent = self.parent[node as usize];
            debug_assert!(
                parent < PACKED_ROOT_TAG
                    || matches!(self.layout, UnionFindLayoutStrategy::ParentRank)
            );
            if !self.is_root(parent) {
                let grandparent = self.parent[parent as usize];
                debug_assert!(
                    grandparent < PACKED_ROOT_TAG
                        || matches!(self.layout, UnionFindLayoutStrategy::ParentRank)
                );
                self.parent[node as usize] = grandparent;
            }
            node = parent;
        }
        (node, steps)
    }

    #[inline]
    pub(crate) fn root_rank(&self, root: u32) -> u8 {
        debug_assert!(self.is_root(root));
        match self.layout {
            UnionFindLayoutStrategy::ParentRank => self
                .rank
                .as_ref()
                .expect("parent-rank layout requires rank vector")[root as usize],
            UnionFindLayoutStrategy::Packed => {
                (self.parent[root as usize] & PACKED_RANK_MASK) as u8
            }
        }
    }

    #[inline]
    pub(crate) fn set_root_rank(&mut self, root: u32, rank: u8) {
        debug_assert!(self.is_root(root));
        match self.layout {
            UnionFindLayoutStrategy::ParentRank => {
                self.rank
                    .as_mut()
                    .expect("parent-rank layout requires rank vector")[root as usize] = rank;
            }
            UnionFindLayoutStrategy::Packed => {
                self.parent[root as usize] = Self::packed_root_word(rank);
            }
        }
    }

    #[inline]
    pub(crate) fn link_root_under(&mut self, child_root: u32, parent_root: u32) {
        debug_assert!(self.is_root(child_root));
        debug_assert!(self.is_root(parent_root));
        debug_assert!(
            parent_root < PACKED_ROOT_TAG
                || matches!(self.layout, UnionFindLayoutStrategy::ParentRank)
        );
        self.parent[child_root as usize] = parent_root;
    }

    #[inline]
    pub(crate) fn rank_state_bytes(&self) -> u64 {
        self.rank
            .as_ref()
            .map(|rank| rank.len() as u64)
            .unwrap_or(0)
    }

    #[inline]
    pub(crate) fn parent_state_bytes(&self) -> u64 {
        self.parent.len() as u64 * core::mem::size_of::<u32>() as u64
    }

    #[inline]
    pub(crate) fn parent_capacity_bytes(&self) -> u64 {
        self.parent.capacity() as u64 * core::mem::size_of::<u32>() as u64
    }

    #[inline]
    pub(crate) fn rank_capacity_bytes(&self) -> u64 {
        self.rank
            .as_ref()
            .map(|rank| rank.capacity() as u64 * core::mem::size_of::<u8>() as u64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise(layout: UnionFindLayoutStrategy, active: ActiveStateStrategy) {
        let mut uf = LocalUnionFindState::new(16, active, layout);
        if matches!(active, ActiveStateStrategy::ParentSentinel) {
            for node in 0..16u32 {
                assert!(!uf.is_active(node as usize));
                uf.activate(node);
                assert!(uf.is_active(node as usize));
            }
        }
        assert!(uf.is_root(0));
        assert_eq!(uf.root_rank(0), 0);
        uf.set_root_rank(0, 1);
        assert_eq!(uf.root_rank(0), 1);
        uf.link_root_under(1, 0);
        assert!(uf.direct_parent_is(1, 0));
        let (root, steps) = uf.find(1);
        assert_eq!(root, 0);
        assert_eq!(steps, 1);
    }

    #[test]
    fn parent_rank_layout_works_with_both_active_states() {
        exercise(
            UnionFindLayoutStrategy::ParentRank,
            ActiveStateStrategy::Separate,
        );
        exercise(
            UnionFindLayoutStrategy::ParentRank,
            ActiveStateStrategy::ParentSentinel,
        );
    }

    #[test]
    fn packed_layout_works_with_both_active_states() {
        exercise(
            UnionFindLayoutStrategy::Packed,
            ActiveStateStrategy::Separate,
        );
        exercise(
            UnionFindLayoutStrategy::Packed,
            ActiveStateStrategy::ParentSentinel,
        );
    }

    fn exercise_grandparent(layout: UnionFindLayoutStrategy, active: ActiveStateStrategy) {
        let mut uf = LocalUnionFindState::new(4, active, layout);
        if matches!(active, ActiveStateStrategy::ParentSentinel) {
            for node in 0..4u32 {
                uf.activate(node);
            }
        }
        uf.link_root_under(2, 1);
        uf.link_root_under(1, 0);
        assert!(!uf.direct_parent_is(2, 0));
        assert!(uf.grandparent_is(2, 0));
        assert!(!uf.grandparent_is(1, 0));
        assert!(!uf.grandparent_is(0, 0));
        assert!(!uf.grandparent_is(3, 0));
    }

    #[test]
    fn grandparent_shortcut_works_for_all_local_uf_layouts() {
        for layout in [
            UnionFindLayoutStrategy::ParentRank,
            UnionFindLayoutStrategy::Packed,
        ] {
            for active in [
                ActiveStateStrategy::Separate,
                ActiveStateStrategy::ParentSentinel,
            ] {
                exercise_grandparent(layout, active);
            }
        }
    }

    fn exercise_cached_find(layout: UnionFindLayoutStrategy, active: ActiveStateStrategy) {
        let mut uf = LocalUnionFindState::new(6, active, layout);
        if matches!(active, ActiveStateStrategy::ParentSentinel) {
            for node in 0..6u32 {
                uf.activate(node);
            }
        }

        uf.link_root_under(3, 2);
        uf.link_root_under(2, 1);
        uf.link_root_under(1, 0);

        assert_eq!(uf.parent_if_nonroot(0), None);
        let first_parent = uf.parent_if_nonroot(3).expect("3 must be non-root");
        assert_eq!(first_parent, 2);
        let (root, steps) = uf.find_from_known_parent(3, first_parent);
        assert_eq!(root, 0);
        assert_eq!(steps, 3);

        let (root_again, steps_again) = uf.find(3);
        assert_eq!(root_again, 0);
        assert!(steps_again <= 2);
    }

    #[test]
    fn cached_parent_find_works_for_all_local_uf_layouts() {
        for layout in [
            UnionFindLayoutStrategy::ParentRank,
            UnionFindLayoutStrategy::Packed,
        ] {
            for active in [
                ActiveStateStrategy::Separate,
                ActiveStateStrategy::ParentSentinel,
            ] {
                exercise_cached_find(layout, active);
            }
        }
    }

    #[test]
    fn packed_layout_uses_no_rank_vector() {
        let uf = LocalUnionFindState::new(
            123,
            ActiveStateStrategy::Separate,
            UnionFindLayoutStrategy::Packed,
        );
        assert_eq!(uf.rank_state_bytes(), 0);
        assert_eq!(uf.parent_state_bytes(), 123 * 4);
    }
}
