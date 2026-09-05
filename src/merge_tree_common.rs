use anyhow::{Result, bail};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::atomic_output::AtomicOutput;

pub(crate) const OUTSIDE_BRANCH_ID: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct H0Branch {
    pub(crate) id: u64,
    pub(crate) birth: u16,
}

impl H0Branch {
    pub(crate) fn older(self, other: Self) -> Self {
        if (self.birth, self.id) <= (other.birth, other.id) {
            self
        } else {
            other
        }
    }

    pub(crate) fn younger(self, other: Self) -> Self {
        if self.older(other) == self {
            other
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H2Branch {
    Outside,
    Finite { id: u64, birth: u16 },
}

impl H2Branch {
    pub(crate) fn older(self, other: Self) -> Self {
        match (self, other) {
            (Self::Outside, _) | (_, Self::Outside) => Self::Outside,
            (
                Self::Finite {
                    id: a_id,
                    birth: a_birth,
                },
                Self::Finite {
                    id: b_id,
                    birth: b_birth,
                },
            ) => {
                if a_birth > b_birth || (a_birth == b_birth && a_id <= b_id) {
                    self
                } else {
                    other
                }
            }
        }
    }

    pub(crate) fn younger(self, other: Self) -> Option<(u64, u16)> {
        let older = self.older(other);
        let younger = if older == self { other } else { self };

        match younger {
            Self::Outside => None,
            Self::Finite { id, birth } => Some((id, birth)),
        }
    }

    pub(crate) fn finite_id(self) -> Option<u64> {
        match self {
            Self::Outside => None,
            Self::Finite { id, .. } => Some(id),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct H0BranchMerge {
    pub(crate) value: u16,
    pub(crate) child: H0Branch,
    pub(crate) parent: H0Branch,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct H2BranchMerge {
    pub(crate) value: u16,
    pub(crate) child_id: u64,
    pub(crate) child_birth: u16,
    pub(crate) parent: H2Branch,
}

/// Compact representation used only for hierarchical finalized-event buckets.
/// The threshold/death value is supplied by the bucket index and is therefore
/// intentionally omitted from every stored event.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactH0BranchMerge {
    pub(crate) child_id: u64,
    pub(crate) parent_id: u64,
    pub(crate) child_birth: u16,
    pub(crate) parent_birth: u16,
}

impl CompactH0BranchMerge {
    pub(crate) fn from_merge(merge: H0BranchMerge) -> Self {
        Self {
            child_id: merge.child.id,
            parent_id: merge.parent.id,
            child_birth: merge.child.birth,
            parent_birth: merge.parent.birth,
        }
    }

    pub(crate) fn expand(self, value: u16) -> H0BranchMerge {
        H0BranchMerge {
            value,
            child: H0Branch {
                id: self.child_id,
                birth: self.child_birth,
            },
            parent: H0Branch {
                id: self.parent_id,
                birth: self.parent_birth,
            },
        }
    }
}

/// Compact representation used only for hierarchical finalized-event buckets.
/// `parent_id == OUTSIDE_BRANCH_ID` encodes the outside H2 root. The bucket
/// index supplies the threshold/death value.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactH2BranchMerge {
    pub(crate) child_id: u64,
    pub(crate) parent_id: u64,
    pub(crate) child_birth: u16,
    pub(crate) parent_birth: u16,
}

impl CompactH2BranchMerge {
    pub(crate) fn from_merge(merge: H2BranchMerge) -> Self {
        let (parent_id, parent_birth) = match merge.parent {
            H2Branch::Outside => (OUTSIDE_BRANCH_ID, 0),
            H2Branch::Finite { id, birth } => (id, birth),
        };
        Self {
            child_id: merge.child_id,
            parent_id,
            child_birth: merge.child_birth,
            parent_birth,
        }
    }

    pub(crate) fn parent(self) -> H2Branch {
        if self.parent_id == OUTSIDE_BRANCH_ID {
            H2Branch::Outside
        } else {
            H2Branch::Finite {
                id: self.parent_id,
                birth: self.parent_birth,
            }
        }
    }

    pub(crate) fn expand(self, value: u16) -> H2BranchMerge {
        H2BranchMerge {
            value,
            child_id: self.child_id,
            child_birth: self.child_birth,
            parent: self.parent(),
        }
    }
}

const PACKED_EMPTY_KEY: u64 = u64::MAX;
const PACKED_INITIAL_CAPACITY: usize = 16;
const PACKED_LOAD_NUMERATOR: usize = 7;
const PACKED_LOAD_DENOMINATOR: usize = 10;

#[inline]
fn packed_hash(key: u64) -> usize {
    // SplitMix64 finalizer: deterministic and fast for monotone voxel IDs.
    let mut z = key.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) as usize
}

/// Reusable open-addressed `u64 -> u64` map for one filtration threshold.
///
/// Occupancy is tracked with a generation stamp, so resetting the table for
/// the next filtration threshold is O(1) instead of scanning populated slots
/// or rebuilding a `HashMap` up to 65,536 times.
#[derive(Debug)]
pub(crate) struct PackedU64Map {
    keys: Vec<u64>,
    values: Vec<u64>,
    generations: Vec<u32>,
    generation: u32,
    len: usize,
}

impl Default for PackedU64Map {
    fn default() -> Self {
        Self::with_capacity(PACKED_INITIAL_CAPACITY)
    }
}

impl PackedU64Map {
    fn with_capacity(requested: usize) -> Self {
        let capacity = requested.max(PACKED_INITIAL_CAPACITY).next_power_of_two();
        Self {
            keys: vec![0; capacity],
            values: vec![0; capacity],
            generations: vec![0; capacity],
            generation: 1,
            len: 0,
        }
    }

    #[inline]
    fn slot_for(&self, key: u64) -> (usize, bool) {
        let mask = self.keys.len() - 1;
        let mut slot = packed_hash(key) & mask;
        loop {
            if self.generations[slot] != self.generation {
                return (slot, false);
            }
            if self.keys[slot] == key {
                return (slot, true);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn ensure_insert_capacity(&mut self) {
        if (self.len + 1) * PACKED_LOAD_DENOMINATOR <= self.keys.len() * PACKED_LOAD_NUMERATOR {
            return;
        }
        self.grow();
    }

    fn grow(&mut self) {
        let new_capacity = self
            .keys
            .len()
            .checked_mul(2)
            .expect("packed map too large");
        let old_keys = std::mem::replace(&mut self.keys, vec![0; new_capacity]);
        let old_values = std::mem::replace(&mut self.values, vec![0; new_capacity]);
        let old_generations = std::mem::replace(&mut self.generations, vec![0; new_capacity]);
        let old_generation = self.generation;
        self.generation = 1;
        self.len = 0;

        for (slot, &key) in old_keys.iter().enumerate() {
            if old_generations[slot] == old_generation {
                self.insert_no_grow(key, old_values[slot]);
            }
        }
    }

    #[inline]
    fn insert_no_grow(&mut self, key: u64, value: u64) {
        let (slot, present) = self.slot_for(key);
        if !present {
            self.generations[slot] = self.generation;
            self.keys[slot] = key;
            self.len += 1;
        }
        self.values[slot] = value;
    }

    #[inline]
    pub(crate) fn insert(&mut self, key: u64, value: u64) {
        let (slot, present) = self.slot_for(key);
        if present {
            self.values[slot] = value;
            return;
        }
        self.ensure_insert_capacity();
        self.insert_no_grow(key, value);
    }

    #[inline]
    pub(crate) fn get(&self, key: u64) -> Option<u64> {
        let (slot, present) = self.slot_for(key);
        present.then_some(self.values[slot])
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Generation wrap is practically unreachable in one reduction
            // (there are only 65,536 thresholds), but keep the table correct
            // for arbitrary reuse.
            self.generations.fill(0);
            self.generation = 1;
        }
    }
}

/// Packed deferred-parent state for the hierarchical root reducer.
///
/// The old implementation maintained both a `HashSet<u64>` of watched IDs and
/// a `HashMap<u64, Branch>` of redirects.  Here one open-addressed table stores
/// the watched key and, when present, its redirected parent ID.  Parent birth
/// values are unnecessary in the hierarchical root replay: duplicate
/// suppression, plateau contraction, and exported parent links depend only on
/// branch identity.  `u64::MAX` may appear as a redirect value for H2 Outside,
/// but never as a finite watched key.
#[derive(Debug)]
pub(crate) struct PackedDeferredIdResolver {
    keys: Vec<u64>,
    redirects: Vec<u64>,
    has_redirect: Vec<u8>,
    len: usize,
}

impl Default for PackedDeferredIdResolver {
    fn default() -> Self {
        Self::with_capacity(PACKED_INITIAL_CAPACITY)
    }
}

impl PackedDeferredIdResolver {
    fn with_capacity(requested: usize) -> Self {
        let capacity = requested.max(PACKED_INITIAL_CAPACITY).next_power_of_two();
        Self {
            keys: vec![PACKED_EMPTY_KEY; capacity],
            redirects: vec![0; capacity],
            has_redirect: vec![0; capacity],
            len: 0,
        }
    }

    #[inline]
    fn slot_for(&self, key: u64) -> (usize, bool) {
        debug_assert_ne!(key, PACKED_EMPTY_KEY);
        let mask = self.keys.len() - 1;
        let mut slot = packed_hash(key) & mask;
        loop {
            let existing = self.keys[slot];
            if existing == PACKED_EMPTY_KEY {
                return (slot, false);
            }
            if existing == key {
                return (slot, true);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn ensure_insert_capacity(&mut self) {
        if (self.len + 1) * PACKED_LOAD_DENOMINATOR <= self.keys.len() * PACKED_LOAD_NUMERATOR {
            return;
        }
        self.grow();
    }

    fn grow(&mut self) {
        let new_capacity = self
            .keys
            .len()
            .checked_mul(2)
            .expect("packed deferred-parent table too large");
        let old_keys = std::mem::replace(&mut self.keys, vec![PACKED_EMPTY_KEY; new_capacity]);
        let old_redirects = std::mem::replace(&mut self.redirects, vec![0; new_capacity]);
        let old_has_redirect = std::mem::replace(&mut self.has_redirect, vec![0; new_capacity]);
        self.len = 0;

        for (slot, &key) in old_keys.iter().enumerate() {
            if key == PACKED_EMPTY_KEY {
                continue;
            }
            let (new_slot, present) = self.slot_for(key);
            debug_assert!(!present);
            self.keys[new_slot] = key;
            self.redirects[new_slot] = old_redirects[slot];
            self.has_redirect[new_slot] = old_has_redirect[slot];
            self.len += 1;
        }
    }

    #[inline]
    pub(crate) fn watch(&mut self, key: u64) {
        if key == PACKED_EMPTY_KEY {
            // H2 Outside is terminal and never needs to be watched.
            return;
        }
        let (_, present) = self.slot_for(key);
        if present {
            return;
        }
        self.ensure_insert_capacity();
        let (slot, present) = self.slot_for(key);
        debug_assert!(!present);
        self.keys[slot] = key;
        self.has_redirect[slot] = 0;
        self.len += 1;
    }

    #[inline]
    pub(crate) fn resolve_id(&self, mut id: u64) -> Result<u64> {
        let mut steps = 0usize;
        while id != PACKED_EMPTY_KEY {
            let (slot, present) = self.slot_for(id);
            if !present || self.has_redirect[slot] == 0 {
                return Ok(id);
            }
            id = self.redirects[slot];
            steps += 1;
            if steps > self.len {
                bail!("cycle detected while resolving packed deferred branch parents");
            }
        }
        Ok(id)
    }

    #[inline]
    pub(crate) fn observe(&mut self, child_id: u64, parent_id: u64) {
        if child_id == PACKED_EMPTY_KEY {
            return;
        }
        let (slot, present) = self.slot_for(child_id);
        if !present {
            return;
        }
        self.redirects[slot] = parent_id;
        self.has_redirect[slot] = 1;
        self.watch(parent_id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone)]
struct H0Node {
    original_id: u64,
    birth: u16,
    death: Option<u16>,
    parent: Option<u64>,
}

#[derive(Debug, Clone)]
struct H2Node {
    original_id: u64,
    birth: Option<u16>,
    death: Option<u16>,
    parent: Option<u64>,
    outside: bool,
}

#[derive(Debug, Clone)]
pub struct MergeTreeRow {
    pub from_node: u64,
    pub to_node: u64,
    pub from_birth_value: String,
    pub from_death_value: String,
    pub to_birth_value: String,
    pub to_death_value: String,
}

#[derive(Debug, Clone)]
pub struct MergeTreeNodeRow {
    pub node: u64,
    pub parent: Option<u64>,
    pub birth_value: String,
    pub death_value: String,
}

#[derive(Debug, Clone)]
pub struct MergeTree {
    /// This structure is an elder-rule branch-decomposition tree. It is not a
    /// canonical merge tree when several old components merge on one plateau:
    /// the deterministic within-value event order chooses the parent chain.
    pub node_count: usize,
    pub nodes: Vec<MergeTreeNodeRow>,
    pub rows: Vec<MergeTreeRow>,
}

#[derive(Debug, Default)]
pub(crate) struct H0TreeRecorder {
    nodes: HashMap<u64, H0Node>,
    diagonal_parent: HashMap<u64, u64>,
    pending_positive: Vec<H0BranchMerge>,
}

impl H0TreeRecorder {
    pub(crate) fn record_merge(&mut self, merge: H0BranchMerge) -> Result<()> {
        if merge.child.birth > merge.value {
            bail!(
                "invalid H0 merge: child birth {} exceeds death {}",
                merge.child.birth,
                merge.value
            );
        }

        if merge.child.birth == merge.value {
            self.diagonal_parent.insert(merge.child.id, merge.parent.id);
        } else {
            self.pending_positive.push(merge);
        }

        Ok(())
    }

    /// Record one compact hierarchical H0 event without first expanding it
    /// into the general-purpose merge representation. Zero-persistence events
    /// stay compact all the way through plateau contraction; only exported
    /// positive-persistence branches are materialized as `H0BranchMerge`.
    pub(crate) fn record_compact_merge(
        &mut self,
        value: u16,
        merge: CompactH0BranchMerge,
        repaired_parent: H0Branch,
    ) -> Result<()> {
        if merge.child_birth > value {
            bail!(
                "invalid H0 merge: child birth {} exceeds death {}",
                merge.child_birth,
                value
            );
        }

        if merge.child_birth == value {
            self.diagonal_parent
                .insert(merge.child_id, repaired_parent.id);
        } else {
            self.pending_positive.push(H0BranchMerge {
                value,
                child: H0Branch {
                    id: merge.child_id,
                    birth: merge.child_birth,
                },
                parent: repaired_parent,
            });
        }

        Ok(())
    }

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent = resolve_h0_parent(merge.parent.id, &self.diagonal_parent)?;
            if let Some(previous) = self.nodes.get(&merge.child.id) {
                bail!(
                    "H0 branch {} died more than once: previous death={:?} parent={:?}, repeated death={} parent={}",
                    merge.child.id,
                    previous.death,
                    previous.parent,
                    merge.value,
                    parent
                );
            }
            self.nodes.insert(
                merge.child.id,
                H0Node {
                    original_id: merge.child.id,
                    birth: merge.child.birth,
                    death: Some(merge.value),
                    parent: Some(parent),
                },
            );
        }

        self.diagonal_parent.clear();
        Ok(())
    }

    pub(crate) fn add_essential(&mut self, branch: H0Branch) -> Result<()> {
        let previous = self.nodes.insert(
            branch.id,
            H0Node {
                original_id: branch.id,
                birth: branch.birth,
                death: None,
                parent: None,
            },
        );

        if previous.is_some() {
            bail!("H0 essential branch {} already has a death", branch.id);
        }

        Ok(())
    }

    pub(crate) fn into_tree(self) -> Result<MergeTree> {
        let mut nodes: Vec<H0Node> = self.nodes.into_values().collect();
        nodes.sort_by_key(|node| node.original_id);

        let dense_ids: HashMap<u64, u64> = nodes
            .iter()
            .enumerate()
            .map(|(dense, node)| (node.original_id, dense as u64))
            .collect();

        let by_id: HashMap<u64, &H0Node> =
            nodes.iter().map(|node| (node.original_id, node)).collect();

        let mut node_rows = Vec::with_capacity(nodes.len());
        let mut rows = Vec::new();
        for node in &nodes {
            let parent = match node.parent {
                Some(parent_id) => Some(*by_id.get(&parent_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "H0 merge-tree parent {} for branch {} was not retained",
                        parent_id,
                        node.original_id
                    )
                })?),
                None => None,
            };

            node_rows.push(MergeTreeNodeRow {
                node: dense_ids[&node.original_id],
                parent: node.parent.map(|parent_id| dense_ids[&parent_id]),
                birth_value: node.birth.to_string(),
                death_value: display_death(node.death),
            });

            if let Some(parent) = parent {
                rows.push(MergeTreeRow {
                    from_node: dense_ids[&node.original_id],
                    to_node: dense_ids[&parent.original_id],
                    from_birth_value: node.birth.to_string(),
                    from_death_value: display_death(node.death),
                    to_birth_value: parent.birth.to_string(),
                    to_death_value: display_death(parent.death),
                });
            }
        }

        rows.sort_by_key(|row| row.from_node);
        Ok(MergeTree {
            node_count: nodes.len(),
            nodes: node_rows,
            rows,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct H0PackedPositiveMerge {
    value: u16,
    child_id: u64,
    child_birth: u16,
    parent_id: u64,
}

/// Hierarchical-root recorder using packed integer state for plateau
/// contraction. The flat/in-memory reducer deliberately keeps using
/// `H0TreeRecorder`, so exact equivalence remains independently testable.
#[derive(Debug, Default)]
pub(crate) struct H0PackedTreeRecorder {
    nodes: HashMap<u64, H0Node>,
    diagonal_parent: PackedU64Map,
    pending_positive: Vec<H0PackedPositiveMerge>,
}

impl H0PackedTreeRecorder {
    pub(crate) fn record_merge(&mut self, merge: H0BranchMerge) -> Result<()> {
        if merge.child.birth > merge.value {
            bail!(
                "invalid H0 merge: child birth {} exceeds death {}",
                merge.child.birth,
                merge.value
            );
        }
        if merge.child.birth == merge.value {
            self.diagonal_parent.insert(merge.child.id, merge.parent.id);
        } else {
            self.pending_positive.push(H0PackedPositiveMerge {
                value: merge.value,
                child_id: merge.child.id,
                child_birth: merge.child.birth,
                parent_id: merge.parent.id,
            });
        }
        Ok(())
    }

    pub(crate) fn record_compact_merge_parent_id(
        &mut self,
        value: u16,
        merge: CompactH0BranchMerge,
        repaired_parent_id: u64,
    ) -> Result<()> {
        if merge.child_birth > value {
            bail!(
                "invalid H0 merge: child birth {} exceeds death {}",
                merge.child_birth,
                value
            );
        }
        if merge.child_birth == value {
            self.diagonal_parent
                .insert(merge.child_id, repaired_parent_id);
        } else {
            self.pending_positive.push(H0PackedPositiveMerge {
                value,
                child_id: merge.child_id,
                child_birth: merge.child_birth,
                parent_id: repaired_parent_id,
            });
        }
        Ok(())
    }

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent = resolve_h0_parent_packed(merge.parent_id, &self.diagonal_parent)?;
            if let Some(previous) = self.nodes.get(&merge.child_id) {
                bail!(
                    "H0 branch {} died more than once: previous death={:?} parent={:?}, repeated death={} parent={}",
                    merge.child_id,
                    previous.death,
                    previous.parent,
                    merge.value,
                    parent
                );
            }
            self.nodes.insert(
                merge.child_id,
                H0Node {
                    original_id: merge.child_id,
                    birth: merge.child_birth,
                    death: Some(merge.value),
                    parent: Some(parent),
                },
            );
        }
        self.diagonal_parent.clear();
        Ok(())
    }

    pub(crate) fn add_essential(&mut self, branch: H0Branch) -> Result<()> {
        let previous = self.nodes.insert(
            branch.id,
            H0Node {
                original_id: branch.id,
                birth: branch.birth,
                death: None,
                parent: None,
            },
        );
        if previous.is_some() {
            bail!("H0 essential branch {} already has a death", branch.id);
        }
        Ok(())
    }

    pub(crate) fn into_tree(self) -> Result<MergeTree> {
        let mut nodes: Vec<H0Node> = self.nodes.into_values().collect();
        nodes.sort_by_key(|node| node.original_id);

        let dense_ids: HashMap<u64, u64> = nodes
            .iter()
            .enumerate()
            .map(|(dense, node)| (node.original_id, dense as u64))
            .collect();
        let by_id: HashMap<u64, &H0Node> =
            nodes.iter().map(|node| (node.original_id, node)).collect();

        let mut node_rows = Vec::with_capacity(nodes.len());
        let mut rows = Vec::new();
        for node in &nodes {
            let parent = match node.parent {
                Some(parent_id) => Some(*by_id.get(&parent_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "H0 merge-tree parent {} for branch {} was not retained",
                        parent_id,
                        node.original_id
                    )
                })?),
                None => None,
            };

            node_rows.push(MergeTreeNodeRow {
                node: dense_ids[&node.original_id],
                parent: node.parent.map(|parent_id| dense_ids[&parent_id]),
                birth_value: node.birth.to_string(),
                death_value: display_death(node.death),
            });
            if let Some(parent) = parent {
                rows.push(MergeTreeRow {
                    from_node: dense_ids[&node.original_id],
                    to_node: dense_ids[&parent.original_id],
                    from_birth_value: node.birth.to_string(),
                    from_death_value: display_death(node.death),
                    to_birth_value: parent.birth.to_string(),
                    to_death_value: display_death(parent.death),
                });
            }
        }
        rows.sort_by_key(|row| row.from_node);
        Ok(MergeTree {
            node_count: nodes.len(),
            nodes: node_rows,
            rows,
        })
    }
}

fn resolve_h0_parent_packed(mut id: u64, diagonal_parent: &PackedU64Map) -> Result<u64> {
    let mut steps = 0usize;
    while let Some(parent) = diagonal_parent.get(id) {
        id = parent;
        steps += 1;
        if steps > diagonal_parent.len() {
            bail!("cycle detected while contracting packed diagonal H0 branches");
        }
    }
    Ok(id)
}

fn resolve_h0_parent(mut id: u64, diagonal_parent: &HashMap<u64, u64>) -> Result<u64> {
    let mut steps = 0usize;
    while let Some(&parent) = diagonal_parent.get(&id) {
        id = parent;
        steps += 1;
        if steps > diagonal_parent.len() {
            bail!("cycle detected while contracting diagonal H0 branches");
        }
    }
    Ok(id)
}

#[derive(Debug, Default)]
pub(crate) struct H2TreeRecorder {
    nodes: HashMap<u64, H2Node>,
    diagonal_parent: HashMap<u64, H2Branch>,
    pending_positive: Vec<H2BranchMerge>,
}

impl H2TreeRecorder {
    pub(crate) fn record_merge(&mut self, merge: H2BranchMerge) -> Result<()> {
        if merge.value > merge.child_birth {
            bail!(
                "invalid H2 merge: foreground birth {} exceeds death {}",
                merge.value,
                merge.child_birth
            );
        }

        if merge.value == merge.child_birth {
            self.diagonal_parent.insert(merge.child_id, merge.parent);
        } else {
            self.pending_positive.push(merge);
        }

        Ok(())
    }

    /// Record one compact hierarchical H2 event without expanding every
    /// zero-persistence event into `H2BranchMerge`. The repaired parent is
    /// supplied by the deferred-parent resolver for the current threshold.
    pub(crate) fn record_compact_merge(
        &mut self,
        value: u16,
        merge: CompactH2BranchMerge,
        repaired_parent: H2Branch,
    ) -> Result<()> {
        if value > merge.child_birth {
            bail!(
                "invalid H2 merge: foreground birth {} exceeds death {}",
                value,
                merge.child_birth
            );
        }

        if value == merge.child_birth {
            self.diagonal_parent.insert(merge.child_id, repaired_parent);
        } else {
            self.pending_positive.push(H2BranchMerge {
                value,
                child_id: merge.child_id,
                child_birth: merge.child_birth,
                parent: repaired_parent,
            });
        }

        Ok(())
    }

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent = resolve_h2_parent(merge.parent, &self.diagonal_parent)?;
            let parent_id = match parent {
                H2Branch::Outside => OUTSIDE_BRANCH_ID,
                H2Branch::Finite { id, .. } => id,
            };

            if let Some(previous) = self.nodes.get(&merge.child_id) {
                bail!(
                    "H2 branch {} died more than once: previous birth={:?} death={:?} parent={:?}, repeated birth={} death={} parent={}",
                    merge.child_id,
                    previous.birth,
                    previous.death,
                    previous.parent,
                    merge.value,
                    merge.child_birth,
                    parent_id
                );
            }
            self.nodes.insert(
                merge.child_id,
                H2Node {
                    original_id: merge.child_id,
                    birth: Some(merge.value),
                    death: Some(merge.child_birth),
                    parent: Some(parent_id),
                    outside: false,
                },
            );
        }

        self.diagonal_parent.clear();
        Ok(())
    }

    pub(crate) fn add_outside_root(&mut self) {
        self.nodes.entry(OUTSIDE_BRANCH_ID).or_insert(H2Node {
            original_id: OUTSIDE_BRANCH_ID,
            birth: None,
            death: None,
            parent: None,
            outside: true,
        });
    }

    pub(crate) fn into_tree(mut self) -> Result<MergeTree> {
        self.add_outside_root();

        let mut nodes: Vec<H2Node> = self.nodes.into_values().collect();
        nodes.sort_by_key(|node| {
            if node.outside {
                (0u8, 0u64)
            } else {
                (1u8, node.original_id)
            }
        });

        let dense_ids: HashMap<u64, u64> = nodes
            .iter()
            .enumerate()
            .map(|(dense, node)| (node.original_id, dense as u64))
            .collect();

        let by_id: HashMap<u64, &H2Node> =
            nodes.iter().map(|node| (node.original_id, node)).collect();

        let mut node_rows = Vec::with_capacity(nodes.len());
        let mut rows = Vec::new();
        for node in &nodes {
            let parent = match node.parent {
                Some(parent_id) => Some(*by_id.get(&parent_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "H2 merge-tree parent {} for branch {} was not retained",
                        parent_id,
                        node.original_id
                    )
                })?),
                None => None,
            };

            node_rows.push(MergeTreeNodeRow {
                node: dense_ids[&node.original_id],
                parent: node.parent.map(|parent_id| dense_ids[&parent_id]),
                birth_value: display_h2_birth(node),
                death_value: display_h2_death(node),
            });

            if let Some(parent) = parent {
                rows.push(MergeTreeRow {
                    from_node: dense_ids[&node.original_id],
                    to_node: dense_ids[&parent.original_id],
                    from_birth_value: display_h2_birth(node),
                    from_death_value: display_h2_death(node),
                    to_birth_value: display_h2_birth(parent),
                    to_death_value: display_h2_death(parent),
                });
            }
        }

        rows.sort_by_key(|row| row.from_node);
        Ok(MergeTree {
            node_count: nodes.len(),
            nodes: node_rows,
            rows,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct H2PackedPositiveMerge {
    value: u16,
    child_id: u64,
    child_birth: u16,
    parent_id: u64,
}

/// Hierarchical-root H2 recorder using packed integer plateau state. Finite
/// parent IDs and the outside root are represented uniformly as `u64`, with
/// `OUTSIDE_BRANCH_ID` reserved for Outside.
#[derive(Debug, Default)]
pub(crate) struct H2PackedTreeRecorder {
    nodes: HashMap<u64, H2Node>,
    diagonal_parent: PackedU64Map,
    pending_positive: Vec<H2PackedPositiveMerge>,
}

impl H2PackedTreeRecorder {
    pub(crate) fn record_merge(&mut self, merge: H2BranchMerge) -> Result<()> {
        if merge.value > merge.child_birth {
            bail!(
                "invalid H2 merge: foreground birth {} exceeds death {}",
                merge.value,
                merge.child_birth
            );
        }
        let parent_id = match merge.parent {
            H2Branch::Outside => OUTSIDE_BRANCH_ID,
            H2Branch::Finite { id, .. } => id,
        };
        if merge.value == merge.child_birth {
            self.diagonal_parent.insert(merge.child_id, parent_id);
        } else {
            self.pending_positive.push(H2PackedPositiveMerge {
                value: merge.value,
                child_id: merge.child_id,
                child_birth: merge.child_birth,
                parent_id,
            });
        }
        Ok(())
    }

    pub(crate) fn record_compact_merge_parent_id(
        &mut self,
        value: u16,
        merge: CompactH2BranchMerge,
        repaired_parent_id: u64,
    ) -> Result<()> {
        if value > merge.child_birth {
            bail!(
                "invalid H2 merge: foreground birth {} exceeds death {}",
                value,
                merge.child_birth
            );
        }
        if value == merge.child_birth {
            self.diagonal_parent
                .insert(merge.child_id, repaired_parent_id);
        } else {
            self.pending_positive.push(H2PackedPositiveMerge {
                value,
                child_id: merge.child_id,
                child_birth: merge.child_birth,
                parent_id: repaired_parent_id,
            });
        }
        Ok(())
    }

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent_id = resolve_h2_parent_packed(merge.parent_id, &self.diagonal_parent)?;
            if let Some(previous) = self.nodes.get(&merge.child_id) {
                bail!(
                    "H2 branch {} died more than once: previous birth={:?} death={:?} parent={:?}, repeated birth={} death={} parent={}",
                    merge.child_id,
                    previous.birth,
                    previous.death,
                    previous.parent,
                    merge.value,
                    merge.child_birth,
                    parent_id
                );
            }
            self.nodes.insert(
                merge.child_id,
                H2Node {
                    original_id: merge.child_id,
                    birth: Some(merge.value),
                    death: Some(merge.child_birth),
                    parent: Some(parent_id),
                    outside: false,
                },
            );
        }
        self.diagonal_parent.clear();
        Ok(())
    }

    pub(crate) fn add_outside_root(&mut self) {
        self.nodes.entry(OUTSIDE_BRANCH_ID).or_insert(H2Node {
            original_id: OUTSIDE_BRANCH_ID,
            birth: None,
            death: None,
            parent: None,
            outside: true,
        });
    }

    pub(crate) fn into_tree(mut self) -> Result<MergeTree> {
        self.add_outside_root();
        let mut nodes: Vec<H2Node> = self.nodes.into_values().collect();
        nodes.sort_by_key(|node| {
            if node.outside {
                (0u8, 0u64)
            } else {
                (1u8, node.original_id)
            }
        });

        let dense_ids: HashMap<u64, u64> = nodes
            .iter()
            .enumerate()
            .map(|(dense, node)| (node.original_id, dense as u64))
            .collect();
        let by_id: HashMap<u64, &H2Node> =
            nodes.iter().map(|node| (node.original_id, node)).collect();

        let mut node_rows = Vec::with_capacity(nodes.len());
        let mut rows = Vec::new();
        for node in &nodes {
            let parent = match node.parent {
                Some(parent_id) => Some(*by_id.get(&parent_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "H2 merge-tree parent {} for branch {} was not retained",
                        parent_id,
                        node.original_id
                    )
                })?),
                None => None,
            };

            node_rows.push(MergeTreeNodeRow {
                node: dense_ids[&node.original_id],
                parent: node.parent.map(|parent_id| dense_ids[&parent_id]),
                birth_value: display_h2_birth(node),
                death_value: display_h2_death(node),
            });
            if let Some(parent) = parent {
                rows.push(MergeTreeRow {
                    from_node: dense_ids[&node.original_id],
                    to_node: dense_ids[&parent.original_id],
                    from_birth_value: display_h2_birth(node),
                    from_death_value: display_h2_death(node),
                    to_birth_value: display_h2_birth(parent),
                    to_death_value: display_h2_death(parent),
                });
            }
        }
        rows.sort_by_key(|row| row.from_node);
        Ok(MergeTree {
            node_count: nodes.len(),
            nodes: node_rows,
            rows,
        })
    }
}

fn resolve_h2_parent_packed(mut id: u64, diagonal_parent: &PackedU64Map) -> Result<u64> {
    let mut steps = 0usize;
    while id != OUTSIDE_BRANCH_ID {
        let Some(parent) = diagonal_parent.get(id) else {
            return Ok(id);
        };
        id = parent;
        steps += 1;
        if steps > diagonal_parent.len() {
            bail!("cycle detected while contracting packed diagonal H2 branches");
        }
    }
    Ok(id)
}

fn resolve_h2_parent(
    mut parent: H2Branch,
    diagonal_parent: &HashMap<u64, H2Branch>,
) -> Result<H2Branch> {
    let mut steps = 0usize;

    loop {
        let Some(id) = parent.finite_id() else {
            return Ok(parent);
        };
        let Some(&next) = diagonal_parent.get(&id) else {
            return Ok(parent);
        };
        parent = next;
        steps += 1;
        if steps > diagonal_parent.len() {
            bail!("cycle detected while contracting diagonal H2 branches");
        }
    }
}

fn display_death(death: Option<u16>) -> String {
    death.map_or_else(|| "inf".to_string(), |value| value.to_string())
}

fn display_h2_birth(node: &H2Node) -> String {
    if node.outside {
        "-inf".to_string()
    } else {
        node.birth
            .expect("finite H2 merge-tree node must have a birth")
            .to_string()
    }
}

fn display_h2_death(node: &H2Node) -> String {
    if node.outside {
        "inf".to_string()
    } else {
        node.death
            .expect("finite H2 merge-tree node must have a death")
            .to_string()
    }
}

fn rebuild_rows(tree: &mut MergeTree) -> Result<()> {
    let mut rows = Vec::new();
    for node in &tree.nodes {
        let Some(parent_id) = node.parent else {
            continue;
        };
        let parent_index = usize::try_from(parent_id)
            .map_err(|_| anyhow::anyhow!("branch-tree parent ID exceeds usize"))?;
        let parent = tree.nodes.get(parent_index).ok_or_else(|| {
            anyhow::anyhow!(
                "branch-tree parent {} for node {} is outside the node table",
                parent_id,
                node.node
            )
        })?;
        rows.push(MergeTreeRow {
            from_node: node.node,
            to_node: parent.node,
            from_birth_value: node.birth_value.clone(),
            from_death_value: node.death_value.clone(),
            to_birth_value: parent.birth_value.clone(),
            to_death_value: parent.death_value.clone(),
        });
    }
    rows.sort_by_key(|row| row.from_node);
    tree.rows = rows;
    Ok(())
}

/// Remove event-order ambiguity inside one H0 merge plateau.
///
/// If a positive branch and its parent die at the same threshold, both belong
/// to the same simultaneous merge plateau.  The plateau-canonical tree makes
/// every such branch point directly to the first ancestor that survives past
/// that threshold.  This is invariant to the order in which equal-valued
/// edges were reduced, which is required by the hierarchical fan-in engine.
pub(crate) fn canonicalize_h0_plateaus(mut tree: MergeTree) -> Result<MergeTree> {
    let original = tree.nodes.clone();
    for index in 0..tree.nodes.len() {
        let death = original[index].death_value.clone();
        if death == "inf" {
            continue;
        }
        let mut parent = original[index].parent;
        let mut steps = 0usize;
        while let Some(parent_id) = parent {
            let parent_index = usize::try_from(parent_id)
                .map_err(|_| anyhow::anyhow!("H0 parent ID exceeds usize"))?;
            let parent_node = original.get(parent_index).ok_or_else(|| {
                anyhow::anyhow!("H0 parent {parent_id} is outside the node table")
            })?;
            if parent_node.death_value != death {
                break;
            }
            parent = parent_node.parent;
            steps += 1;
            if steps > original.len() {
                bail!("cycle detected while canonicalizing an H0 merge plateau");
            }
        }
        tree.nodes[index].parent = parent;
    }
    rebuild_rows(&mut tree)?;
    Ok(tree)
}

/// Remove event-order ambiguity inside one H2 merge plateau.
///
/// H2 is reduced in superlevel order, so the foreground branch birth stored
/// in the exported tree is the merge threshold.  Equal birth thresholds along
/// a parent chain therefore represent one simultaneous background merge
/// plateau and are flattened to its oldest surviving ancestor.
pub(crate) fn canonicalize_h2_plateaus(mut tree: MergeTree) -> Result<MergeTree> {
    let original = tree.nodes.clone();
    for index in 0..tree.nodes.len() {
        let birth = original[index].birth_value.clone();
        if birth == "-inf" {
            continue;
        }
        let mut parent = original[index].parent;
        let mut steps = 0usize;
        while let Some(parent_id) = parent {
            let parent_index = usize::try_from(parent_id)
                .map_err(|_| anyhow::anyhow!("H2 parent ID exceeds usize"))?;
            let parent_node = original.get(parent_index).ok_or_else(|| {
                anyhow::anyhow!("H2 parent {parent_id} is outside the node table")
            })?;
            if parent_node.birth_value != birth {
                break;
            }
            parent = parent_node.parent;
            steps += 1;
            if steps > original.len() {
                bail!("cycle detected while canonicalizing an H2 merge plateau");
            }
        }
        tree.nodes[index].parent = parent;
    }
    rebuild_rows(&mut tree)?;
    Ok(tree)
}

pub fn write_merge_tree_csv(path: &Path, tree: &MergeTree) -> Result<()> {
    let mut writer = AtomicOutput::create(path)?;
    writeln!(
        writer,
        "from_node,to_node,from_birth_value,from_death_value,to_birth_value,to_death_value"
    )?;

    for row in &tree.rows {
        writeln!(
            writer,
            "{},{},{},{},{},{}",
            row.from_node,
            row.to_node,
            row.from_birth_value,
            row.from_death_value,
            row.to_birth_value,
            row.to_death_value
        )?;
    }

    writer.commit()
}

pub fn write_merge_tree_nodes_csv(path: &Path, tree: &MergeTree) -> Result<()> {
    let mut writer = AtomicOutput::create(path)?;
    writeln!(writer, "node,parent,birth_value,death_value")?;

    for node in &tree.nodes {
        let parent = node
            .parent
            .map_or_else(String::new, |parent| parent.to_string());
        writeln!(
            writer,
            "{},{},{},{}",
            node.node, parent, node.birth_value, node.death_value
        )?;
    }

    writer.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_h0_merge_round_trips_and_is_smaller() {
        let merge = H0BranchMerge {
            value: 17,
            child: H0Branch { id: 101, birth: 9 },
            parent: H0Branch { id: 55, birth: 3 },
        };
        let compact = CompactH0BranchMerge::from_merge(merge);
        let restored = compact.expand(merge.value);

        assert_eq!(restored.value, merge.value);
        assert_eq!(restored.child, merge.child);
        assert_eq!(restored.parent, merge.parent);
        assert!(std::mem::size_of::<CompactH0BranchMerge>() < std::mem::size_of::<H0BranchMerge>());
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<CompactH0BranchMerge>(), 24);
    }

    #[test]
    fn compact_h2_merge_round_trips_finite_and_outside_parents() {
        for parent in [H2Branch::Finite { id: 77, birth: 21 }, H2Branch::Outside] {
            let merge = H2BranchMerge {
                value: 12,
                child_id: 101,
                child_birth: 31,
                parent,
            };
            let compact = CompactH2BranchMerge::from_merge(merge);
            let restored = compact.expand(merge.value);

            assert_eq!(restored.value, merge.value);
            assert_eq!(restored.child_id, merge.child_id);
            assert_eq!(restored.child_birth, merge.child_birth);
            assert_eq!(restored.parent, merge.parent);
        }
        assert!(std::mem::size_of::<CompactH2BranchMerge>() < std::mem::size_of::<H2BranchMerge>());
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<CompactH2BranchMerge>(), 24);
    }

    #[test]
    fn compact_h0_recorder_matches_expanded_replay() {
        let diagonal = H0BranchMerge {
            value: 5,
            child: H0Branch { id: 20, birth: 5 },
            parent: H0Branch { id: 10, birth: 1 },
        };
        let positive = H0BranchMerge {
            value: 5,
            child: H0Branch { id: 30, birth: 2 },
            parent: diagonal.child,
        };

        let mut full = H0TreeRecorder::default();
        full.record_merge(diagonal).unwrap();
        full.record_merge(positive).unwrap();
        full.finish_threshold().unwrap();
        full.add_essential(diagonal.parent).unwrap();
        let full = full.into_tree().unwrap();

        let mut compact = H0TreeRecorder::default();
        compact
            .record_compact_merge(
                diagonal.value,
                CompactH0BranchMerge::from_merge(diagonal),
                diagonal.parent,
            )
            .unwrap();
        compact
            .record_compact_merge(
                positive.value,
                CompactH0BranchMerge::from_merge(positive),
                positive.parent,
            )
            .unwrap();
        compact.finish_threshold().unwrap();
        compact.add_essential(diagonal.parent).unwrap();
        let compact = compact.into_tree().unwrap();

        assert_eq!(compact.node_count, full.node_count);
        assert_eq!(compact.nodes.len(), full.nodes.len());
        assert_eq!(compact.rows.len(), full.rows.len());
        for (a, b) in compact.nodes.iter().zip(full.nodes.iter()) {
            assert_eq!(a.node, b.node);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.birth_value, b.birth_value);
            assert_eq!(a.death_value, b.death_value);
        }
    }

    #[test]
    fn compact_h2_recorder_matches_expanded_replay() {
        let diagonal = H2BranchMerge {
            value: 5,
            child_id: 20,
            child_birth: 5,
            parent: H2Branch::Outside,
        };
        let positive = H2BranchMerge {
            value: 5,
            child_id: 30,
            child_birth: 9,
            parent: H2Branch::Finite { id: 20, birth: 5 },
        };

        let mut full = H2TreeRecorder::default();
        full.record_merge(diagonal).unwrap();
        full.record_merge(positive).unwrap();
        full.finish_threshold().unwrap();
        full.add_outside_root();
        let full = full.into_tree().unwrap();

        let mut compact = H2TreeRecorder::default();
        compact
            .record_compact_merge(
                diagonal.value,
                CompactH2BranchMerge::from_merge(diagonal),
                diagonal.parent,
            )
            .unwrap();
        compact
            .record_compact_merge(
                positive.value,
                CompactH2BranchMerge::from_merge(positive),
                positive.parent,
            )
            .unwrap();
        compact.finish_threshold().unwrap();
        compact.add_outside_root();
        let compact = compact.into_tree().unwrap();

        assert_eq!(compact.node_count, full.node_count);
        assert_eq!(compact.nodes.len(), full.nodes.len());
        assert_eq!(compact.rows.len(), full.rows.len());
        for (a, b) in compact.nodes.iter().zip(full.nodes.iter()) {
            assert_eq!(a.node, b.node);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.birth_value, b.birth_value);
            assert_eq!(a.death_value, b.death_value);
        }
    }

    #[test]
    fn packed_u64_map_grows_updates_and_clears() {
        let mut map = PackedU64Map::default();
        for id in 0..2_000u64 {
            map.insert(id, id + 10);
        }
        assert_eq!(map.len(), 2_000);
        for id in 0..2_000u64 {
            assert_eq!(map.get(id), Some(id + 10));
        }
        map.insert(17, 999_999);
        assert_eq!(map.len(), 2_000);
        assert_eq!(map.get(17), Some(999_999));

        map.clear();
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(17), None);
        map.insert(42, 7);
        assert_eq!(map.get(42), Some(7));
    }

    #[test]
    fn packed_deferred_resolver_matches_watched_redirect_semantics() {
        let mut resolver = PackedDeferredIdResolver::default();
        for id in 0..2_000u64 {
            resolver.watch(id * 3 + 1);
        }
        resolver.watch(20);
        resolver.observe(20, 30);
        resolver.observe(30, 40);
        assert_eq!(resolver.resolve_id(20).unwrap(), 40);
        assert!(resolver.len() >= 2_002);

        // Outside is a terminal redirect value, not a finite watched key.
        resolver.observe(40, OUTSIDE_BRANCH_ID);
        assert_eq!(resolver.resolve_id(20).unwrap(), OUTSIDE_BRANCH_ID);
    }

    #[test]
    fn packed_h0_root_recorder_matches_reference_recorder() {
        let diagonal = H0BranchMerge {
            value: 5,
            child: H0Branch { id: 20, birth: 5 },
            parent: H0Branch { id: 10, birth: 1 },
        };
        let positive = H0BranchMerge {
            value: 5,
            child: H0Branch { id: 30, birth: 2 },
            parent: diagonal.child,
        };

        let mut full = H0TreeRecorder::default();
        full.record_merge(diagonal).unwrap();
        full.record_merge(positive).unwrap();
        full.finish_threshold().unwrap();
        full.add_essential(diagonal.parent).unwrap();
        let full = full.into_tree().unwrap();

        let mut packed = H0PackedTreeRecorder::default();
        packed
            .record_compact_merge_parent_id(
                diagonal.value,
                CompactH0BranchMerge::from_merge(diagonal),
                diagonal.parent.id,
            )
            .unwrap();
        packed
            .record_compact_merge_parent_id(
                positive.value,
                CompactH0BranchMerge::from_merge(positive),
                positive.parent.id,
            )
            .unwrap();
        packed.finish_threshold().unwrap();
        packed.add_essential(diagonal.parent).unwrap();
        let packed = packed.into_tree().unwrap();

        assert_eq!(packed.node_count, full.node_count);
        assert_eq!(packed.nodes.len(), full.nodes.len());
        assert_eq!(packed.rows.len(), full.rows.len());
        for (a, b) in packed.nodes.iter().zip(full.nodes.iter()) {
            assert_eq!(a.node, b.node);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.birth_value, b.birth_value);
            assert_eq!(a.death_value, b.death_value);
        }
    }

    #[test]
    fn packed_h2_root_recorder_matches_reference_recorder() {
        let diagonal = H2BranchMerge {
            value: 5,
            child_id: 20,
            child_birth: 5,
            parent: H2Branch::Outside,
        };
        let positive = H2BranchMerge {
            value: 5,
            child_id: 30,
            child_birth: 9,
            parent: H2Branch::Finite { id: 20, birth: 5 },
        };

        let mut full = H2TreeRecorder::default();
        full.record_merge(diagonal).unwrap();
        full.record_merge(positive).unwrap();
        full.finish_threshold().unwrap();
        full.add_outside_root();
        let full = full.into_tree().unwrap();

        let mut packed = H2PackedTreeRecorder::default();
        packed
            .record_compact_merge_parent_id(
                diagonal.value,
                CompactH2BranchMerge::from_merge(diagonal),
                OUTSIDE_BRANCH_ID,
            )
            .unwrap();
        packed
            .record_compact_merge_parent_id(
                positive.value,
                CompactH2BranchMerge::from_merge(positive),
                20,
            )
            .unwrap();
        packed.finish_threshold().unwrap();
        packed.add_outside_root();
        let packed = packed.into_tree().unwrap();

        assert_eq!(packed.node_count, full.node_count);
        assert_eq!(packed.nodes.len(), full.nodes.len());
        assert_eq!(packed.rows.len(), full.rows.len());
        for (a, b) in packed.nodes.iter().zip(full.nodes.iter()) {
            assert_eq!(a.node, b.node);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.birth_value, b.birth_value);
            assert_eq!(a.death_value, b.death_value);
        }
    }

    #[test]
    fn h0_root_is_retained_in_the_node_table_without_an_edge() {
        let mut recorder = H0TreeRecorder::default();
        recorder
            .add_essential(H0Branch { id: 7, birth: 3 })
            .unwrap();
        let tree = recorder.into_tree().unwrap();

        assert_eq!(tree.node_count, 1);
        assert!(tree.rows.is_empty());
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].parent, None);
        assert_eq!(tree.nodes[0].birth_value, "3");
        assert_eq!(tree.nodes[0].death_value, "inf");
    }

    #[test]
    fn h0_plateau_canonicalization_flattens_equal_death_chain() {
        let tree = MergeTree {
            node_count: 3,
            nodes: vec![
                MergeTreeNodeRow {
                    node: 0,
                    parent: Some(1),
                    birth_value: "2".into(),
                    death_value: "5".into(),
                },
                MergeTreeNodeRow {
                    node: 1,
                    parent: Some(2),
                    birth_value: "1".into(),
                    death_value: "5".into(),
                },
                MergeTreeNodeRow {
                    node: 2,
                    parent: None,
                    birth_value: "0".into(),
                    death_value: "inf".into(),
                },
            ],
            rows: Vec::new(),
        };
        let tree = canonicalize_h0_plateaus(tree).unwrap();
        assert_eq!(tree.nodes[0].parent, Some(2));
        assert_eq!(tree.nodes[1].parent, Some(2));
        assert_eq!(tree.rows.len(), 2);
    }

    #[test]
    fn h2_plateau_canonicalization_flattens_equal_birth_chain() {
        let tree = MergeTree {
            node_count: 3,
            nodes: vec![
                MergeTreeNodeRow {
                    node: 0,
                    parent: None,
                    birth_value: "-inf".into(),
                    death_value: "inf".into(),
                },
                MergeTreeNodeRow {
                    node: 1,
                    parent: Some(0),
                    birth_value: "5".into(),
                    death_value: "9".into(),
                },
                MergeTreeNodeRow {
                    node: 2,
                    parent: Some(1),
                    birth_value: "5".into(),
                    death_value: "8".into(),
                },
            ],
            rows: Vec::new(),
        };
        let tree = canonicalize_h2_plateaus(tree).unwrap();
        assert_eq!(tree.nodes[2].parent, Some(0));
        assert_eq!(tree.rows.len(), 2);
    }

    #[test]
    fn h2_outside_root_is_retained_in_the_node_table() {
        let tree = H2TreeRecorder::default().into_tree().unwrap();

        assert_eq!(tree.node_count, 1);
        assert!(tree.rows.is_empty());
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].parent, None);
        assert_eq!(tree.nodes[0].birth_value, "-inf");
        assert_eq!(tree.nodes[0].death_value, "inf");
    }

    #[test]
    fn same_level_three_component_merge_records_the_documented_branch_chain() {
        // Three components born at 0, 1, and 2 merge at q=5. Processing the
        // 1--2 merge before the 0--1 merge creates the elder-rule branch chain
        // 2 -> 1 -> 0. The positive barcode is independent of this hierarchy.
        let oldest = H0Branch { id: 10, birth: 0 };
        let middle = H0Branch { id: 11, birth: 1 };
        let youngest = H0Branch { id: 12, birth: 2 };
        let mut recorder = H0TreeRecorder::default();
        recorder
            .record_merge(H0BranchMerge {
                child: youngest,
                parent: middle,
                value: 5,
            })
            .unwrap();
        recorder
            .record_merge(H0BranchMerge {
                child: middle,
                parent: oldest,
                value: 5,
            })
            .unwrap();
        recorder.finish_threshold().unwrap();
        recorder.add_essential(oldest).unwrap();

        let tree = recorder.into_tree().unwrap();
        let parents: HashMap<(String, String), (String, String)> = tree
            .rows
            .iter()
            .map(|row| {
                (
                    (row.from_birth_value.clone(), row.from_death_value.clone()),
                    (row.to_birth_value.clone(), row.to_death_value.clone()),
                )
            })
            .collect();
        assert_eq!(
            parents[&(String::from("2"), String::from("5"))],
            (String::from("1"), String::from("5"))
        );
        assert_eq!(
            parents[&(String::from("1"), String::from("5"))],
            (String::from("0"), String::from("inf"))
        );
    }
}
