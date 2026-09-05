use crate::connectivity::Connectivity;

const MAX_NEIGHBORS: usize = 26;
const DEFAULT_CACHE_SIZE: usize = 1 << 16;
const EMPTY_KEY: u32 = u32::MAX;

const OFFSETS_6: [(isize, isize, isize); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

const OFFSETS_26: [(isize, isize, isize); 26] = [
    (-1, -1, -1),
    (0, -1, -1),
    (1, -1, -1),
    (-1, 0, -1),
    (0, 0, -1),
    (1, 0, -1),
    (-1, 1, -1),
    (0, 1, -1),
    (1, 1, -1),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (-1, -1, 1),
    (0, -1, 1),
    (1, -1, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
];

#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearNeighborOffsets {
    offsets: [isize; MAX_NEIGHBORS],
    len: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NeighborhoodPruningStats {
    pub(crate) mask_calls: u64,
    pub(crate) active_state_checks: u64,
    pub(crate) active_neighbor_hits: u64,
    pub(crate) representative_visits: u64,
    pub(crate) cache_lookups: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) component_mask_computations: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RepresentativeNeighbors {
    indices: [usize; MAX_NEIGHBORS],
    len: usize,
}

impl RepresentativeNeighbors {
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices[..self.len].iter().copied()
    }
}

/// Exact graph-neighborhood pruning for event-based voxel activation.
///
/// For the newly activated voxel, the active neighboring voxels are split into
/// connected components inside the active induced neighborhood. The caller only
/// needs to union the new voxel with one representative from each such component.
/// Any omitted edge is already redundant because its endpoint is connected to the
/// retained representative through active neighbors without using the new voxel.
///
/// This requires the caller to activate and process one voxel at a time: all edges
/// among previously active voxels must already have been processed. Pre-activating
/// an entire tie group violates that invariant and can incorrectly prune an edge.
///
/// For 26-connectivity, the representative mask is cached in a bounded
/// direct-mapped cache keyed by the 26-bit active-neighbor mask. A miss is
/// computed at run time. Cache collisions only cause recomputation; they never
/// affect correctness. This is a graph-connectivity cache, not a precomputed
/// cubical lower-star cell-pair table. For 6-connectivity, no two face neighbors
/// of the center are 6-adjacent, so the active mask is already minimal.
#[derive(Debug)]
pub(crate) struct NeighborhoodComponentPruner {
    connectivity: Connectivity,
    adjacency_masks: [u32; MAX_NEIGHBORS],
    /// Six-connectivity never uses the 26-neighbor pattern cache. Keeping the
    /// two 65,536-entry arrays optional avoids about 512 KiB per active slab
    /// worker in that common case.
    cache: Option<PruningCache>,
    diagnostics: Option<NeighborhoodPruningStats>,
}

#[derive(Debug)]
struct PruningCache {
    keys: Vec<u32>,
    values: Vec<u32>,
    slot_mask: usize,
}

impl NeighborhoodComponentPruner {
    pub(crate) fn new(connectivity: Connectivity) -> Self {
        Self::with_diagnostics_and_cache_entries(connectivity, false, Some(DEFAULT_CACHE_SIZE))
    }

    pub(crate) fn with_diagnostics(connectivity: Connectivity, diagnostics: bool) -> Self {
        Self::with_diagnostics_and_cache_entries(
            connectivity,
            diagnostics,
            Some(DEFAULT_CACHE_SIZE),
        )
    }

    pub(crate) fn with_diagnostics_and_cache_entries(
        connectivity: Connectivity,
        diagnostics: bool,
        cache_entries: Option<usize>,
    ) -> Self {
        let adjacency_masks = build_adjacency_masks(connectivity);
        if let Some(entries) = cache_entries {
            assert!(
                entries.is_power_of_two(),
                "pruning cache size must be a power of two"
            );
            assert!(entries > 0, "pruning cache size must be nonzero");
        }

        Self {
            connectivity,
            adjacency_masks,
            cache: if matches!(connectivity, Connectivity::TwentySix) {
                cache_entries.map(|entries| PruningCache {
                    keys: vec![EMPTY_KEY; entries],
                    values: vec![0u32; entries],
                    slot_mask: entries - 1,
                })
            } else {
                None
            },
            diagnostics: diagnostics.then(NeighborhoodPruningStats::default),
        }
    }

    pub(crate) fn diagnostic_stats(&self) -> NeighborhoodPruningStats {
        self.diagnostics.unwrap_or_default()
    }

    pub(crate) fn cache_capacity_bytes(&self) -> u64 {
        self.cache
            .as_ref()
            .map(|cache| {
                (cache.keys.capacity() + cache.values.capacity()) as u64
                    * core::mem::size_of::<u32>() as u64
            })
            .unwrap_or(0)
    }

    fn record_masks(&mut self, active_mask: u32, representative_mask: u32) {
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.mask_calls += 1;
            stats.active_neighbor_hits += u64::from(active_mask.count_ones());
            stats.representative_visits += u64::from(representative_mask.count_ones());
        }
    }

    /// Returns one already-active neighbor from each locally connected component of
    /// the active neighborhood around `(x, y, z)`.
    ///
    /// `is_active` is deliberately generic so callers can benchmark either a
    /// dedicated byte array or union-find parent sentinel state without unsafe
    /// aliasing between the pruning scan and subsequent union operations.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn representative_neighbors_by<F>(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
        mut is_active: F,
    ) -> RepresentativeNeighbors
    where
        F: FnMut(usize) -> bool,
    {
        let slice_size = width
            .checked_mul(height)
            .expect("slice size overflow in local pruning");
        let mut neighbor_indices = [0usize; MAX_NEIGHBORS];
        let (active_mask, active_state_checks) = self.build_active_mask_by(
            x,
            y,
            z,
            width,
            height,
            depth,
            slice_size,
            &mut neighbor_indices,
            &mut is_active,
        );

        let representative_mask = self.representative_mask(active_mask);
        self.record_masks(active_mask, representative_mask);
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.active_state_checks += active_state_checks;
        }
        representative_list(neighbor_indices, representative_mask)
    }

    /// Fast variant for a center voxel strictly inside all six local slab faces.
    pub(crate) fn representative_neighbors_interior_by<F>(
        &mut self,
        center: usize,
        linear_offsets: &LinearNeighborOffsets,
        mut is_active: F,
    ) -> RepresentativeNeighbors
    where
        F: FnMut(usize) -> bool,
    {
        debug_assert_eq!(linear_offsets.len, self.offsets().len());
        let mut neighbor_indices = [0usize; MAX_NEIGHBORS];
        let mut active_mask = 0u32;

        for (bit, neighbor_slot) in neighbor_indices
            .iter_mut()
            .enumerate()
            .take(linear_offsets.len)
        {
            let neighbor = center.wrapping_add_signed(linear_offsets.offsets[bit]);
            if is_active(neighbor) {
                *neighbor_slot = neighbor;
                active_mask |= 1u32 << bit;
            }
        }

        let representative_mask = self.representative_mask(active_mask);
        self.record_masks(active_mask, representative_mask);
        if let Some(stats) = self.diagnostics.as_mut() {
            stats.active_state_checks += linear_offsets.len as u64;
        }
        representative_list(neighbor_indices, representative_mask)
    }

    /// Visits one already-active neighbor from each locally connected component of
    /// the active neighborhood around `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_each_representative_neighbor<F>(
        &mut self,
        active: &[u8],
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
        mut visit: F,
    ) where
        F: FnMut(usize),
    {
        let representatives =
            self.representative_neighbors_by(x, y, z, width, height, depth, |neighbor| {
                active[neighbor] != 0
            });
        for neighbor in representatives.iter() {
            visit(neighbor);
        }
    }

    /// Precompute linear-index deltas for the current slab geometry.
    ///
    /// These deltas are valid only for voxels strictly inside all six local
    /// slab faces. Boundary voxels must use `for_each_representative_neighbor`.
    pub(crate) fn linear_offsets(&self, width: usize, height: usize) -> LinearNeighborOffsets {
        let slice_size = width
            .checked_mul(height)
            .expect("slice size overflow in local pruning");
        let width = isize::try_from(width).expect("width exceeds isize in local pruning");
        let slice_size =
            isize::try_from(slice_size).expect("slice size exceeds isize in local pruning");
        let mut linear = [0isize; MAX_NEIGHBORS];
        let offsets = self.offsets();
        for (bit, &(dx, dy, dz)) in offsets.iter().enumerate() {
            linear[bit] = dz * slice_size + dy * width + dx;
        }
        LinearNeighborOffsets {
            offsets: linear,
            len: offsets.len(),
        }
    }

    /// Fast path for a center voxel strictly inside the local slab.
    #[cfg(test)]
    pub(crate) fn for_each_representative_neighbor_interior<F>(
        &mut self,
        active: &[u8],
        center: usize,
        linear_offsets: &LinearNeighborOffsets,
        mut visit: F,
    ) where
        F: FnMut(usize),
    {
        let representatives =
            self.representative_neighbors_interior_by(center, linear_offsets, |neighbor| {
                debug_assert!(neighbor < active.len());
                active[neighbor] != 0
            });
        for neighbor in representatives.iter() {
            visit(neighbor);
        }
    }

    fn offsets(&self) -> &'static [(isize, isize, isize)] {
        match self.connectivity {
            Connectivity::Six => &OFFSETS_6,
            Connectivity::TwentySix => &OFFSETS_26,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_active_mask_by<F>(
        &self,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
        slice_size: usize,
        neighbor_indices: &mut [usize; MAX_NEIGHBORS],
        is_active: &mut F,
    ) -> (u32, u64)
    where
        F: FnMut(usize) -> bool,
    {
        let mut mask = 0u32;
        let mut checks = 0u64;

        for (bit, &(dx, dy, dz)) in self.offsets().iter().enumerate() {
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

            let neighbor = nz as usize * slice_size + ny as usize * width + nx as usize;
            checks += 1;
            if is_active(neighbor) {
                neighbor_indices[bit] = neighbor;
                mask |= 1u32 << bit;
            }
        }

        (mask, checks)
    }

    fn representative_mask(&mut self, active_mask: u32) -> u32 {
        if active_mask == 0 || matches!(self.connectivity, Connectivity::Six) {
            return active_mask;
        }

        if self.cache.is_some() {
            if let Some(stats) = self.diagnostics.as_mut() {
                stats.cache_lookups += 1;
            }
            let (slot, cached_value) = {
                let cache = self
                    .cache
                    .as_ref()
                    .expect("cache disappeared during lookup");
                let slot = cache_slot(active_mask, cache.slot_mask);
                let cached_value = (cache.keys[slot] == active_mask).then_some(cache.values[slot]);
                (slot, cached_value)
            };
            if let Some(value) = cached_value {
                if let Some(stats) = self.diagnostics.as_mut() {
                    stats.cache_hits += 1;
                }
                return value;
            }

            if let Some(stats) = self.diagnostics.as_mut() {
                stats.cache_misses += 1;
            }
            let representatives = self.compute_representative_mask(active_mask);
            if let Some(stats) = self.diagnostics.as_mut() {
                stats.component_mask_computations += 1;
            }
            let cache = self
                .cache
                .as_mut()
                .expect("cache disappeared during lookup");
            cache.keys[slot] = active_mask;
            cache.values[slot] = representatives;
            return representatives;
        }

        if let Some(stats) = self.diagnostics.as_mut() {
            stats.component_mask_computations += 1;
        }
        self.compute_representative_mask(active_mask)
    }

    fn compute_representative_mask(&self, active_mask: u32) -> u32 {
        let mut remaining = active_mask;
        let mut representatives = 0u32;

        while remaining != 0 {
            let seed = remaining.trailing_zeros() as usize;
            let seed_bit = 1u32 << seed;
            representatives |= seed_bit;

            let mut component = 0u32;
            let mut frontier = seed_bit;

            while frontier != 0 {
                let bit = frontier.trailing_zeros() as usize;
                let bit_mask = 1u32 << bit;
                frontier &= frontier - 1;

                if component & bit_mask != 0 {
                    continue;
                }

                component |= bit_mask;
                frontier |= self.adjacency_masks[bit] & active_mask & !component;
            }

            remaining &= !component;
        }

        representatives
    }
}

fn representative_list(
    neighbor_indices: [usize; MAX_NEIGHBORS],
    mut representative_mask: u32,
) -> RepresentativeNeighbors {
    let mut indices = [0usize; MAX_NEIGHBORS];
    let mut len = 0usize;
    while representative_mask != 0 {
        let bit = representative_mask.trailing_zeros() as usize;
        representative_mask &= representative_mask - 1;
        indices[len] = neighbor_indices[bit];
        len += 1;
    }
    RepresentativeNeighbors { indices, len }
}

fn cache_slot(mask: u32, slot_mask: usize) -> usize {
    (mask.wrapping_mul(0x9E37_79B1) as usize) & slot_mask
}

fn build_adjacency_masks(connectivity: Connectivity) -> [u32; MAX_NEIGHBORS] {
    let offsets: &[(isize, isize, isize)] = match connectivity {
        Connectivity::Six => &OFFSETS_6,
        Connectivity::TwentySix => &OFFSETS_26,
    };

    let mut masks = [0u32; MAX_NEIGHBORS];

    for (i, &a) in offsets.iter().enumerate() {
        for (j, &b) in offsets.iter().enumerate() {
            if i != j && offsets_are_adjacent(a, b, connectivity) {
                masks[i] |= 1u32 << j;
            }
        }
    }

    masks
}

fn offsets_are_adjacent(
    a: (isize, isize, isize),
    b: (isize, isize, isize),
    connectivity: Connectivity,
) -> bool {
    let dx = (a.0 - b.0).abs();
    let dy = (a.1 - b.1).abs();
    let dz = (a.2 - b.2).abs();

    match connectivity {
        Connectivity::Six => dx + dy + dz == 1,
        Connectivity::TwentySix => dx.max(dy).max(dz) == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_connectivity_does_not_prune_face_neighbors() {
        let mut pruner = NeighborhoodComponentPruner::new(Connectivity::Six);
        let mask = 0b11_1111u32;
        assert_eq!(pruner.representative_mask(mask), mask);
    }

    #[test]
    fn twenty_six_connectivity_prunes_locally_connected_neighbors() {
        let mut pruner = NeighborhoodComponentPruner::new(Connectivity::TwentySix);

        // OFFSETS_26[0] = (-1,-1,-1) and OFFSETS_26[1] = (0,-1,-1)
        // are 26-adjacent, so one representative is sufficient.
        let mask = (1u32 << 0) | (1u32 << 1);
        assert_eq!(pruner.representative_mask(mask).count_ones(), 1);
    }

    #[test]
    fn twenty_six_connectivity_keeps_disconnected_neighbors() {
        let mut pruner = NeighborhoodComponentPruner::new(Connectivity::TwentySix);

        // Opposite corners of the 3x3x3 neighborhood are not adjacent and have
        // no active intermediate neighbor in this mask.
        let mask = (1u32 << 0) | (1u32 << 25);
        assert_eq!(pruner.representative_mask(mask).count_ones(), 2);
    }

    #[test]
    fn interior_fast_path_matches_generic_path() {
        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let width = 5usize;
            let height = 5usize;
            let depth = 5usize;
            let center = 2 * width * height + 2 * width + 2;
            let mut active = vec![0u8; width * height * depth];

            // Deterministic nontrivial neighborhood pattern around the center.
            for (index, flag) in active.iter_mut().enumerate() {
                if index != center && (index.wrapping_mul(17).wrapping_add(11)) % 7 < 3 {
                    *flag = 1;
                }
            }

            let mut generic = NeighborhoodComponentPruner::new(connectivity);
            let mut generic_neighbors = Vec::new();
            generic.for_each_representative_neighbor(
                &active,
                2,
                2,
                2,
                width,
                height,
                depth,
                |neighbor| generic_neighbors.push(neighbor),
            );

            let mut fast = NeighborhoodComponentPruner::new(connectivity);
            let linear_offsets = fast.linear_offsets(width, height);
            let mut fast_neighbors = Vec::new();
            fast.for_each_representative_neighbor_interior(
                &active,
                center,
                &linear_offsets,
                |neighbor| fast_neighbors.push(neighbor),
            );

            assert_eq!(fast_neighbors, generic_neighbors);
        }
    }

    #[test]
    fn interior_fast_path_matches_generic_across_many_masks() {
        let width = 5usize;
        let height = 5usize;
        let depth = 5usize;
        let center_x = 2usize;
        let center_y = 2usize;
        let center_z = 2usize;
        let center = center_z * width * height + center_y * width + center_x;

        for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
            let offsets: &[(isize, isize, isize)] = match connectivity {
                Connectivity::Six => &OFFSETS_6,
                Connectivity::TwentySix => &OFFSETS_26,
            };
            let mask_count = if matches!(connectivity, Connectivity::Six) {
                64u32
            } else {
                1024u32
            };

            for i in 0..mask_count {
                let mask = if matches!(connectivity, Connectivity::Six) {
                    i
                } else {
                    // Deterministic spread without trying to enumerate 2^26.
                    i.wrapping_mul(0x9E37_79B1) & ((1u32 << 26) - 1)
                };
                let mut active = vec![0u8; width * height * depth];
                for (bit, &(dx, dy, dz)) in offsets.iter().enumerate() {
                    if mask & (1u32 << bit) == 0 {
                        continue;
                    }
                    let x = (center_x as isize + dx) as usize;
                    let y = (center_y as isize + dy) as usize;
                    let z = (center_z as isize + dz) as usize;
                    active[z * width * height + y * width + x] = 1;
                }

                let mut generic = NeighborhoodComponentPruner::new(connectivity);
                let mut generic_neighbors = Vec::new();
                generic.for_each_representative_neighbor(
                    &active,
                    center_x,
                    center_y,
                    center_z,
                    width,
                    height,
                    depth,
                    |neighbor| generic_neighbors.push(neighbor),
                );

                let mut fast = NeighborhoodComponentPruner::new(connectivity);
                let linear_offsets = fast.linear_offsets(width, height);
                let mut fast_neighbors = Vec::new();
                fast.for_each_representative_neighbor_interior(
                    &active,
                    center,
                    &linear_offsets,
                    |neighbor| fast_neighbors.push(neighbor),
                );

                assert_eq!(fast_neighbors, generic_neighbors, "mask={mask:#x}");
            }
        }
    }

    #[test]
    fn pruning_cache_sizes_and_off_preserve_representative_masks() {
        let strategies = [
            None,
            Some(1usize << 12),
            Some(1usize << 14),
            Some(1usize << 16),
            Some(1usize << 18),
        ];
        let mut pruners = strategies
            .iter()
            .copied()
            .map(|entries| {
                NeighborhoodComponentPruner::with_diagnostics_and_cache_entries(
                    Connectivity::TwentySix,
                    true,
                    entries,
                )
            })
            .collect::<Vec<_>>();

        for i in 0..4096u32 {
            let mask = i.wrapping_mul(0x9E37_79B1) & ((1u32 << 26) - 1);
            let expected = pruners[0].representative_mask(mask);
            for (entries, pruner) in strategies.iter().zip(pruners.iter_mut()).skip(1) {
                assert_eq!(
                    pruner.representative_mask(mask),
                    expected,
                    "mask={mask:#x} entries={entries:?}"
                );
            }
        }
    }

    #[test]
    fn disabled_pruning_cache_records_no_lookups_or_misses() {
        let mut pruner = NeighborhoodComponentPruner::with_diagnostics_and_cache_entries(
            Connectivity::TwentySix,
            true,
            None,
        );
        let mask = (1u32 << 0) | (1u32 << 1) | (1u32 << 25);
        let _ = pruner.representative_mask(mask);
        let stats = pruner.diagnostic_stats();
        assert_eq!(stats.cache_lookups, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.component_mask_computations, 1);
    }
}
