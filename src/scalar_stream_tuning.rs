use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeStrategy {
    Scan,
    Heap,
    Auto,
}

impl MergeStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "scan" => Ok(Self::Scan),
            "heap" => Ok(Self::Heap),
            "auto" => Ok(Self::Auto),
            _ => bail!("invalid --merge-strategy {value:?}; expected scan, heap, or auto"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Heap => "heap",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterfaceOrderStrategy {
    Comparison,
    Radix,
}

impl InterfaceOrderStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "comparison" => Ok(Self::Comparison),
            "radix" => Ok(Self::Radix),
            _ => bail!("invalid --interface-order {value:?}; expected comparison or radix"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Comparison => "comparison",
            Self::Radix => "radix",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EventOrderStrategy {
    Resort,
    Verify,
}

impl EventOrderStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "resort" => Ok(Self::Resort),
            "verify" => Ok(Self::Verify),
            _ => bail!("invalid --event-order {value:?}; expected resort or verify"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Resort => "resort",
            Self::Verify => "verify",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F32KeyMode {
    Legacy64,
    Native32,
}

impl F32KeyMode {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "legacy64" => Ok(Self::Legacy64),
            "native32" => Ok(Self::Native32),
            _ => bail!("invalid --f32-key-mode {value:?}; expected legacy64 or native32"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy64 => "legacy64",
            Self::Native32 => "native32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NeighborKernelStrategy {
    Generic,
    InteriorFast,
}

impl NeighborKernelStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "generic" => Ok(Self::Generic),
            "interior-fast" => Ok(Self::InteriorFast),
            _ => bail!("invalid --neighbor-kernel {value:?}; expected generic or interior-fast"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::InteriorFast => "interior-fast",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RepresentativeActiveCheckStrategy {
    Recheck,
    TrustPruner,
}

impl RepresentativeActiveCheckStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "recheck" => Ok(Self::Recheck),
            "trust-pruner" => Ok(Self::TrustPruner),
            _ => bail!(
                "invalid --representative-active-check {value:?}; expected recheck or trust-pruner"
            ),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Recheck => "recheck",
            Self::TrustPruner => "trust-pruner",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnionKernelStrategy {
    Conventional,
    RootCarrying,
}

impl UnionKernelStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "conventional" => Ok(Self::Conventional),
            "root-carrying" => Ok(Self::RootCarrying),
            _ => bail!("invalid --union-kernel {value:?}; expected conventional or root-carrying"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Conventional => "conventional",
            Self::RootCarrying => "root-carrying",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NeighborRootCheckStrategy {
    Find,
    ParentShortcut,
}

impl NeighborRootCheckStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "find" => Ok(Self::Find),
            "parent-shortcut" => Ok(Self::ParentShortcut),
            _ => bail!("invalid --neighbor-root-check {value:?}; expected find or parent-shortcut"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Find => "find",
            Self::ParentShortcut => "parent-shortcut",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H0PruningCacheStrategy {
    Off,
    Entries4K,
    Entries16K,
    Entries64K,
    Entries256K,
}

impl H0PruningCacheStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "4k" => Ok(Self::Entries4K),
            "16k" => Ok(Self::Entries16K),
            "64k" => Ok(Self::Entries64K),
            "256k" => Ok(Self::Entries256K),
            _ => bail!("invalid --h0-pruning-cache {value:?}; expected off, 4k, 16k, 64k, or 256k"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Entries4K => "4k",
            Self::Entries16K => "16k",
            Self::Entries64K => "64k",
            Self::Entries256K => "256k",
        }
    }

    pub(crate) const fn entries(self) -> Option<usize> {
        match self {
            Self::Off => None,
            Self::Entries4K => Some(1 << 12),
            Self::Entries16K => Some(1 << 14),
            Self::Entries64K => Some(1 << 16),
            Self::Entries256K => Some(1 << 18),
        }
    }

    pub(crate) const fn storage_bytes(self) -> usize {
        match self.entries() {
            Some(entries) => entries * 2 * core::mem::size_of::<u32>(),
            None => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveStateStrategy {
    Separate,
    ParentSentinel,
}

impl ActiveStateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "separate" => Ok(Self::Separate),
            "parent-sentinel" => Ok(Self::ParentSentinel),
            _ => bail!("invalid --active-state {value:?}; expected separate or parent-sentinel"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::ParentSentinel => "parent-sentinel",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterfaceStateStrategy {
    Vector,
    RootInvariant,
}

impl InterfaceStateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "vector" => Ok(Self::Vector),
            "root-invariant" => Ok(Self::RootInvariant),
            _ => bail!("invalid --interface-state {value:?}; expected vector or root-invariant"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::RootInvariant => "root-invariant",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnionFindLayoutStrategy {
    ParentRank,
    Packed,
}

impl UnionFindLayoutStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "parent-rank" => Ok(Self::ParentRank),
            "packed" => Ok(Self::Packed),
            _ => bail!("invalid --uf-layout {value:?}; expected parent-rank or packed"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ParentRank => "parent-rank",
            Self::Packed => "packed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhaseTrimStrategy {
    Off,
    BeforeReduce,
}

impl PhaseTrimStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "before-reduce" => Ok(Self::BeforeReduce),
            _ => bail!("invalid --phase-trim {value:?}; expected off or before-reduce"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::BeforeReduce => "before-reduce",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalH2BirthStateStrategy {
    Tagged,
    Compact,
}

impl LocalH2BirthStateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "tagged" => Ok(Self::Tagged),
            "compact" => Ok(Self::Compact),
            _ => bail!("invalid --local-h2-birth-state {value:?}; expected tagged or compact"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Tagged => "tagged",
            Self::Compact => "compact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H0BirthBufferStrategy {
    Copy,
    ReuseInput,
}

impl H0BirthBufferStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "copy" => Ok(Self::Copy),
            "reuse-input" => Ok(Self::ReuseInput),
            _ => bail!("invalid --h0-birth-buffer {value:?}; expected copy or reuse-input"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::ReuseInput => "reuse-input",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H0EventStorageStrategy {
    Buffered,
    Direct,
}

impl H0EventStorageStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "buffered" => Ok(Self::Buffered),
            "direct" => Ok(Self::Direct),
            _ => bail!("invalid --h0-event-storage {value:?}; expected buffered or direct"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Buffered => "buffered",
            Self::Direct => "direct",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H0HierAttachPruningStrategy {
    Off,
    ElderDominated,
}

impl H0HierAttachPruningStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "elder-dominated" => Ok(Self::ElderDominated),
            _ => {
                bail!("invalid --h0-hier-attach-pruning {value:?}; expected off or elder-dominated")
            }
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ElderDominated => "elder-dominated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H2HierCrossStorageStrategy {
    Disk,
    Direct,
}

impl H2HierCrossStorageStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "disk" => Ok(Self::Disk),
            "direct" => Ok(Self::Direct),
            _ => bail!("invalid --h2-hier-cross-storage {value:?}; expected disk or direct"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Disk => "disk",
            Self::Direct => "direct",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H2HierOutsideStructuralPruningStrategy {
    Off,
    OutsideDominated,
}

impl H2HierOutsideStructuralPruningStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "outside-dominated" => Ok(Self::OutsideDominated),
            _ => bail!(
                "invalid --h2-hier-outside-structural-pruning {value:?}; expected off or outside-dominated"
            ),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::OutsideDominated => "outside-dominated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalH2BirthStateStrategy {
    Tagged,
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalH0UnionFindLayoutStrategy {
    ParentRank,
    Packed,
}

impl GlobalH0UnionFindLayoutStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "parent-rank" => Ok(Self::ParentRank),
            "packed" => Ok(Self::Packed),
            _ => bail!("invalid --global-h0-uf-layout {value:?}; expected parent-rank or packed"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ParentRank => "parent-rank",
            Self::Packed => "packed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalH2UnionFindLayoutStrategy {
    ParentRank,
    Packed,
}

impl GlobalH2UnionFindLayoutStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "parent-rank" => Ok(Self::ParentRank),
            "packed" => Ok(Self::Packed),
            _ => bail!("invalid --global-h2-uf-layout {value:?}; expected parent-rank or packed"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ParentRank => "parent-rank",
            Self::Packed => "packed",
        }
    }
}
impl GlobalH2BirthStateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "tagged" => Ok(Self::Tagged),
            "compact" => Ok(Self::Compact),
            _ => bail!("invalid --global-h2-birth-state {value:?}; expected tagged or compact"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Tagged => "tagged",
            Self::Compact => "compact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScalarStreamTuning {
    pub(crate) merge_strategy: MergeStrategy,
    pub(crate) interface_order: InterfaceOrderStrategy,
    pub(crate) event_order: EventOrderStrategy,
    pub(crate) f32_key_mode: F32KeyMode,
    pub(crate) neighbor_kernel: NeighborKernelStrategy,
    pub(crate) representative_active_check: RepresentativeActiveCheckStrategy,
    pub(crate) union_kernel: UnionKernelStrategy,
    pub(crate) neighbor_root_check: NeighborRootCheckStrategy,
    pub(crate) h0_pruning_cache: H0PruningCacheStrategy,
    pub(crate) h0_birth_buffer: H0BirthBufferStrategy,
    pub(crate) h0_event_storage: H0EventStorageStrategy,
    pub(crate) h0_hier_attach_pruning: H0HierAttachPruningStrategy,
    pub(crate) h2_hier_cross_storage: H2HierCrossStorageStrategy,
    pub(crate) h2_hier_outside_structural_pruning: H2HierOutsideStructuralPruningStrategy,
    pub(crate) active_state: ActiveStateStrategy,
    pub(crate) interface_state: InterfaceStateStrategy,
    pub(crate) uf_layout: UnionFindLayoutStrategy,
    pub(crate) phase_trim: PhaseTrimStrategy,
    pub(crate) local_h2_birth_state: LocalH2BirthStateStrategy,
    pub(crate) global_h0_uf_layout: GlobalH0UnionFindLayoutStrategy,
    pub(crate) global_h2_birth_state: GlobalH2BirthStateStrategy,
    pub(crate) global_h2_uf_layout: GlobalH2UnionFindLayoutStrategy,
    pub(crate) sweep_diagnostics: bool,
    pub(crate) h2_memory_audit: bool,
}

impl Default for ScalarStreamTuning {
    fn default() -> Self {
        Self {
            // CX09T1 factorial ablation (slab depth 16) selected this
            // configuration for both H0 and H2.
            merge_strategy: MergeStrategy::Scan,
            interface_order: InterfaceOrderStrategy::Radix,
            event_order: EventOrderStrategy::Verify,
            f32_key_mode: F32KeyMode::Native32,
            // CX09T1 v1.3 profiling showed the interior fast path improves both
            // H0 and H2 while preserving exact persistence.
            neighbor_kernel: NeighborKernelStrategy::InteriorFast,
            // The v1.4 CX09T1 ablation found no repeatable timing benefit from
            // removing this defensive check, so recheck remains the default.
            representative_active_check: RepresentativeActiveCheckStrategy::Recheck,
            // v1.5 CX09T1 profiling showed root-carrying is neutral for H0 and
            // consistently faster for H2, so it is the production default.
            union_kernel: UnionKernelStrategy::RootCarrying,
            // v1.6 CX09T1 profiling showed the one-link shortcut catches most
            // redundant H2 same-root encounters and improves wall time.
            neighbor_root_check: NeighborRootCheckStrategy::ParentShortcut,
            // v1.7 investigates whether the bounded 26-neighbor pruning cache
            // pays for itself in H0. 64k is the historical/reference size.
            h0_pruning_cache: H0PruningCacheStrategy::Entries64K,
            // v1.20 tests transferring the decoded F32 slab buffer directly into
            // H0 component-birth storage. Keep copy as the reference default
            // until exact CX09T1 profiling validates reuse-input.
            h0_birth_buffer: H0BirthBufferStrategy::Copy,
            // Direct event sinks are the v1.20 large-slab memory candidate.
            // Buffered remains the reference until CX09T1/F32 equivalence and profiling pass.
            h0_event_storage: H0EventStorageStrategy::Buffered,
            // v1.23 tests exact early finalization of attach events whose branch
            // birth is no older than the boundary-connected component. Keep off
            // as the reference until CX09T1 and real-prefix validation pass.
            h0_hier_attach_pruning: H0HierAttachPruningStrategy::Off,
            // v1.25 CX09T1 profiling showed direct cross-interface consumption preserves exact persistence,
            // reduces filesystem output, and improves d8/d16 runtime without a meaningful d32 regression.
            h2_hier_cross_storage: H2HierCrossStorageStrategy::Direct,
            // v1.26 ablates outside-aware structural summary pruning.
            // Keep off as the exact v1.25 reference until equivalence/profiling passes.
            h2_hier_outside_structural_pruning: H2HierOutsideStructuralPruningStrategy::Off,
            // v1.8 investigates whether the separate byte-per-voxel active
            // array can be eliminated by reserving u32::MAX in parent.
            active_state: ActiveStateStrategy::Separate,
            // v1.9 CX09T1 profiling showed the root invariant is faster for
            // both H0 and H2 and removes the explicit interface_rep vector.
            interface_state: InterfaceStateStrategy::RootInvariant,
            // v1.10 profiling showed the packed layout improves H0 and H2 and
            // removes the separate byte-per-voxel rank vector.
            uf_layout: UnionFindLayoutStrategy::Packed,
            // H0 does not use phase trimming. H2 production defaults are
            // applied mode-specifically in main after command-line parsing.
            phase_trim: PhaseTrimStrategy::Off,
            // v1.17 promotes compact local H2 births after exact CX09T1
            // equivalence and balanced real-stack profiling. Tagged remains
            // available only as the reference/debug representation.
            local_h2_birth_state: LocalH2BirthStateStrategy::Compact,
            // v1.19 promotes packed global H0 rank encoding after exact real-F32
            // equivalence and balanced profiling. Parent-rank remains available as
            // the explicit reference/debug representation.
            global_h0_uf_layout: GlobalH0UnionFindLayoutStrategy::Packed,
            // v1.14 CX09T1 profiling showed compact global H2 births are exact,
            // reduce explicit global-UF allocation by ~65%, and improve runtime.
            global_h2_birth_state: GlobalH2BirthStateStrategy::Compact,
            // v1.15 ablates the remaining separate global rank byte. Keep the
            // parent+rank layout as the reference until real-data profiling.
            global_h2_uf_layout: GlobalH2UnionFindLayoutStrategy::ParentRank,
            sweep_diagnostics: false,
            h2_memory_audit: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_default_uses_compact_local_h2_births() {
        let tuning = ScalarStreamTuning::default();
        assert_eq!(
            tuning.local_h2_birth_state,
            LocalH2BirthStateStrategy::Compact
        );
        assert_eq!(
            tuning.global_h0_uf_layout,
            GlobalH0UnionFindLayoutStrategy::Packed
        );
        assert_eq!(tuning.h0_birth_buffer, H0BirthBufferStrategy::Copy);
        assert_eq!(tuning.h0_event_storage, H0EventStorageStrategy::Buffered);
        assert_eq!(
            tuning.h0_hier_attach_pruning,
            H0HierAttachPruningStrategy::Off
        );
        assert_eq!(
            tuning.global_h2_birth_state,
            GlobalH2BirthStateStrategy::Compact
        );
        assert_eq!(
            tuning.global_h2_uf_layout,
            GlobalH2UnionFindLayoutStrategy::ParentRank
        );
    }

    #[test]
    fn parses_all_ablation_levels() {
        assert_eq!(MergeStrategy::parse("scan").unwrap(), MergeStrategy::Scan);
        assert_eq!(MergeStrategy::parse("heap").unwrap(), MergeStrategy::Heap);
        assert_eq!(MergeStrategy::parse("auto").unwrap(), MergeStrategy::Auto);
        assert_eq!(
            InterfaceOrderStrategy::parse("comparison").unwrap(),
            InterfaceOrderStrategy::Comparison
        );
        assert_eq!(
            InterfaceOrderStrategy::parse("radix").unwrap(),
            InterfaceOrderStrategy::Radix
        );
        assert_eq!(
            EventOrderStrategy::parse("resort").unwrap(),
            EventOrderStrategy::Resort
        );
        assert_eq!(
            EventOrderStrategy::parse("verify").unwrap(),
            EventOrderStrategy::Verify
        );
        assert_eq!(F32KeyMode::parse("legacy64").unwrap(), F32KeyMode::Legacy64);
        assert_eq!(F32KeyMode::parse("native32").unwrap(), F32KeyMode::Native32);
        assert_eq!(
            NeighborKernelStrategy::parse("generic").unwrap(),
            NeighborKernelStrategy::Generic
        );
        assert_eq!(
            NeighborKernelStrategy::parse("interior-fast").unwrap(),
            NeighborKernelStrategy::InteriorFast
        );
        assert_eq!(
            RepresentativeActiveCheckStrategy::parse("recheck").unwrap(),
            RepresentativeActiveCheckStrategy::Recheck
        );
        assert_eq!(
            RepresentativeActiveCheckStrategy::parse("trust-pruner").unwrap(),
            RepresentativeActiveCheckStrategy::TrustPruner
        );
        assert_eq!(
            UnionKernelStrategy::parse("conventional").unwrap(),
            UnionKernelStrategy::Conventional
        );
        assert_eq!(
            UnionKernelStrategy::parse("root-carrying").unwrap(),
            UnionKernelStrategy::RootCarrying
        );
        assert_eq!(
            NeighborRootCheckStrategy::parse("find").unwrap(),
            NeighborRootCheckStrategy::Find
        );
        assert_eq!(
            NeighborRootCheckStrategy::parse("parent-shortcut").unwrap(),
            NeighborRootCheckStrategy::ParentShortcut
        );
        assert_eq!(
            H0PruningCacheStrategy::parse("off").unwrap(),
            H0PruningCacheStrategy::Off
        );
        assert_eq!(
            H0PruningCacheStrategy::parse("4k").unwrap(),
            H0PruningCacheStrategy::Entries4K
        );
        assert_eq!(
            H0PruningCacheStrategy::parse("16k").unwrap(),
            H0PruningCacheStrategy::Entries16K
        );
        assert_eq!(
            H0PruningCacheStrategy::parse("64k").unwrap(),
            H0PruningCacheStrategy::Entries64K
        );
        assert_eq!(
            H0PruningCacheStrategy::parse("256k").unwrap(),
            H0PruningCacheStrategy::Entries256K
        );
        assert_eq!(
            ActiveStateStrategy::parse("separate").unwrap(),
            ActiveStateStrategy::Separate
        );
        assert_eq!(
            ActiveStateStrategy::parse("parent-sentinel").unwrap(),
            ActiveStateStrategy::ParentSentinel
        );
        assert_eq!(
            InterfaceStateStrategy::parse("vector").unwrap(),
            InterfaceStateStrategy::Vector
        );
        assert_eq!(
            InterfaceStateStrategy::parse("root-invariant").unwrap(),
            InterfaceStateStrategy::RootInvariant
        );
        assert_eq!(
            UnionFindLayoutStrategy::parse("parent-rank").unwrap(),
            UnionFindLayoutStrategy::ParentRank
        );
        assert_eq!(
            UnionFindLayoutStrategy::parse("packed").unwrap(),
            UnionFindLayoutStrategy::Packed
        );
        assert_eq!(
            PhaseTrimStrategy::parse("off").unwrap(),
            PhaseTrimStrategy::Off
        );
        assert_eq!(
            LocalH2BirthStateStrategy::parse("tagged").unwrap(),
            LocalH2BirthStateStrategy::Tagged
        );
        assert_eq!(
            LocalH2BirthStateStrategy::parse("compact").unwrap(),
            LocalH2BirthStateStrategy::Compact
        );
        assert_eq!(
            GlobalH2BirthStateStrategy::parse("tagged").unwrap(),
            GlobalH2BirthStateStrategy::Tagged
        );
        assert_eq!(
            GlobalH2BirthStateStrategy::parse("compact").unwrap(),
            GlobalH2BirthStateStrategy::Compact
        );
        assert_eq!(
            GlobalH0UnionFindLayoutStrategy::parse("parent-rank").unwrap(),
            GlobalH0UnionFindLayoutStrategy::ParentRank
        );
        assert_eq!(
            GlobalH0UnionFindLayoutStrategy::parse("packed").unwrap(),
            GlobalH0UnionFindLayoutStrategy::Packed
        );
        assert_eq!(
            GlobalH2UnionFindLayoutStrategy::parse("parent-rank").unwrap(),
            GlobalH2UnionFindLayoutStrategy::ParentRank
        );
        assert_eq!(
            GlobalH2UnionFindLayoutStrategy::parse("packed").unwrap(),
            GlobalH2UnionFindLayoutStrategy::Packed
        );
        assert_eq!(
            PhaseTrimStrategy::parse("before-reduce").unwrap(),
            PhaseTrimStrategy::BeforeReduce
        );
    }

    #[test]
    fn default_matches_production_h0_base() {
        let tuning = ScalarStreamTuning::default();
        assert_eq!(tuning.merge_strategy, MergeStrategy::Scan);
        assert_eq!(tuning.interface_order, InterfaceOrderStrategy::Radix);
        assert_eq!(tuning.event_order, EventOrderStrategy::Verify);
        assert_eq!(tuning.f32_key_mode, F32KeyMode::Native32);
        assert_eq!(tuning.neighbor_kernel, NeighborKernelStrategy::InteriorFast);
        assert_eq!(
            tuning.representative_active_check,
            RepresentativeActiveCheckStrategy::Recheck
        );
        assert_eq!(tuning.union_kernel, UnionKernelStrategy::RootCarrying);
        assert_eq!(
            tuning.neighbor_root_check,
            NeighborRootCheckStrategy::ParentShortcut
        );
        assert_eq!(tuning.h0_pruning_cache, H0PruningCacheStrategy::Entries64K);
        assert_eq!(tuning.active_state, ActiveStateStrategy::Separate);
        assert_eq!(
            tuning.interface_state,
            InterfaceStateStrategy::RootInvariant
        );
        assert_eq!(tuning.uf_layout, UnionFindLayoutStrategy::Packed);
        assert_eq!(tuning.phase_trim, PhaseTrimStrategy::Off);
        assert_eq!(
            tuning.global_h0_uf_layout,
            GlobalH0UnionFindLayoutStrategy::Packed
        );
        assert_eq!(
            tuning.global_h2_birth_state,
            GlobalH2BirthStateStrategy::Compact
        );
        assert_eq!(
            tuning.global_h2_uf_layout,
            GlobalH2UnionFindLayoutStrategy::ParentRank
        );
        assert!(!tuning.sweep_diagnostics);
        assert!(!tuning.h2_memory_audit);
    }
}
