use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    Six,
    TwentySix,
}

pub fn parse_connectivity(s: &str) -> Result<Connectivity> {
    match s {
        "6" | "six" | "Six" => Ok(Connectivity::Six),
        "26" | "twenty-six" | "TwentySix" | "twentysix" => Ok(Connectivity::TwentySix),
        _ => bail!("Connectivity must be either 6 or 26, got {:?}", s),
    }
}

pub fn dual_connectivity(foreground: Connectivity) -> Connectivity {
    match foreground {
        Connectivity::Six => Connectivity::TwentySix,
        Connectivity::TwentySix => Connectivity::Six,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Betti0,
    Betti2,
    Both,
    EventBetti0,
    EventBetti0Stream,
    EventBetti0ScalarStream,
    EventBetti2,
    EventBetti2Stream,
    EventBetti2ScalarStream,
    PersistenceH0,
    PersistenceH0Stream,
    PersistenceH2,
    PersistenceH2Stream,
    PersistenceH0Scalar,
    PersistenceH0ScalarHierarchical,
    PersistenceH0ScalarHierarchicalStream,
    PersistenceH0ScalarBatch,
    PersistenceH0ScalarStream,
    PersistenceH0ScalarBatchStream,
    PersistenceH2Scalar,
    PersistenceH2ScalarBatch,
    PersistenceH2ScalarStream,
    PersistenceH2ScalarHierarchicalStream,
    PersistenceH2ScalarBatchStream,
    MergeTreeH0,
    MergeTreeH0Hierarchical,
    MergeTreeH0Stream,
    MergeTreeH2,
    MergeTreeH2Hierarchical,
    MergeTreeH2Stream,
}

pub fn parse_mode(s: &str) -> Result<Mode> {
    match s {
        "betti0" | "b0" | "0" => Ok(Mode::Betti0),
        "betti2" | "b2" | "2" => Ok(Mode::Betti2),
        "both" | "all" => Ok(Mode::Both),
        "event-betti0" | "event_betti0" | "eb0" => Ok(Mode::EventBetti0),
        "event-betti0-stream" | "event_betti0_stream" | "eb0-stream" => Ok(Mode::EventBetti0Stream),
        "event-betti0-scalar-stream" | "event_betti0_scalar_stream" | "eb0-scalar-stream" => {
            Ok(Mode::EventBetti0ScalarStream)
        }
        "event-betti2" | "event_betti2" | "eb2" => Ok(Mode::EventBetti2),
        "event-betti2-stream" | "event_betti2_stream" | "eb2-stream" => Ok(Mode::EventBetti2Stream),
        "event-betti2-scalar-stream" | "event_betti2_scalar_stream" | "eb2-scalar-stream" => {
            Ok(Mode::EventBetti2ScalarStream)
        }
        "h0" | "persistence-h0" | "persistence_h0" | "ph0" => Ok(Mode::PersistenceH0),
        "h0-stream" | "persistence-h0-stream" | "persistence_h0_stream" | "ph0-stream" => {
            Ok(Mode::PersistenceH0Stream)
        }
        "h0-scalar" | "h0-float" | "persistence-h0-scalar" | "ph0-scalar" => {
            Ok(Mode::PersistenceH0Scalar)
        }
        "h0-scalar-hierarchical"
        | "h0-float-hierarchical"
        | "persistence-h0-scalar-hierarchical" => Ok(Mode::PersistenceH0ScalarHierarchical),
        "h0-scalar-hierarchical-stream"
        | "h0-float-hierarchical-stream"
        | "persistence-h0-scalar-hierarchical-stream" => {
            Ok(Mode::PersistenceH0ScalarHierarchicalStream)
        }
        "h0-scalar-batch" | "h0-float-batch" | "persistence-h0-scalar-batch" => {
            Ok(Mode::PersistenceH0ScalarBatch)
        }
        "h0-scalar-stream" | "h0-float-stream" | "persistence-h0-scalar-stream" => {
            Ok(Mode::PersistenceH0ScalarStream)
        }
        "h0-scalar-batch-stream"
        | "h0-float-batch-stream"
        | "persistence-h0-scalar-batch-stream" => Ok(Mode::PersistenceH0ScalarBatchStream),
        "h2" | "persistence-h2" | "persistence_h2" | "ph2" => Ok(Mode::PersistenceH2),
        "h2-stream" | "persistence-h2-stream" | "persistence_h2_stream" | "ph2-stream" => {
            Ok(Mode::PersistenceH2Stream)
        }
        "h2-scalar" | "h2-float" | "persistence-h2-scalar" | "ph2-scalar" => {
            Ok(Mode::PersistenceH2Scalar)
        }
        "h2-scalar-batch" | "h2-float-batch" | "persistence-h2-scalar-batch" => {
            Ok(Mode::PersistenceH2ScalarBatch)
        }
        "h2-scalar-stream" | "h2-float-stream" | "persistence-h2-scalar-stream" => {
            Ok(Mode::PersistenceH2ScalarStream)
        }
        "h2-scalar-hierarchical-stream"
        | "h2-float-hierarchical-stream"
        | "persistence-h2-scalar-hierarchical-stream" => {
            Ok(Mode::PersistenceH2ScalarHierarchicalStream)
        }
        "h2-scalar-batch-stream"
        | "h2-float-batch-stream"
        | "persistence-h2-scalar-batch-stream" => Ok(Mode::PersistenceH2ScalarBatchStream),
        "branch-tree-h0" | "branch_tree_h0" | "bt-h0" | "merge-tree-h0" | "merge_tree_h0"
        | "mt-h0" => Ok(Mode::MergeTreeH0),
        "branch-tree-h0-hierarchical"
        | "branch_tree_h0_hierarchical"
        | "bt-h0-hierarchical"
        | "merge-tree-h0-hierarchical"
        | "merge_tree_h0_hierarchical"
        | "mt-h0-hierarchical" => Ok(Mode::MergeTreeH0Hierarchical),
        "branch-tree-h0-stream"
        | "branch_tree_h0_stream"
        | "bt-h0-stream"
        | "merge-tree-h0-stream"
        | "merge_tree_h0_stream"
        | "mt-h0-stream" => Ok(Mode::MergeTreeH0Stream),
        "branch-tree-h2" | "branch_tree_h2" | "bt-h2" | "merge-tree-h2" | "merge_tree_h2"
        | "mt-h2" => Ok(Mode::MergeTreeH2),
        "branch-tree-h2-hierarchical"
        | "branch_tree_h2_hierarchical"
        | "bt-h2-hierarchical"
        | "merge-tree-h2-hierarchical"
        | "merge_tree_h2_hierarchical"
        | "mt-h2-hierarchical" => Ok(Mode::MergeTreeH2Hierarchical),
        "branch-tree-h2-stream"
        | "branch_tree_h2_stream"
        | "bt-h2-stream"
        | "merge-tree-h2-stream"
        | "merge_tree_h2_stream"
        | "mt-h2-stream" => Ok(Mode::MergeTreeH2Stream),
        _ => bail!(
            "Mode must be betti0, betti2, both, event-betti0, \
             event-betti0-stream, event-betti0-scalar-stream, event-betti2, \
             event-betti2-stream, event-betti2-scalar-stream, \
             h0, h0-stream, h0-scalar, h0-scalar-hierarchical, h0-scalar-hierarchical-stream, h0-scalar-stream, h0-scalar-batch, h0-scalar-batch-stream, h2, h2-stream, h2-scalar, h2-scalar-stream, h2-scalar-hierarchical-stream, h2-scalar-batch, h2-scalar-batch-stream, branch-tree-h0, \
             branch-tree-h0-hierarchical, branch-tree-h0-stream, branch-tree-h2, branch-tree-h2-hierarchical, or branch-tree-h2-stream; got {:?}",
            s
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, parse_mode};

    #[test]
    fn hierarchical_scalar_modes_are_registered() {
        assert!(matches!(
            parse_mode("h0-scalar-hierarchical-stream"),
            Ok(Mode::PersistenceH0ScalarHierarchicalStream)
        ));
        assert!(matches!(
            parse_mode("h2-scalar-hierarchical-stream"),
            Ok(Mode::PersistenceH2ScalarHierarchicalStream)
        ));
        assert!(matches!(
            parse_mode("branch-tree-h0-hierarchical"),
            Ok(Mode::MergeTreeH0Hierarchical)
        ));
        assert!(matches!(
            parse_mode("branch-tree-h2-hierarchical"),
            Ok(Mode::MergeTreeH2Hierarchical)
        ));
    }
}
