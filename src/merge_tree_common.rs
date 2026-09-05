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

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent = resolve_h0_parent(merge.parent.id, &self.diagonal_parent)?;
            let previous = self.nodes.insert(
                merge.child.id,
                H0Node {
                    original_id: merge.child.id,
                    birth: merge.child.birth,
                    death: Some(merge.value),
                    parent: Some(parent),
                },
            );

            if previous.is_some() {
                bail!("H0 branch {} died more than once", merge.child.id);
            }
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

    pub(crate) fn finish_threshold(&mut self) -> Result<()> {
        for merge in self.pending_positive.drain(..) {
            let parent = resolve_h2_parent(merge.parent, &self.diagonal_parent)?;
            let parent_id = match parent {
                H2Branch::Outside => OUTSIDE_BRANCH_ID,
                H2Branch::Finite { id, .. } => id,
            };

            let previous = self.nodes.insert(
                merge.child_id,
                H2Node {
                    original_id: merge.child_id,
                    birth: Some(merge.value),
                    death: Some(merge.child_birth),
                    parent: Some(parent_id),
                    outside: false,
                },
            );

            if previous.is_some() {
                bail!("H2 branch {} died more than once", merge.child_id);
            }
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
