//! Exact, linear-memory sparsification of edges between neighboring slab faces.
//!
//! The complete 26-connected bipartite face graph has almost nine edges per
//! face voxel. Materializing all of those edges creates an avoidable memory
//! spike. We instead activate the `2 * width * height` face vertices in
//! filtration order. When a vertex is activated, every edge to an earlier
//! vertex has just become available. Applying union--find to those edges is
//! exactly Kruskal's algorithm, but no candidate-edge array is needed.

use anyhow::{Result, bail};

use crate::connectivity::Connectivity;

const NUM_U16_VALUES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterfaceFiltration {
    /// Foreground sublevel filtration. An edge appears at the larger endpoint
    /// value and events are processed from small to large.
    SublevelMax,

    /// Background superlevel filtration. An edge appears at the smaller
    /// endpoint value and events are processed from large to small.
    SuperlevelMin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseCrossEdge {
    pub value: u16,
    pub left_face_index: u32,
    pub right_face_index: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct InterfaceSparsificationStats {
    /// Number of edges in the complete cross-face adjacency graph.
    pub candidate_edges: u64,
    /// Number of edges retained in the filtration-preserving forest.
    pub retained_edges: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InterfaceSpec {
    pub width: usize,
    pub height: usize,
    pub connectivity: Connectivity,
    pub filtration: InterfaceFiltration,
}

#[derive(Debug)]
struct CompactUnionFind {
    /// A root stores the negative component size; any other node stores its
    /// parent. This uses half the metadata of a separate parent and rank pair.
    parent_or_size: Vec<i32>,
}

impl CompactUnionFind {
    fn new(node_count: usize) -> Result<Self> {
        if node_count > i32::MAX as usize {
            bail!(
                "interface graph has {node_count} nodes; the compact union-find supports at most {}",
                i32::MAX
            );
        }
        Ok(Self {
            parent_or_size: vec![-1; node_count],
        })
    }

    fn find(&mut self, x: u32) -> u32 {
        let mut root = x as usize;
        while self.parent_or_size[root] >= 0 {
            root = self.parent_or_size[root] as usize;
        }

        let mut current = x as usize;
        while self.parent_or_size[current] >= 0 {
            let parent = self.parent_or_size[current] as usize;
            self.parent_or_size[current] = root as i32;
            current = parent;
        }
        root as u32
    }

    fn union(&mut self, a: u32, b: u32) -> bool {
        let mut root_a = self.find(a) as usize;
        let mut root_b = self.find(b) as usize;
        if root_a == root_b {
            return false;
        }

        // More negative means a larger component.
        if self.parent_or_size[root_a] > self.parent_or_size[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent_or_size[root_a] += self.parent_or_size[root_b];
        self.parent_or_size[root_b] = root_a as i32;
        true
    }
}

fn basic_face_size(width: usize, height: usize) -> Result<usize> {
    let face_size = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("interface face size overflow"))?;
    if face_size > u32::MAX as usize {
        bail!("interface face has {face_size} vertices; u32 face indices are required");
    }
    Ok(face_size)
}

fn checked_face_size(width: usize, height: usize) -> Result<usize> {
    let face_size = basic_face_size(width, height)?;
    if face_size > (i32::MAX as usize) / 2 {
        bail!(
            "two interface faces contain {} nodes; the compact union-find limit is {}",
            face_size.saturating_mul(2),
            i32::MAX
        );
    }
    Ok(face_size)
}

fn vertex_value<T: Copy>(
    vertex: u32,
    face_size: usize,
    left_values: &[T],
    right_values: &[T],
) -> T {
    let index = vertex as usize;
    if index < face_size {
        left_values[index]
    } else {
        right_values[index - face_size]
    }
}

fn is_earlier<T: Ord>(
    neighbor_value: T,
    neighbor_id: u32,
    current_value: T,
    current_id: u32,
    filtration: InterfaceFiltration,
) -> bool {
    match filtration {
        InterfaceFiltration::SublevelMax => {
            (neighbor_value, neighbor_id) < (current_value, current_id)
        }
        InterfaceFiltration::SuperlevelMin => {
            (neighbor_value, neighbor_id) > (current_value, current_id)
        }
    }
}

/// Shared vertex-activation implementation used by integer and scalar faces.
///
/// `ascending_vertices` must contain every local vertex ID exactly once and be
/// sorted by `(value, vertex_id)` from small to large. The superlevel sweep
/// reads that order in reverse. The callback therefore always receives edges
/// in filtration order.
pub(crate) fn sparsify_ordered_interface<T: Copy + Ord>(
    left_values: &[T],
    right_values: &[T],
    spec: InterfaceSpec,
    ascending_vertices: &[u32],
    mut emit: impl FnMut(T, u32, u32) -> Result<()>,
) -> Result<InterfaceSparsificationStats> {
    let face_size = if matches!(spec.connectivity, Connectivity::TwentySix) {
        checked_face_size(spec.width, spec.height)?
    } else {
        basic_face_size(spec.width, spec.height)?
    };
    if left_values.len() != face_size {
        bail!(
            "left interface has {} values, expected {face_size}",
            left_values.len()
        );
    }
    if right_values.len() != face_size {
        bail!(
            "right interface has {} values, expected {face_size}",
            right_values.len()
        );
    }

    let node_count = face_size
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("interface node count overflow"))?;
    if ascending_vertices.len() != node_count {
        bail!(
            "vertex order has {} entries, expected {node_count}",
            ascending_vertices.len()
        );
    }

    let right_offset = face_size as u32;
    let mut union_find = if matches!(spec.connectivity, Connectivity::TwentySix) {
        Some(CompactUnionFind::new(node_count)?)
    } else {
        None
    };
    let mut candidate_edges = 0u64;
    let mut retained_edges = 0u64;

    let mut process_vertex = |vertex: u32| -> Result<()> {
        if vertex as usize >= node_count {
            bail!("vertex order contains out-of-range ID {vertex}");
        }
        let current_value = vertex_value(vertex, face_size, left_values, right_values);
        let (on_left, face_index) = if vertex < right_offset {
            (true, vertex as usize)
        } else {
            (false, (vertex - right_offset) as usize)
        };
        let x = face_index % spec.width;
        let y = face_index / spec.width;

        let radius = if matches!(spec.connectivity, Connectivity::Six) {
            0isize
        } else {
            1isize
        };
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= spec.width as isize || ny >= spec.height as isize {
                    continue;
                }

                let other_face_index = ny as usize * spec.width + nx as usize;
                let neighbor = if on_left {
                    right_offset + other_face_index as u32
                } else {
                    other_face_index as u32
                };
                let neighbor_value = vertex_value(neighbor, face_size, left_values, right_values);

                // Exactly one endpoint sees each edge: the endpoint that is
                // later in the chosen filtration order.
                if !is_earlier(
                    neighbor_value,
                    neighbor,
                    current_value,
                    vertex,
                    spec.filtration,
                ) {
                    continue;
                }

                candidate_edges = candidate_edges
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("candidate edge count overflow"))?;

                let keep = match union_find.as_mut() {
                    Some(uf) => uf.union(vertex, neighbor),
                    // Under 6-connectivity the cross graph is a matching, so
                    // every edge is necessary and no union--find is needed.
                    None => true,
                };
                if keep {
                    let (left_index, right_index) = if on_left {
                        (face_index as u32, other_face_index as u32)
                    } else {
                        (other_face_index as u32, face_index as u32)
                    };
                    emit(current_value, left_index, right_index)?;
                    retained_edges = retained_edges
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("retained edge count overflow"))?;
                }
            }
        }
        Ok(())
    };

    match spec.filtration {
        InterfaceFiltration::SublevelMax => {
            for &vertex in ascending_vertices {
                process_vertex(vertex)?;
            }
        }
        InterfaceFiltration::SuperlevelMin => {
            for &vertex in ascending_vertices.iter().rev() {
                process_vertex(vertex)?;
            }
        }
    }

    Ok(InterfaceSparsificationStats {
        candidate_edges,
        retained_edges,
    })
}

/// Counting-sort the two U16 faces by `(value, vertex_id)`.
fn sorted_u16_face_vertices(left_values: &[u16], right_values: &[u16]) -> Result<Vec<u32>> {
    let node_count = left_values
        .len()
        .checked_add(right_values.len())
        .ok_or_else(|| anyhow::anyhow!("interface node count overflow"))?;
    if node_count > u32::MAX as usize {
        bail!("interface has too many vertices for u32 IDs");
    }

    let mut counts = vec![0usize; NUM_U16_VALUES];
    for &value in left_values.iter().chain(right_values) {
        counts[value as usize] = counts[value as usize]
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("interface value count overflow"))?;
    }

    let mut offsets = vec![0usize; NUM_U16_VALUES];
    let mut next = 0usize;
    for (value, count) in counts.into_iter().enumerate() {
        offsets[value] = next;
        next = next
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("interface order offset overflow"))?;
    }

    let mut cursors = offsets;
    let mut order = vec![0u32; node_count];
    for vertex in 0..node_count {
        let value = if vertex < left_values.len() {
            left_values[vertex]
        } else {
            right_values[vertex - left_values.len()]
        };
        let position = cursors[value as usize];
        order[position] = vertex as u32;
        cursors[value as usize] += 1;
    }
    Ok(order)
}

/// Emit the 6-connected matching in edge-filtration order.
///
/// Every matching edge is necessary, so this path needs neither the 2A vertex
/// order nor a union-find. A counting-ordered list of A edge indices is kept
/// only because disk runs require sorted records.
fn emit_u16_matching(
    left_values: &[u16],
    right_values: &[u16],
    width: usize,
    height: usize,
    filtration: InterfaceFiltration,
    emit: &mut impl FnMut(SparseCrossEdge) -> Result<()>,
) -> Result<InterfaceSparsificationStats> {
    let face_size = basic_face_size(width, height)?;
    if left_values.len() != face_size || right_values.len() != face_size {
        bail!(
            "6-connected interface lengths are {}, {}, expected {face_size}",
            left_values.len(),
            right_values.len()
        );
    }

    let mut counts = vec![0usize; NUM_U16_VALUES];
    for index in 0..face_size {
        let value = match filtration {
            InterfaceFiltration::SublevelMax => left_values[index].max(right_values[index]),
            InterfaceFiltration::SuperlevelMin => left_values[index].min(right_values[index]),
        };
        counts[value as usize] += 1;
    }
    let mut offsets = vec![0usize; NUM_U16_VALUES];
    let mut next = 0usize;
    for (value, count) in counts.into_iter().enumerate() {
        offsets[value] = next;
        next = next
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("matching-edge order overflow"))?;
    }
    let mut cursors = offsets;
    let mut order = vec![0u32; face_size];
    for index in 0..face_size {
        let value = match filtration {
            InterfaceFiltration::SublevelMax => left_values[index].max(right_values[index]),
            InterfaceFiltration::SuperlevelMin => left_values[index].min(right_values[index]),
        };
        order[cursors[value as usize]] = index as u32;
        cursors[value as usize] += 1;
    }

    let mut emit_index = |index: u32| -> Result<()> {
        let position = index as usize;
        let value = match filtration {
            InterfaceFiltration::SublevelMax => left_values[position].max(right_values[position]),
            InterfaceFiltration::SuperlevelMin => left_values[position].min(right_values[position]),
        };
        emit(SparseCrossEdge {
            value,
            left_face_index: index,
            right_face_index: index,
        })
    };
    match filtration {
        InterfaceFiltration::SublevelMax => {
            for index in order {
                emit_index(index)?;
            }
        }
        InterfaceFiltration::SuperlevelMin => {
            for index in order.into_iter().rev() {
                emit_index(index)?;
            }
        }
    }
    let edge_count = face_size as u64;
    Ok(InterfaceSparsificationStats {
        candidate_edges: edge_count,
        retained_edges: edge_count,
    })
}

/// Construct a filtration-preserving spanning forest of the adjacency graph
/// between two neighboring U16 slab faces.
pub(crate) fn sparsify_cross_interface(
    left_values: &[u16],
    right_values: &[u16],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    filtration: InterfaceFiltration,
    mut emit: impl FnMut(SparseCrossEdge) -> Result<()>,
) -> Result<InterfaceSparsificationStats> {
    if matches!(connectivity, Connectivity::Six) {
        return emit_u16_matching(
            left_values,
            right_values,
            width,
            height,
            filtration,
            &mut emit,
        );
    }
    let order = sorted_u16_face_vertices(left_values, right_values)?;
    sparsify_ordered_interface(
        left_values,
        right_values,
        InterfaceSpec {
            width,
            height,
            connectivity,
            filtration,
        },
        &order,
        |value, left_face_index, right_face_index| {
            emit(SparseCrossEdge {
                value,
                left_face_index,
                right_face_index,
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn component_labels(
        node_count: usize,
        edges: &[(u16, u32, u32)],
        threshold: u16,
        filtration: InterfaceFiltration,
    ) -> Vec<u32> {
        let mut uf = CompactUnionFind::new(node_count).unwrap();
        for &(value, left, right) in edges {
            let active = match filtration {
                InterfaceFiltration::SublevelMax => value <= threshold,
                InterfaceFiltration::SuperlevelMin => value >= threshold,
            };
            if active {
                uf.union(left, right);
            }
        }
        (0..node_count as u32).map(|node| uf.find(node)).collect()
    }

    fn same_partition(labels_a: &[u32], labels_b: &[u32]) -> bool {
        (0..labels_a.len()).all(|i| {
            (0..labels_a.len())
                .all(|j| (labels_a[i] == labels_a[j]) == (labels_b[i] == labels_b[j]))
        })
    }

    fn complete_edges(
        left: &[u16],
        right: &[u16],
        width: usize,
        height: usize,
        connectivity: Connectivity,
        filtration: InterfaceFiltration,
    ) -> Vec<(u16, u32, u32)> {
        let face_size = width * height;
        let mut result = Vec::new();
        let radius = if matches!(connectivity, Connectivity::Six) {
            0isize
        } else {
            1isize
        };
        for y in 0..height {
            for x in 0..width {
                let left_index = y * width + x;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                            continue;
                        }
                        let right_index = ny as usize * width + nx as usize;
                        let value = match filtration {
                            InterfaceFiltration::SublevelMax => {
                                left[left_index].max(right[right_index])
                            }
                            InterfaceFiltration::SuperlevelMin => {
                                left[left_index].min(right[right_index])
                            }
                        };
                        result.push((
                            value,
                            left_index as u32,
                            face_size as u32 + right_index as u32,
                        ));
                    }
                }
            }
        }
        result
    }

    fn assert_prefix_partitions(
        left: &[u16],
        right: &[u16],
        width: usize,
        height: usize,
        connectivity: Connectivity,
        filtration: InterfaceFiltration,
    ) {
        let face_size = width * height;
        let complete = complete_edges(left, right, width, height, connectivity, filtration);
        let mut forest = Vec::new();
        let stats = sparsify_cross_interface(
            left,
            right,
            width,
            height,
            connectivity,
            filtration,
            |edge| {
                forest.push((
                    edge.value,
                    edge.left_face_index,
                    face_size as u32 + edge.right_face_index,
                ));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.candidate_edges as usize, complete.len());

        let thresholds: BTreeSet<u16> = left.iter().chain(right).copied().collect();
        for threshold in thresholds {
            let complete_labels = component_labels(2 * face_size, &complete, threshold, filtration);
            let forest_labels = component_labels(2 * face_size, &forest, threshold, filtration);
            assert!(same_partition(&complete_labels, &forest_labels));
        }
    }

    #[test]
    fn six_connectivity_keeps_matching_edges_in_filtration_order() {
        let left = [1u16, 4, 2, 8];
        let right = [3u16, 2, 7, 5];
        let mut emitted = Vec::new();
        let stats = sparsify_cross_interface(
            &left,
            &right,
            2,
            2,
            Connectivity::Six,
            InterfaceFiltration::SublevelMax,
            |edge| {
                emitted.push(edge);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.candidate_edges, 4);
        assert_eq!(stats.retained_edges, 4);
        assert!(
            emitted
                .windows(2)
                .all(|pair| pair[0].value <= pair[1].value)
        );
    }

    #[test]
    fn constant_twenty_six_interface_reduces_to_a_spanning_tree() {
        let left = [10u16; 4];
        let right = [10u16; 4];
        let mut emitted = Vec::new();
        let stats = sparsify_cross_interface(
            &left,
            &right,
            2,
            2,
            Connectivity::TwentySix,
            InterfaceFiltration::SublevelMax,
            |edge| {
                emitted.push(edge);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.candidate_edges, 16);
        assert_eq!(stats.retained_edges, 7);
    }

    #[test]
    fn exhaustive_binary_two_by_two_faces_preserve_every_prefix() {
        for mask in 0u16..=255 {
            let mut values = [0u16; 8];
            for (bit, value) in values.iter_mut().enumerate() {
                *value = (mask >> bit) & 1;
            }
            let (left, right) = values.split_at(4);
            for connectivity in [Connectivity::Six, Connectivity::TwentySix] {
                for filtration in [
                    InterfaceFiltration::SublevelMax,
                    InterfaceFiltration::SuperlevelMin,
                ] {
                    assert_prefix_partitions(left, right, 2, 2, connectivity, filtration);
                }
            }
        }
    }
}
