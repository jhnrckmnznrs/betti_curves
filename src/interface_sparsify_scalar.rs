//! Scalar-key wrapper around the shared cross-face vertex-activation sweep.

use anyhow::{Result, bail};

use crate::connectivity::Connectivity;
use crate::interface_sparsify::{
    InterfaceFiltration, InterfaceSparsificationStats, InterfaceSpec, sparsify_ordered_interface,
};
use crate::scalar::ScalarKey;
use crate::scalar_order::{RadixScalarKey, radix_order_generic_by_key};
use crate::scalar_stream_tuning::InterfaceOrderStrategy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseScalarCrossEdge<K = ScalarKey> {
    pub value: K,
    pub left_face_index: u32,
    pub right_face_index: u32,
}

pub(crate) type ScalarInterfaceSparsificationStats = InterfaceSparsificationStats;

fn sorted_scalar_face_vertices<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    order_strategy: InterfaceOrderStrategy,
) -> Result<Vec<u32>> {
    let node_count = left_values
        .len()
        .checked_add(right_values.len())
        .ok_or_else(|| anyhow::anyhow!("scalar interface node count overflow"))?;
    if node_count > u32::MAX as usize {
        bail!("scalar interface has too many vertices for u32 IDs");
    }

    let value_at = |index: usize| {
        if index < left_values.len() {
            left_values[index]
        } else {
            right_values[index - left_values.len()]
        }
    };

    match order_strategy {
        InterfaceOrderStrategy::Radix => Ok(radix_order_generic_by_key(node_count, value_at)),
        InterfaceOrderStrategy::Comparison => {
            let mut order: Vec<u32> = (0..node_count as u32).collect();
            order.sort_unstable_by_key(|&vertex| (value_at(vertex as usize), vertex));
            Ok(order)
        }
    }
}

fn emit_scalar_matching<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    filtration: InterfaceFiltration,
    order_strategy: InterfaceOrderStrategy,
    emit: &mut impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    if left_values.len() > u32::MAX as usize {
        bail!("scalar interface face has too many vertices for u32 indices");
    }
    let edge_value = |index: usize| match filtration {
        InterfaceFiltration::SublevelMax => left_values[index].max(right_values[index]),
        InterfaceFiltration::SuperlevelMin => left_values[index].min(right_values[index]),
    };
    let order = match order_strategy {
        InterfaceOrderStrategy::Radix => radix_order_generic_by_key(left_values.len(), edge_value),
        InterfaceOrderStrategy::Comparison => {
            let mut order: Vec<u32> = (0..left_values.len() as u32).collect();
            order.sort_unstable_by_key(|&index| (edge_value(index as usize), index));
            order
        }
    };

    let mut emit_index = |index: u32| {
        emit(SparseScalarCrossEdge {
            value: edge_value(index as usize),
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
    let edge_count = left_values.len() as u64;
    Ok(InterfaceSparsificationStats {
        candidate_edges: edge_count,
        retained_edges: edge_count,
    })
}

#[allow(clippy::too_many_arguments)] // Internal interface kernel keeps filtration knobs explicit.
fn sparsify_scalar_interface<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    filtration: InterfaceFiltration,
    order_strategy: InterfaceOrderStrategy,
    mut emit: impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    let face_size = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("scalar interface face size overflow"))?;
    if left_values.len() != face_size || right_values.len() != face_size {
        bail!("scalar interface face length does not match width * height");
    }

    if matches!(connectivity, Connectivity::Six) {
        return emit_scalar_matching(
            left_values,
            right_values,
            filtration,
            order_strategy,
            &mut emit,
        );
    }

    let order = sorted_scalar_face_vertices(left_values, right_values, order_strategy)?;
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
            emit(SparseScalarCrossEdge {
                value,
                left_face_index,
                right_face_index,
            })
        },
    )
}

/// Produce a filtration-preserving forest for a foreground sublevel face.
pub(crate) fn sparsify_scalar_sublevel_interface_with_order<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    order_strategy: InterfaceOrderStrategy,
    emit: impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    sparsify_scalar_interface(
        left_values,
        right_values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SublevelMax,
        order_strategy,
        emit,
    )
}

pub(crate) fn sparsify_scalar_sublevel_interface<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    emit: impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    sparsify_scalar_sublevel_interface_with_order(
        left_values,
        right_values,
        width,
        height,
        connectivity,
        InterfaceOrderStrategy::Radix,
        emit,
    )
}

/// Produce a filtration-preserving forest for a background superlevel face.
pub(crate) fn sparsify_scalar_superlevel_interface_with_order<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    order_strategy: InterfaceOrderStrategy,
    emit: impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    sparsify_scalar_interface(
        left_values,
        right_values,
        width,
        height,
        connectivity,
        InterfaceFiltration::SuperlevelMin,
        order_strategy,
        emit,
    )
}

pub(crate) fn sparsify_scalar_superlevel_interface<K: RadixScalarKey>(
    left_values: &[K],
    right_values: &[K],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    emit: impl FnMut(SparseScalarCrossEdge<K>) -> Result<()>,
) -> Result<ScalarInterfaceSparsificationStats> {
    sparsify_scalar_superlevel_interface_with_order(
        left_values,
        right_values,
        width,
        height,
        connectivity,
        InterfaceOrderStrategy::Radix,
        emit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(values: &[u16]) -> Vec<ScalarKey> {
        values.iter().copied().map(ScalarKey::from_u16).collect()
    }

    #[test]
    fn constant_twenty_six_interface_reduces_to_a_spanning_tree() {
        let left = keys(&[10; 4]);
        let right = keys(&[10; 4]);
        let mut retained = Vec::new();
        let stats = sparsify_scalar_sublevel_interface(
            &left,
            &right,
            2,
            2,
            Connectivity::TwentySix,
            |edge| {
                retained.push(edge);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.candidate_edges, 16);
        assert_eq!(stats.retained_edges, 7);
    }

    #[test]
    fn superlevel_edges_are_emitted_in_descending_order() {
        let left = keys(&[10, 30]);
        let right = keys(&[20, 40]);
        let mut values = Vec::new();
        sparsify_scalar_superlevel_interface(&left, &right, 2, 1, Connectivity::Six, |edge| {
            values.push(edge.value.to_f64());
            Ok(())
        })
        .unwrap();
        assert_eq!(values, vec![30.0, 10.0]);
    }

    #[test]
    fn radix_face_vertex_order_matches_comparison_oracle() {
        let left = keys(&[4, 1, 4, 9, 1, 3]);
        let right = keys(&[2, 4, 1, 9, 8, 3]);
        let radix =
            sorted_scalar_face_vertices(&left, &right, InterfaceOrderStrategy::Radix).unwrap();
        let mut comparison: Vec<u32> = (0..(left.len() + right.len()) as u32).collect();
        comparison.sort_unstable_by_key(|&vertex| {
            let index = vertex as usize;
            let value = if index < left.len() {
                left[index]
            } else {
                right[index - left.len()]
            };
            (value, vertex)
        });
        assert_eq!(radix, comparison);
    }

    #[test]
    fn six_connected_radix_emission_matches_comparison_oracle() {
        let left = keys(&[7, 1, 5, 5, 2, 9]);
        let right = keys(&[3, 8, 5, 1, 2, 4]);
        for filtration in [
            InterfaceFiltration::SublevelMax,
            InterfaceFiltration::SuperlevelMin,
        ] {
            let edge_value = |index: usize| match filtration {
                InterfaceFiltration::SublevelMax => left[index].max(right[index]),
                InterfaceFiltration::SuperlevelMin => left[index].min(right[index]),
            };
            let mut comparison: Vec<u32> = (0..left.len() as u32).collect();
            comparison.sort_unstable_by_key(|&index| (edge_value(index as usize), index));
            if matches!(filtration, InterfaceFiltration::SuperlevelMin) {
                comparison.reverse();
            }

            let mut emitted = Vec::new();
            emit_scalar_matching(
                &left,
                &right,
                filtration,
                InterfaceOrderStrategy::Radix,
                &mut |edge| {
                    emitted.push(edge.left_face_index);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(emitted, comparison);
        }
    }
    #[test]
    fn comparison_and_radix_full_interface_emit_same_edges() {
        let left = keys(&[7, 1, 5, 5, 2, 9, 4, 4, 8]);
        let right = keys(&[3, 8, 5, 1, 2, 4, 9, 4, 0]);
        for filtration in [
            InterfaceFiltration::SublevelMax,
            InterfaceFiltration::SuperlevelMin,
        ] {
            let mut comparison = Vec::new();
            sparsify_scalar_interface(
                &left,
                &right,
                3,
                3,
                Connectivity::TwentySix,
                filtration,
                InterfaceOrderStrategy::Comparison,
                |edge| {
                    comparison.push(edge);
                    Ok(())
                },
            )
            .unwrap();

            let mut radix = Vec::new();
            sparsify_scalar_interface(
                &left,
                &right,
                3,
                3,
                Connectivity::TwentySix,
                filtration,
                InterfaceOrderStrategy::Radix,
                |edge| {
                    radix.push(edge);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(radix, comparison);
        }
    }
}
