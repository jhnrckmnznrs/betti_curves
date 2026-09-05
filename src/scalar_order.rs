//! Ordering backends for exact scalar sweeps.
//!
//! U8 and U16 stacks use counting sort because their alphabets are bounded.
//! F64 stacks use an eight-pass least-significant-digit radix sort on the
//! order-preserving `ScalarKey` representation. Native F32 streaming uses the
//! four-byte `F32Key` representation and therefore needs only four byte passes.
//! All routes return the exact same value-then-voxel-ID order as comparison
//! sorting.

use crate::io_scalar::ScalarPixelType;
use crate::scalar::{F32Key, ScalarKey};

pub(crate) trait RadixScalarKey: Copy + Ord {
    const RADIX_BYTES: usize;
    fn raw_u64(self) -> u64;
}

impl RadixScalarKey for ScalarKey {
    const RADIX_BYTES: usize = 8;

    #[inline]
    fn raw_u64(self) -> u64 {
        self.raw()
    }
}

impl RadixScalarKey for F32Key {
    const RADIX_BYTES: usize = 4;

    #[inline]
    fn raw_u64(self) -> u64 {
        u64::from(self.raw())
    }
}

pub(crate) fn sorted_scalar_indices(values: &[ScalarKey], pixel_type: ScalarPixelType) -> Vec<u32> {
    assert!(
        values.len() <= u32::MAX as usize,
        "scalar slab has too many voxels for u32 indices"
    );

    match pixel_type {
        ScalarPixelType::U8 => counting_order(values, 256),
        ScalarPixelType::U16 => counting_order(values, 65_536),
        ScalarPixelType::F32 | ScalarPixelType::F64 => radix_order(values),
    }
}

/// Exact stable value-then-index order for native-width F32 keys.
pub(crate) fn sorted_f32_indices(values: &[F32Key]) -> Vec<u32> {
    radix_order_f32_by_key(values.len(), |index| values[index])
}

fn counting_order(values: &[ScalarKey], alphabet_size: usize) -> Vec<u32> {
    let mut counts = vec![0usize; alphabet_size];
    for &value in values {
        let bucket = value.to_f64() as usize;
        assert!(
            bucket < alphabet_size,
            "integer ScalarKey exceeds source alphabet"
        );
        counts[bucket] += 1;
    }

    let mut offsets = vec![0usize; alphabet_size];
    let mut next = 0usize;
    for (bucket, count) in counts.into_iter().enumerate() {
        offsets[bucket] = next;
        next += count;
    }

    let mut order = vec![0u32; values.len()];
    for (index, &value) in values.iter().enumerate() {
        let bucket = value.to_f64() as usize;
        let position = offsets[bucket];
        order[position] = index as u32;
        offsets[bucket] += 1;
    }
    order
}

fn radix_order(values: &[ScalarKey]) -> Vec<u32> {
    radix_order_by_key(values.len(), |index| values[index])
}

/// Stable exact radix order for an implicit array of `ScalarKey`s.
///
/// Indices start in increasing order and every radix pass is stable, so ties
/// are resolved by the original index. This is therefore exactly equivalent
/// to sorting by `(key(index), index)` while avoiding comparison sorting.
pub(crate) fn radix_order_by_key(len: usize, key: impl Fn(usize) -> ScalarKey) -> Vec<u32> {
    radix_order_generic_by_key(len, key)
}

/// Four-pass stable radix order for an implicit array of native F32 keys.
pub(crate) fn radix_order_f32_by_key(len: usize, key: impl Fn(usize) -> F32Key) -> Vec<u32> {
    radix_order_generic_by_key(len, key)
}

/// Stable exact radix order for either canonical 64-bit scalar keys or native
/// 32-bit F32 keys.
pub(crate) fn radix_order_generic_by_key<K: RadixScalarKey>(
    len: usize,
    key: impl Fn(usize) -> K,
) -> Vec<u32> {
    radix_order_raw(len, K::RADIX_BYTES, |index| key(index).raw_u64())
}

fn radix_order_raw(len: usize, byte_passes: usize, raw: impl Fn(usize) -> u64) -> Vec<u32> {
    assert!(
        len <= u32::MAX as usize,
        "radix order has too many entries for u32 indices"
    );
    assert!(byte_passes <= 8);

    const RADIX: usize = 256;
    let mut from: Vec<u32> = (0..len as u32).collect();
    let mut to = vec![0u32; len];

    for pass in 0..byte_passes {
        let shift = pass * 8;
        let mut counts = [0usize; RADIX];
        for &index in &from {
            let digit = ((raw(index as usize) >> shift) & 0xff) as usize;
            counts[digit] += 1;
        }

        let mut offsets = [0usize; RADIX];
        let mut next = 0usize;
        for (digit, count) in counts.into_iter().enumerate() {
            offsets[digit] = next;
            next += count;
        }

        for &index in &from {
            let digit = ((raw(index as usize) >> shift) & 0xff) as usize;
            to[offsets[digit]] = index;
            offsets[digit] += 1;
        }
        std::mem::swap(&mut from, &mut to);
    }
    from
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparison_order(values: &[ScalarKey]) -> Vec<u32> {
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_unstable_by_key(|&index| (values[index as usize], index));
        order
    }

    fn comparison_f32_order(values: &[F32Key]) -> Vec<u32> {
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_unstable_by_key(|&index| (values[index as usize], index));
        order
    }

    #[test]
    fn integer_counting_orders_match_comparison_order() {
        let u8_values: Vec<_> = [255u16, 0, 2, 2, 17, 1, 255]
            .into_iter()
            .map(ScalarKey::from_u16)
            .collect();
        let u16_values: Vec<_> = [65_535u16, 1, 9_000, 1, 0, 42]
            .into_iter()
            .map(ScalarKey::from_u16)
            .collect();
        assert_eq!(
            sorted_scalar_indices(&u8_values, ScalarPixelType::U8),
            comparison_order(&u8_values)
        );
        assert_eq!(
            sorted_scalar_indices(&u16_values, ScalarPixelType::U16),
            comparison_order(&u16_values)
        );
    }

    #[test]
    fn floating_radix_order_matches_comparison_order() {
        let values: Vec<_> = [
            -1.0,
            f64::MIN_POSITIVE,
            -0.0,
            0.0,
            9.5,
            -1.0e100,
            f64::MAX,
            9.5,
        ]
        .into_iter()
        .map(|value| ScalarKey::from_f64(value).unwrap())
        .collect();
        assert_eq!(
            sorted_scalar_indices(&values, ScalarPixelType::F64),
            comparison_order(&values)
        );
    }

    #[test]
    fn native_f32_radix_order_matches_comparison_order() {
        let values: Vec<_> = [
            -1.0f32,
            f32::MIN_POSITIVE,
            -0.0,
            0.0,
            9.5,
            -1.0e30,
            f32::MAX,
            9.5,
        ]
        .into_iter()
        .map(|value| F32Key::from_f32(value).unwrap())
        .collect();
        assert_eq!(sorted_f32_indices(&values), comparison_f32_order(&values));
    }

    #[test]
    fn implicit_radix_order_matches_comparison_order_with_ties() {
        let values: Vec<_> = [5u16, 1, 5, 2, 1, 9, 2]
            .into_iter()
            .map(ScalarKey::from_u16)
            .collect();
        assert_eq!(
            radix_order_by_key(values.len(), |index| values[index]),
            comparison_order(&values)
        );
    }

    #[test]
    fn native_f32_and_wide_f32_orders_are_identical() {
        let source = [-3.5f32, 2.0, -0.0, 0.0, 2.0, f32::MIN_POSITIVE, -99.0];
        let native: Vec<_> = source
            .iter()
            .copied()
            .map(|value| F32Key::from_f32(value).unwrap())
            .collect();
        let wide: Vec<_> = source
            .iter()
            .copied()
            .map(|value| ScalarKey::from_f32(value).unwrap())
            .collect();
        assert_eq!(
            sorted_f32_indices(&native),
            sorted_scalar_indices(&wide, ScalarPixelType::F32)
        );
    }
}
