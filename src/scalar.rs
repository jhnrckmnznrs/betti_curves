use anyhow::{Result, bail};
use std::fmt;

/// An order-preserving key for a finite IEEE-754 `f64` scalar.
///
/// The unsigned ordering of `ScalarKey` is exactly the numerical ordering of
/// the original finite scalar values. Integer, `f32`, and `f64` TIFF samples
/// can therefore share the same persistence implementation without
/// quantization.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarKey(u64);

impl ScalarKey {
    /// Ordered representation of `0.0`.
    pub const ZERO: Self = Self(0x8000_0000_0000_0000);

    /// Reserved local-H2 marker. This is the ordered encoding of +infinity,
    /// which cannot occur in an accepted scalar image because non-finite
    /// samples are rejected during decoding.
    pub(crate) const OUTSIDE_MARKER: Self = Self(0xfff0_0000_0000_0000);

    pub fn from_f64(mut value: f64) -> Result<Self> {
        if !value.is_finite() {
            bail!("scalar field contains a non-finite value: {value}");
        }

        // Persistence should not distinguish -0.0 from +0.0.
        if value == 0.0 {
            value = 0.0;
        }

        let bits = value.to_bits();
        let ordered = if bits & 0x8000_0000_0000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000_0000_0000
        };

        Ok(Self(ordered))
    }

    pub fn from_f32(value: f32) -> Result<Self> {
        Self::from_f64(f64::from(value))
    }

    pub fn from_u8(value: u8) -> Self {
        Self::from_f64(f64::from(value)).expect("u8 is always finite")
    }

    pub fn from_u16(value: u16) -> Self {
        Self::from_f64(f64::from(value)).expect("u16 is always finite")
    }

    pub fn to_f64(self) -> f64 {
        let bits = if self.0 & 0x8000_0000_0000_0000 != 0 {
            self.0 ^ 0x8000_0000_0000_0000
        } else {
            !self.0
        };

        f64::from_bits(bits)
    }

    pub fn raw(self) -> u64 {
        self.0
    }

    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }

    pub fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

/// A native-width order-preserving key for a finite IEEE-754 `f32` scalar.
///
/// The unsigned ordering of `F32Key` is exactly the numerical ordering of
/// finite `f32` values, with `-0.0` normalized to `+0.0`.  The scalar stream
/// implementation uses this four-byte representation inside F32 slabs so the
/// hottest ordering and union-find state no longer widen every voxel to an
/// eight-byte `ScalarKey`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct F32Key(u32);

impl F32Key {
    /// Reserved local-H2 marker. This is the ordered encoding of +infinity,
    /// which cannot occur in an accepted F32 image because non-finite
    /// samples are rejected during decoding.
    pub(crate) const OUTSIDE_MARKER: Self = Self(0xff80_0000);

    pub fn from_f32(mut value: f32) -> Result<Self> {
        if !value.is_finite() {
            bail!("scalar field contains a non-finite value: {value}");
        }
        if value == 0.0 {
            value = 0.0;
        }

        let bits = value.to_bits();
        let ordered = if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        };
        Ok(Self(ordered))
    }

    pub fn to_f32(self) -> f32 {
        let bits = if self.0 & 0x8000_0000 != 0 {
            self.0 ^ 0x8000_0000
        } else {
            !self.0
        };
        f32::from_bits(bits)
    }

    pub fn raw(self) -> u32 {
        self.0
    }

    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Widen this exact F32 key to the canonical F64-width `ScalarKey` without
    /// a floating-point conversion. IEEE-754 binary32 values are exactly
    /// representable in binary64, so exponent/mantissa expansion is lossless.
    #[inline]
    pub fn to_scalar_key(self) -> ScalarKey {
        let bits32 = if self.0 & 0x8000_0000 != 0 {
            self.0 ^ 0x8000_0000
        } else {
            !self.0
        };
        let sign = u64::from(bits32 >> 31);
        let exponent32 = (bits32 >> 23) & 0xff;
        let fraction32 = bits32 & 0x007f_ffff;

        let (exponent64, fraction64) = if exponent32 == 0 {
            if fraction32 == 0 {
                (0u64, 0u64)
            } else {
                let highest = 31 - fraction32.leading_zeros();
                let leading = 1u32 << highest;
                let exponent64 = u64::from(highest + 874);
                let fraction64 = u64::from(fraction32 - leading) << (52 - highest);
                (exponent64, fraction64)
            }
        } else {
            debug_assert!(exponent32 < 0xff);
            (u64::from(exponent32 + 896), u64::from(fraction32) << 29)
        };

        let bits64 = (sign << 63) | (exponent64 << 52) | fraction64;
        let ordered64 = if bits64 & 0x8000_0000_0000_0000 != 0 {
            !bits64
        } else {
            bits64 ^ 0x8000_0000_0000_0000
        };
        ScalarKey(ordered64)
    }
}

impl fmt::Display for F32Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.to_f32())
    }
}

pub(crate) trait LocalScalarKey: Copy + Ord {
    fn widen(self) -> ScalarKey;
    fn outside_marker() -> Self;

    #[inline]
    fn is_outside_marker(self) -> bool {
        self == Self::outside_marker()
    }
}

impl LocalScalarKey for ScalarKey {
    #[inline]
    fn widen(self) -> ScalarKey {
        debug_assert!(!self.is_outside_marker());
        self
    }

    #[inline]
    fn outside_marker() -> Self {
        Self::OUTSIDE_MARKER
    }
}

impl LocalScalarKey for F32Key {
    #[inline]
    fn widen(self) -> ScalarKey {
        debug_assert!(!self.is_outside_marker());
        self.to_scalar_key()
    }

    #[inline]
    fn outside_marker() -> Self {
        Self::OUTSIDE_MARKER
    }
}

impl fmt::Display for ScalarKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_order_matches_float_order() {
        let values = [-100.5, -1.0, -0.0, 0.0, 0.25, 7.0, 1.0e20];
        let keys: Vec<_> = values
            .iter()
            .copied()
            .map(|value| ScalarKey::from_f64(value).unwrap())
            .collect();

        for pair in keys.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn round_trip_is_exact() {
        for value in [-123.25, -1.0, 0.0, 0.5, 42.0, f64::MAX] {
            let key = ScalarKey::from_f64(value).unwrap();
            assert_eq!(key.to_f64(), value);
        }
    }

    #[test]
    fn local_h2_outside_markers_are_above_every_finite_key() {
        let f32_max = F32Key::from_f32(f32::MAX).unwrap();
        assert!(f32_max < F32Key::OUTSIDE_MARKER);
        assert!(F32Key::OUTSIDE_MARKER.is_outside_marker());

        let f64_max = ScalarKey::from_f64(f64::MAX).unwrap();
        assert!(f64_max < ScalarKey::OUTSIDE_MARKER);
        assert!(ScalarKey::OUTSIDE_MARKER.is_outside_marker());
    }

    #[test]
    fn f32_native_key_order_and_round_trip_are_exact() {
        let values = [-100.5f32, -1.0, -0.0, 0.0, 0.25, 7.0, f32::MAX];
        let keys: Vec<_> = values
            .iter()
            .copied()
            .map(|value| F32Key::from_f32(value).unwrap())
            .collect();
        for pair in keys.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
        for value in values {
            let normalized = if value == 0.0 { 0.0 } else { value };
            assert_eq!(F32Key::from_f32(value).unwrap().to_f32(), normalized);
        }
    }

    #[test]
    fn native_f32_widening_matches_float_conversion_exactly() {
        let fractions = [0u32, 1, 0x0001_2345, 0x007f_ffff];
        for sign in [0u32, 1] {
            for exponent in 0u32..0xff {
                for fraction in fractions {
                    let bits = (sign << 31) | (exponent << 23) | fraction;
                    let value = f32::from_bits(bits);
                    let native = F32Key::from_f32(value).unwrap().to_scalar_key();
                    let legacy = ScalarKey::from_f32(value).unwrap();
                    assert_eq!(native, legacy, "mismatch for f32 bits {bits:#010x}");
                }
            }
        }
    }

    #[test]
    fn native_f32_and_wide_scalar_keys_have_identical_order() {
        let values = [-17.25f32, -1.0, -0.0, 0.0, f32::MIN_POSITIVE, 1.5, f32::MAX];
        for pair in values.windows(2) {
            let a32 = F32Key::from_f32(pair[0]).unwrap();
            let b32 = F32Key::from_f32(pair[1]).unwrap();
            let a64 = ScalarKey::from_f32(pair[0]).unwrap();
            let b64 = ScalarKey::from_f32(pair[1]).unwrap();
            assert_eq!(a32.cmp(&b32), a64.cmp(&b64));
        }
    }
}
