#!/usr/bin/env python3
"""Exhaustive model check for the v20 compact U16 persistence key."""
from __future__ import annotations

import struct


def scalar_key_from_f64(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits ^ (1 << 63)


def main() -> int:
    prev_native = -1
    prev_wide = -1
    for value in range(65_536):
        native = value
        wide = scalar_key_from_f64(float(value))
        if native <= prev_native:
            raise AssertionError(f"native U16 order failed at {value}")
        if wide <= prev_wide:
            raise AssertionError(f"wide ScalarKey order failed at {value}")
        prev_native = native
        prev_wide = wide
    outside = 65_536
    assert outside > prev_native
    print("PASS: all 65536 U16 values preserve exact order; Outside marker is disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
