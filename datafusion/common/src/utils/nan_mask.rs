// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Utilities for working with NaN values

use arrow::array::{
    Array, BooleanArray, Datum, Float16Array, Float32Array, Float64Array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::DataType;
#[cfg(feature = "nan_mask_counter")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "nan_mask_counter")]
pub static BUILD_NAN_MASK_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Build a [`BooleanArray`] marking `NaN` values within `arr`.
///
/// For floating point arrays (`Float16`, `Float32`, `Float64`) this returns a
/// boolean array where each entry is `true` if the corresponding value is `NaN`.
/// Null values in the input are propagated to the mask. The implementation uses
/// [`arrow::compute::is_nan`] for `Float32` and `Float64` types for improved
/// performance, falling back to a manual scan for `Float16`. For non-floating
/// types, this returns a mask of all `false` values with no nulls.
pub fn build_nan_mask(arr: &dyn Array) -> BooleanArray {
    #[cfg(feature = "nan_mask_counter")]
    BUILD_NAN_MASK_CALLS.fetch_add(1, Ordering::SeqCst);
    match arr.data_type() {
        DataType::Float16 => {
            // Arrow compute currently lacks native `is_nan` support for `Float16`,
            // so fall back to a manual iteration.
            let arr = arr.as_any().downcast_ref::<Float16Array>().unwrap();
            BooleanArray::from_iter(arr.iter().map(|v| v.map(|x| x.is_nan())))
        }
        DataType::Float32 => {
            let arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            BooleanArray::from_unary(arr, |x| x.is_nan())
        }
        DataType::Float64 => {
            let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            BooleanArray::from_unary(arr, |x| x.is_nan())
        }
        _ => BooleanArray::new(BooleanBuffer::new_unset(arr.len()), None),
    }
}

/// Returns a boolean mask marking `NaN` values within the provided [`Datum`].
///
/// This mask is used to implement IEEE-754 *unordered* semantics for
/// floating-point comparisons: if either side is `NaN`, the comparison
/// should evaluate to `false`. For scalar [`Datum`] values the mask is
/// expanded to `len` entries so it can be combined with array results.
///
/// Nulls in the input propagate as nulls in the mask, allowing later
/// boolean operators to preserve SQL's three-valued logic.
pub fn mask_datum_nan(d: &dyn Datum, len: usize) -> BooleanArray {
    let (array, is_scalar) = d.get();
    let mask = build_nan_mask(array);
    if is_scalar && len != array.len() {
        if mask.is_null(0) {
            BooleanArray::new(
                BooleanBuffer::new_unset(len),
                Some(NullBuffer::new_null(len)),
            )
        } else {
            let buf = if mask.value(0) {
                BooleanBuffer::new_set(len)
            } else {
                BooleanBuffer::new_unset(len)
            };
            BooleanArray::new(buf, None)
        }
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::ScalarValue;
    use arrow::array::{Float32Array, Int32Array};

    #[test]
    fn test_build_nan_mask_float() {
        let arr = Float32Array::from(vec![f32::NAN, 1.0, f32::NAN, f32::INFINITY, 0.0]);
        let mask = build_nan_mask(&arr);
        assert_eq!(mask.len(), 5);
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));
        assert!(!mask.value(3));
        assert!(!mask.value(4));
    }

    #[test]
    fn test_build_nan_mask_non_float() {
        let arr = Int32Array::from(vec![1, 2, 3]);
        let mask = build_nan_mask(&arr);
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.true_count(), 0);
    }

    #[test]
    fn test_mask_datum_nan_scalar() {
        let scalar = ScalarValue::Float32(Some(f32::NAN)).to_scalar().unwrap();
        let mask = mask_datum_nan(&scalar, 3);
        assert_eq!(mask.len(), 3);
        for i in 0..3 {
            assert!(mask.value(i));
        }

        let null_scalar = ScalarValue::Float32(None).to_scalar().unwrap();
        let mask = mask_datum_nan(&null_scalar, 3);
        assert_eq!(mask.null_count(), 3);
    }
}
