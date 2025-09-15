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

use crate::core::greatest_least_utils::{
    build_nan_masks, compare_with_nan, GreatestLeastOperator,
};
use arrow::array::{make_comparator, Array, BooleanArray};
use arrow::compute::kernels::{boolean::or, cmp};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_doc::Documentation;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion_macros::user_doc;
use std::any::Any;

const SORT_OPTIONS: SortOptions = SortOptions {
    // Having the smallest result first
    descending: false,

    // NULL will be greater than any other value
    nulls_first: false,
};

#[user_doc(
    doc_section(label = "Conditional Functions"),
    description = "Returns the smallest value in a list of expressions. Returns _null_ if all expressions are _null_.",
    syntax_example = "least(expression1[, ..., expression_n])",
    sql_example = r#"```sql
> select least(4, 7, 5);
+---------------------------+
| least(4,7,5)              |
+---------------------------+
| 4                         |
+---------------------------+
```"#,
    argument(
        name = "expression1, expression_n",
        description = "Expressions to compare and return the smallest value. Can be a constant, column, or function, and any combination of arithmetic operators. Pass as many expression arguments as necessary."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LeastFunc {
    signature: Signature,
}

impl Default for LeastFunc {
    fn default() -> Self {
        LeastFunc::new()
    }
}

impl LeastFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl GreatestLeastOperator for LeastFunc {
    const NAME: &'static str = "least";

    fn keep_scalar<'a>(
        lhs: &'a ScalarValue,
        rhs: &'a ScalarValue,
    ) -> Result<&'a ScalarValue> {
        // Manual checking for nulls as:
        // 1. If we're going to use <=, in Rust None is smaller than Some(T), which we don't want
        // 2. And we can't use make_comparator as it has no natural order (Arrow error)
        if lhs.is_null() {
            return Ok(rhs);
        }

        if rhs.is_null() {
            return Ok(lhs);
        }

        if lhs.is_nan() {
            if rhs.is_nan() {
                return Ok(lhs);
            }
            return Ok(rhs);
        }

        if rhs.is_nan() {
            return Ok(lhs);
        }

        if !lhs.data_type().is_nested() {
            return if lhs <= rhs { Ok(lhs) } else { Ok(rhs) };
        }

        // Not using <= as in Rust None is smaller than Some(T)

        // If complex type we can't compare directly as we want null values to be larger
        let cmp = make_comparator(
            lhs.to_array()?.as_ref(),
            rhs.to_array()?.as_ref(),
            SORT_OPTIONS,
        )?;

        if cmp(0, 0).is_le() {
            Ok(lhs)
        } else {
            Ok(rhs)
        }
    }

    /// Return boolean array where `arr[i] = lhs[i] <= rhs[i]` for all i, where `arr` is the result array
    /// Nulls are always considered larger than any other value
    fn get_indexes_to_keep(lhs: &dyn Array, rhs: &dyn Array) -> Result<BooleanArray> {
        // Fast path:
        // If both arrays are not nested, have the same length and no nulls, we can use the faster vectorized kernel
        // - If both arrays are not nested: Nested types, such as lists, are not supported as the null semantics are not well-defined.
        // - both array does not have any nulls: cmp::lt_eq will return null if any of the input is null while we want to return false in that case
        if !lhs.data_type().is_nested()
            && lhs.logical_null_count() == 0
            && rhs.logical_null_count() == 0
        {
            let mut result = cmp::lt_eq(&lhs, &rhs)
                .map_err(datafusion_common::DataFusionError::from)?;
            if rhs.data_type().is_floating() {
                let rhs_nan = datafusion_common::utils::nan_mask::build_nan_mask(rhs);
                result = or(&result, &rhs_nan)?;
            }
            return Ok(result);
        }

        let (lhs_nan, rhs_nan) = build_nan_masks(lhs, rhs);

        let cmp = make_comparator(lhs, rhs, SORT_OPTIONS)?;

        compare_with_nan(
            lhs,
            rhs,
            Self::NAME,
            lhs_nan.as_ref(),
            rhs_nan.as_ref(),
            |i, j| cmp(i, j).is_le(),
            false,
            true,
        )
    }
}

impl ScalarUDFImpl for LeastFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "least"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        super::greatest_least_utils::execute_conditional::<Self>(&args.args)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let coerced_type =
            super::greatest_least_utils::find_coerced_type::<Self>(arg_types)?;

        Ok(vec![coerced_type; arg_types.len()])
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod test {
    use crate::core::greatest_least_utils::GreatestLeastOperator;
    use crate::core::least::LeastFunc;
    use arrow::array::{Float16Array, Float64Array, Int32Array};
    use arrow::datatypes::DataType;
    use datafusion_common::{
        cast::{as_float16_array, as_float64_array},
        utils::nan_mask::BUILD_NAN_MASK_CALLS,
        ScalarValue,
    };
    use datafusion_expr::{ColumnarValue, ScalarUDFImpl};
    use half::f16;
    use std::sync::{atomic::Ordering, Arc};

    #[test]
    fn test_least_return_types_without_common_supertype_in_arg_type() {
        let least = LeastFunc::new();
        let return_type = least
            .coerce_types(&[DataType::Decimal128(10, 3), DataType::Decimal128(10, 4)])
            .unwrap();
        assert_eq!(
            return_type,
            vec![DataType::Decimal128(11, 4), DataType::Decimal128(11, 4)]
        );
    }

    #[test]
    fn test_least_nan_scalar() {
        let nan = ScalarValue::Float64(Some(f64::NAN));
        let val = ScalarValue::Float64(Some(1.0));

        let result = LeastFunc::keep_scalar(&nan, &val).unwrap();
        assert_eq!(result, &val);

        let result = LeastFunc::keep_scalar(&val, &nan).unwrap();
        assert_eq!(result, &val);

        let nan = ScalarValue::Float16(Some(f16::NAN));
        let val = ScalarValue::Float16(Some(f16::from_f32(1.0)));

        let result = LeastFunc::keep_scalar(&nan, &val).unwrap();
        assert_eq!(result, &val);

        let result = LeastFunc::keep_scalar(&val, &nan).unwrap();
        assert_eq!(result, &val);
    }

    #[test]
    fn test_least_nan_array() {
        let args = vec![
            ColumnarValue::Array(Arc::new(Float64Array::from(vec![f64::NAN, 5.0]))),
            ColumnarValue::Array(Arc::new(Float64Array::from(vec![1.0, f64::NAN]))),
        ];

        let result =
            crate::core::greatest_least_utils::execute_conditional::<LeastFunc>(&args)
                .unwrap();
        let array_ref = result.into_array(2).unwrap();
        let array = as_float64_array(&array_ref).expect("failed to convert");
        assert_eq!(array.value(0), 1.0);
        assert_eq!(array.value(1), 5.0);

        let args = vec![
            ColumnarValue::Array(Arc::new(Float16Array::from(vec![
                f16::NAN,
                f16::from_f32(5.0),
            ]))),
            ColumnarValue::Array(Arc::new(Float16Array::from(vec![
                f16::from_f32(1.0),
                f16::NAN,
            ]))),
        ];

        let result =
            crate::core::greatest_least_utils::execute_conditional::<LeastFunc>(&args)
                .unwrap();
        let array_ref = result.into_array(2).unwrap();
        let array = as_float16_array(&array_ref).expect("failed to convert");
        assert_eq!(array.value(0), f16::from_f32(1.0));
        assert_eq!(array.value(1), f16::from_f32(5.0));
    }

    #[test]
    fn test_least_non_float_bypasses_nan_mask() {
        let lhs = Int32Array::from(vec![1, 4]);
        let rhs = Int32Array::from(vec![2, 1]);
        BUILD_NAN_MASK_CALLS.store(0, Ordering::SeqCst);
        let result =
            crate::core::greatest_least_utils::execute_conditional::<LeastFunc>(&[
                ColumnarValue::Array(Arc::new(lhs)),
                ColumnarValue::Array(Arc::new(rhs)),
            ])
            .unwrap();
        let int_calls = BUILD_NAN_MASK_CALLS.load(Ordering::SeqCst);

        let arr = result.into_array(2).unwrap();
        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(arr.value(0), 1);
        assert_eq!(arr.value(1), 1);

        let lhsf = Float64Array::from(vec![1.0, 4.0]);
        let rhsf = Float64Array::from(vec![2.0, f64::NAN]);
        BUILD_NAN_MASK_CALLS.store(0, Ordering::SeqCst);
        crate::core::greatest_least_utils::execute_conditional::<LeastFunc>(&[
            ColumnarValue::Array(Arc::new(lhsf)),
            ColumnarValue::Array(Arc::new(rhsf)),
        ])
        .unwrap();
        let float_calls = BUILD_NAN_MASK_CALLS.load(Ordering::SeqCst);

        assert!(int_calls < float_calls);
    }
}
