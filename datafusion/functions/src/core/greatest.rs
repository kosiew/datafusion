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

use crate::core::greatest_least_utils::GreatestLeastOperator;
use arrow::array::{make_comparator, Array, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{kernels::cmp, SortOptions};
use arrow::datatypes::DataType;
use datafusion_common::{internal_err, Result, ScalarValue};
use datafusion_doc::Documentation;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion_macros::user_doc;
use std::any::Any;

const SORT_OPTIONS: SortOptions = SortOptions {
    // We want greatest first
    descending: false,

    // NULL will be less than any other value
    nulls_first: true,
};

#[user_doc(
    doc_section(label = "Conditional Functions"),
    description = "Returns the greatest value in a list of expressions. Returns _null_ if all expressions are _null_.",
    syntax_example = "greatest(expression1[, ..., expression_n])",
    sql_example = r#"```sql
> select greatest(4, 7, 5);
+---------------------------+
| greatest(4,7,5)           |
+---------------------------+
| 7                         |
+---------------------------+
```"#,
    argument(
        name = "expression1, expression_n",
        description = "Expressions to compare and return the greatest value.. Can be a constant, column, or function, and any combination of arithmetic operators. Pass as many expression arguments as necessary."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GreatestFunc {
    signature: Signature,
}

impl Default for GreatestFunc {
    fn default() -> Self {
        GreatestFunc::new()
    }
}

impl GreatestFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl GreatestLeastOperator for GreatestFunc {
    const NAME: &'static str = "greatest";

    fn keep_scalar<'a>(
        lhs: &'a ScalarValue,
        rhs: &'a ScalarValue,
    ) -> Result<&'a ScalarValue> {
        if lhs.is_nan() {
            return Ok(lhs);
        }

        if rhs.is_nan() {
            return Ok(rhs);
        }

        if !lhs.data_type().is_nested() {
            return if lhs >= rhs { Ok(lhs) } else { Ok(rhs) };
        }

        // If complex type we can't compare directly as we want null values to be smaller
        let cmp = make_comparator(
            lhs.to_array()?.as_ref(),
            rhs.to_array()?.as_ref(),
            SORT_OPTIONS,
        )?;

        if cmp(0, 0).is_ge() {
            Ok(lhs)
        } else {
            Ok(rhs)
        }
    }

    /// Return boolean array where `arr[i] = lhs[i] >= rhs[i]` for all i, where `arr` is the result array
    /// Nulls are always considered smaller than any other value
    fn get_indexes_to_keep(lhs: &dyn Array, rhs: &dyn Array) -> Result<BooleanArray> {
        let lhs_nan = lhs
            .data_type()
            .is_floating()
            .then(|| datafusion_common::utils::nan_mask::build_nan_mask(lhs));
        let rhs_nan = rhs
            .data_type()
            .is_floating()
            .then(|| datafusion_common::utils::nan_mask::build_nan_mask(rhs));

        // Fast path:
        // If both arrays are not nested, have the same length, no nulls and no NaNs, we can use the faster vectorized kernel
        if !lhs.data_type().is_nested()
            && lhs.logical_null_count() == 0
            && rhs.logical_null_count() == 0
            && lhs_nan.as_ref().map_or(true, |a| a.true_count() == 0)
            && rhs_nan.as_ref().map_or(true, |a| a.true_count() == 0)
        {
            return cmp::gt_eq(&lhs, &rhs).map_err(|e| e.into());
        }

        let cmp = make_comparator(lhs, rhs, SORT_OPTIONS)?;

        if lhs.len() != rhs.len() {
            return internal_err!(
                "All arrays should have the same length for greatest comparison"
            );
        }

        let values = BooleanBuffer::collect_bool(lhs.len(), |i| {
            if lhs_nan.as_ref().map_or(false, |a| a.value(i)) {
                true
            } else if rhs_nan.as_ref().map_or(false, |a| a.value(i)) {
                false
            } else {
                cmp(i, i).is_ge()
            }
        });

        // No nulls as we only want to keep the values that are larger, its either true or false
        Ok(BooleanArray::new(values, None))
    }
}

impl ScalarUDFImpl for GreatestFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "greatest"
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
    use crate::core;
    use arrow::array::{Float16Array, Float64Array, Int32Array};
    use arrow::datatypes::DataType;
    use datafusion_common::{utils::nan_mask::BUILD_NAN_MASK_CALLS, ScalarValue};
    use datafusion_expr::{ColumnarValue, ScalarUDFImpl};
    use half::f16;
    use std::sync::{atomic::Ordering, Arc};

    #[test]
    fn test_greatest_return_types_without_common_supertype_in_arg_type() {
        let greatest = core::greatest::GreatestFunc::new();
        let return_type = greatest
            .coerce_types(&[DataType::Decimal128(10, 3), DataType::Decimal128(10, 4)])
            .unwrap();
        assert_eq!(
            return_type,
            vec![DataType::Decimal128(11, 4), DataType::Decimal128(11, 4)]
        );
    }

    #[test]
    fn test_greatest_nan_scalar() {
        let result = core::greatest_least_utils::execute_conditional::<
            core::greatest::GreatestFunc,
        >(&[
            ColumnarValue::Scalar(ScalarValue::Float64(Some(f64::NAN))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
        ])
        .unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) => assert!(v.is_nan()),
            _ => panic!("expected scalar"),
        }

        let result = core::greatest_least_utils::execute_conditional::<
            core::greatest::GreatestFunc,
        >(&[
            ColumnarValue::Scalar(ScalarValue::Float16(Some(f16::NAN))),
            ColumnarValue::Scalar(ScalarValue::Float16(Some(f16::from_f32(1.0)))),
        ])
        .unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Float16(Some(v))) => assert!(v.is_nan()),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn test_greatest_nan_array() {
        let lhs = Float64Array::from(vec![f64::NAN, 1.0]);
        let rhs = Float64Array::from(vec![1.0, f64::NAN]);
        let result = core::greatest_least_utils::execute_conditional::<
            core::greatest::GreatestFunc,
        >(&[
            ColumnarValue::Array(Arc::new(lhs)),
            ColumnarValue::Array(Arc::new(rhs)),
        ])
        .unwrap();

        match result {
            ColumnarValue::Array(arr) => {
                let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                assert!(arr.value(0).is_nan());
                assert!(arr.value(1).is_nan());
            }
            _ => panic!("expected array"),
        }

        let lhs = Float16Array::from(vec![f16::NAN, f16::from_f32(1.0)]);
        let rhs = Float16Array::from(vec![f16::from_f32(1.0), f16::NAN]);
        let result = core::greatest_least_utils::execute_conditional::<
            core::greatest::GreatestFunc,
        >(&[
            ColumnarValue::Array(Arc::new(lhs)),
            ColumnarValue::Array(Arc::new(rhs)),
        ])
        .unwrap();

        match result {
            ColumnarValue::Array(arr) => {
                let arr = arr.as_any().downcast_ref::<Float16Array>().unwrap();
                assert!(arr.value(0).is_nan());
                assert!(arr.value(1).is_nan());
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_greatest_non_float_bypasses_nan_mask() {
        let lhs = Int32Array::from(vec![1, 3]);
        let rhs = Int32Array::from(vec![2, 1]);
        BUILD_NAN_MASK_CALLS.store(0, Ordering::SeqCst);
        let result = core::greatest_least_utils::execute_conditional::<
            core::greatest::GreatestFunc,
        >(&[
            ColumnarValue::Array(Arc::new(lhs)),
            ColumnarValue::Array(Arc::new(rhs)),
        ])
        .unwrap();
        let int_calls = BUILD_NAN_MASK_CALLS.load(Ordering::SeqCst);

        let arr = result.into_array(2).unwrap();
        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(arr.value(0), 2);
        assert_eq!(arr.value(1), 3);

        let lhsf = Float64Array::from(vec![1.0, 3.0]);
        let rhsf = Float64Array::from(vec![2.0, f64::NAN]);
        BUILD_NAN_MASK_CALLS.store(0, Ordering::SeqCst);
        core::greatest_least_utils::execute_conditional::<core::greatest::GreatestFunc>(
            &[
                ColumnarValue::Array(Arc::new(lhsf)),
                ColumnarValue::Array(Arc::new(rhsf)),
            ],
        )
        .unwrap();
        let float_calls = BUILD_NAN_MASK_CALLS.load(Ordering::SeqCst);

        assert!(int_calls < float_calls);
    }
}
