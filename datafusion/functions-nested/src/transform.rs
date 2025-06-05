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

//! [`ScalarUDFImpl`] definitions for array_transform and array_reduce functions.

use std::any::Any;
use std::sync::{Arc, OnceLock};

use arrow::array::{Int64Array, Int64Builder, ListArray};
use arrow_array::{Array, ArrayRef, GenericListArray, OffsetSizeTrait};
use arrow_schema::{DataType, Field};
use datafusion_common::cast::{as_int64_array, as_large_list_array, as_list_array};
use datafusion_common::{exec_err, Result, ScalarValue};
use datafusion_expr::scalar_doc_sections::DOC_SECTION_ARRAY;
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarUDFImpl, Signature, Volatility,
};

use crate::utils::make_scalar_function;

make_udf_expr_and_func!(
    ArrayTransform,
    array_transform,
    array func,
    "applies a scalar function to each element of the array.",
    array_transform_udf
);
make_udf_expr_and_func!(
    ArrayReduce,
    array_reduce,
    array func,
    "reduces array elements using the specified aggregate function.",
    array_reduce_udf
);

#[derive(Debug)]
pub struct ArrayTransform {
    signature: Signature,
    aliases: Vec<String>,
}

impl ArrayTransform {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: vec![String::from("list_transform")],
        }
    }
}

impl Default for ArrayTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ArrayTransform {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_transform"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(array_transform_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(get_array_transform_doc())
    }
}

static DOC_TRANSFORM: OnceLock<Documentation> = OnceLock::new();

fn get_array_transform_doc() -> &'static Documentation {
    DOC_TRANSFORM.get_or_init(|| {
        Documentation::builder()
            .with_doc_section(DOC_SECTION_ARRAY)
            .with_description(
                "Applies a scalar function to each element of the array.",
            )
            .with_syntax_example("array_transform(array, func)")
            .with_sql_example(
                r#"```sql
> select array_transform([1, -2, 3], 'abs');
+---------------------------------------------+
| array_transform(List([1,-2,3]),Utf8("abs")) |
+---------------------------------------------+
| [1, 2, 3]                                   |
+---------------------------------------------+
```"#,
            )
            .with_argument(
                "array",
                "Array expression. Can be a constant, column, or function, and any combination of array operators.",
            )
            .with_argument(
                "func",
                "Name of a scalar function (e.g. 'abs').",
            )
            .build()
            .unwrap()
    })
}

#[derive(Debug)]
pub struct ArrayReduce {
    signature: Signature,
    aliases: Vec<String>,
}

impl ArrayReduce {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: vec![String::from("list_reduce")],
        }
    }
}

impl Default for ArrayReduce {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ArrayReduce {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_reduce"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(array_reduce_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(get_array_reduce_doc())
    }
}

static DOC_REDUCE: OnceLock<Documentation> = OnceLock::new();

fn get_array_reduce_doc() -> &'static Documentation {
    DOC_REDUCE.get_or_init(|| {
        Documentation::builder()
            .with_doc_section(DOC_SECTION_ARRAY)
            .with_description(
                "Aggregates the array elements using the specified aggregate function.",
            )
            .with_syntax_example("array_reduce(array, func)")
            .with_sql_example(
                r#"```sql
> select array_reduce([1,2,3], 'sum');
+------------------------------------------+
| array_reduce(List([1,2,3]),Utf8("sum")) |
+------------------------------------------+
| 6                                        |
+------------------------------------------+
```"#,
            )
            .with_argument(
                "array",
                "Array expression. Can be a constant, column, or function, and any combination of array operators.",
            )
            .with_argument(
                "func",
                "Name of an aggregate function (e.g. 'sum').",
            )
            .build()
            .unwrap()
    })
}

fn array_transform_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return exec_err!("array_transform expects two arguments");
    }
    let func_name = ScalarValue::try_from_array(&args[1], 0)?;
    let func_name = match func_name {
        ScalarValue::Utf8(Some(s)) => s,
        ScalarValue::LargeUtf8(Some(s)) => s,
        _ => return exec_err!("function name must be a string"),
    };
    match &args[0].data_type() {
        DataType::List(_) => {
            let array = as_list_array(&args[0])?;
            general_array_transform::<i32>(array, &func_name)
        }
        DataType::LargeList(_) => {
            let array = as_large_list_array(&args[0])?;
            general_array_transform::<i64>(array, &func_name)
        }
        dt => exec_err!("array_transform does not support type '{dt:?}'"),
    }
}

fn general_array_transform<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    func: &str,
) -> Result<ArrayRef> {
    let values = array.values();
    let transformed_values = apply_scalar_function(values.clone(), func)?;
    Ok(Arc::new(GenericListArray::<O>::try_new(
        Arc::new(Field::new("item", transformed_values.data_type().clone(), true)),
        array.offsets().clone(),
        transformed_values,
        array.nulls().cloned(),
    )?))
}

fn apply_scalar_function(values: ArrayRef, func: &str) -> Result<ArrayRef> {
    use datafusion_functions::math;
    let udf = match func.to_ascii_lowercase().as_str() {
        "abs" => math::abs(),
        _ => return exec_err!("unsupported function '{func}'"),
    };
    let res = udf.invoke(&[ColumnarValue::Array(values.clone())])?;
    res.into_array(values.len())
}

fn array_reduce_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 2 {
        return exec_err!("array_reduce expects two arguments");
    }
    let func_name = ScalarValue::try_from_array(&args[1], 0)?;
    let func_name = match func_name {
        ScalarValue::Utf8(Some(s)) => s,
        ScalarValue::LargeUtf8(Some(s)) => s,
        _ => return exec_err!("function name must be a string"),
    };
    match &args[0].data_type() {
        DataType::List(_) => {
            let array = as_list_array(&args[0])?;
            general_array_reduce::<i32>(array, &func_name)
        }
        DataType::LargeList(_) => {
            let array = as_large_list_array(&args[0])?;
            general_array_reduce::<i64>(array, &func_name)
        }
        dt => exec_err!("array_reduce does not support type '{dt:?}'"),
    }
}

fn general_array_reduce<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    func: &str,
) -> Result<ArrayRef> {
    match func.to_ascii_lowercase().as_str() {
        "sum" => array_reduce_sum(array),
        _ => exec_err!("unsupported aggregate '{func}'"),
    }
}

fn array_reduce_sum<O: OffsetSizeTrait>(array: &GenericListArray<O>) -> Result<ArrayRef> {
    let values = as_int64_array(array.values())?;
    let mut builder = Int64Builder::with_capacity(array.len());
    for (i, window) in array.offsets().windows(2).enumerate() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        let start = window[0].to_usize().unwrap();
        let end = window[1].to_usize().unwrap();
        let mut acc = 0i64;
        let mut has_value = false;
        for idx in start..end {
            if values.is_valid(idx) {
                acc += values.value(idx);
                has_value = true;
            }
        }
        if has_value {
            builder.append_value(acc);
        } else {
            builder.append_null();
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Int64Type;

    #[test]
    fn test_array_transform_abs() -> Result<()> {
        let array = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(-1), Some(2), Some(-3)]),
        ]));
        let result = array_transform_udf().invoke(&[
            ColumnarValue::Array(array as ArrayRef),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abs".to_string()))),
        ])?;
        let result = result.into_array(1)?;
        let expected = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
        ]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_array_reduce_sum() -> Result<()> {
        let array = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
        ]));
        let result = array_reduce_udf().invoke(&[
            ColumnarValue::Array(array as ArrayRef),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("sum".to_string()))),
        ])?;
        let result = result.into_array(1)?;
        let expected = Int64Array::from(vec![Some(6)]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }
}

