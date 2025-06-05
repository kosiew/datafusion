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

//! [`ScalarUDFImpl`] definition for array_sum function.

use crate::utils::make_scalar_function;
use arrow::array::{
    Array, ArrayRef, Decimal128Array, Decimal256Array, GenericListArray,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeListArray, ListArray,
    OffsetSizeTrait, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    Float32Array, Float64Array, AsArray,
};
use arrow::compute;
use arrow::datatypes::DataType::{self, Decimal128, Decimal256, LargeList, List};
use datafusion_common::cast::{
    as_generic_list_array, as_int16_array, as_int32_array, as_int64_array,
    as_int8_array, as_uint16_array, as_uint32_array, as_uint64_array,
    as_uint8_array, as_float32_array, as_float64_array,
    as_decimal128_array, as_decimal256_array,
};
use datafusion_common::utils::take_function_args;
use datafusion_common::{exec_err, plan_err, Result, ScalarValue};
use datafusion_doc::Documentation;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;
use std::any::Any;

make_udf_expr_and_func!(
    ArraySum,
    array_sum,
    array,
    "returns the sum of the array elements.",
    array_sum_udf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the sum of all values in the array.",
    syntax_example = "array_sum(array)",
    sql_example = r#"```sql
> select array_sum([1, 2, 3]);
+---------------------------+
| array_sum(List([1,2,3]))  |
+---------------------------+
| 6                         |
+---------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    )
)]
#[derive(Debug)]
pub struct ArraySum {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArraySum {
    fn default() -> Self {
        Self::new()
    }
}

impl ArraySum {
    pub fn new() -> Self {
        Self {
            signature: Signature::array(Volatility::Immutable),
            aliases: vec!["list_sum".to_string()],
        }
    }
}

impl ScalarUDFImpl for ArraySum {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_sum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [array] = take_function_args(self.name(), arg_types)?;
        match array {
            List(field) | LargeList(field) => Ok(field.data_type().clone()),
            arg_type => plan_err!("{} does not support type {arg_type}", self.name()),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(array_sum_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

pub fn array_sum_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array] = take_function_args("array_sum", args)?;
    match array.data_type() {
        List(_) => general_array_sum(as_generic_list_array::<i32>(array)?),
        LargeList(_) => general_array_sum(as_generic_list_array::<i64>(array)?),
        arg_type => exec_err!("array_sum does not support type: {arg_type}"),
    }
}

fn general_array_sum<O: OffsetSizeTrait>(array: &GenericListArray<O>) -> Result<ArrayRef> {
    let null_value = ScalarValue::try_from(array.value_type())?;
    let result_vec: Vec<ScalarValue> = array
        .iter()
        .map(|arr| match arr {
            Some(arr) => sum_values(arr.as_ref()),
            None => Ok(null_value.clone()),
        })
        .collect::<Result<Vec<_>>>()?;
    ScalarValue::iter_to_array(result_vec)
}

fn sum_values(array: &dyn Array) -> Result<ScalarValue> {
    use DataType::*;
    match array.data_type() {
        Int8 => Ok(ScalarValue::Int8(compute::sum(as_int8_array(array))?)),
        Int16 => Ok(ScalarValue::Int16(compute::sum(as_int16_array(array))?)),
        Int32 => Ok(ScalarValue::Int32(compute::sum(as_int32_array(array))?)),
        Int64 => Ok(ScalarValue::Int64(compute::sum(as_int64_array(array))?)),
        UInt8 => Ok(ScalarValue::UInt8(compute::sum(as_uint8_array(array))?)),
        UInt16 => Ok(ScalarValue::UInt16(compute::sum(as_uint16_array(array))?)),
        UInt32 => Ok(ScalarValue::UInt32(compute::sum(as_uint32_array(array))?)),
        UInt64 => Ok(ScalarValue::UInt64(compute::sum(as_uint64_array(array))?)),
        Float32 => Ok(ScalarValue::Float32(compute::sum(as_float32_array(array))?)),
        Float64 => Ok(ScalarValue::Float64(compute::sum(as_float64_array(array))?)),
        Decimal128(precision, scale) => Ok(ScalarValue::Decimal128(
            compute::sum(as_decimal128_array(array)?),
            *precision,
            *scale,
        )),
        Decimal256(precision, scale) => Ok(ScalarValue::Decimal256(
            compute::sum(as_decimal256_array(array)?),
            *precision,
            *scale,
        )),
        data_type => exec_err!("array_sum does not support inner type {data_type}"),
    }
}
