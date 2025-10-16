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

//! [`ScalarUDFImpl`] definitions for array_has, array_has_all and array_has_any functions.

use ahash::RandomState;
use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, BinaryArray, BinaryViewArray,
    BooleanArray, Date32Array, Date64Array, Datum, Decimal128Array,
    DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray,
    DurationSecondArray, FixedSizeBinaryArray, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeStringArray,
    NullBufferBuilder, Scalar, StringArray, StringViewArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::row::{RowConverter, Rows, SortField};
use arrow::util::bit_iterator::BitIndexIterator;
use datafusion_common::cast::{as_fixed_size_list_array, as_generic_list_array};
use datafusion_common::hash_utils::HashValue;
use datafusion_common::utils::take_function_args;
use datafusion_common::{exec_err, DataFusionError, HashMap, Result, ScalarValue};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::ExprSimplifyResult;
use datafusion_expr::{
    in_list, ColumnarValue, Documentation, Expr, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::datum::compare_with_eq;
use hashbrown::hash_map::RawEntryMut;
use itertools::Itertools;

use crate::make_array::make_array_udf;
use crate::utils::make_scalar_function;

use std::any::Any;
use std::sync::Arc;

// Create static instances of ScalarUDFs for each function
make_udf_expr_and_func!(ArrayHas,
    array_has,
    haystack_array element, // arg names
    "returns true, if the element appears in the first array, otherwise false.", // doc
    array_has_udf // internal function name
);
make_udf_expr_and_func!(ArrayHasAll,
    array_has_all,
    haystack_array needle_array, // arg names
    "returns true if each element of the second array appears in the first array; otherwise, it returns false.", // doc
    array_has_all_udf // internal function name
);
make_udf_expr_and_func!(ArrayHasAny,
    array_has_any,
    haystack_array needle_array, // arg names
    "returns true if at least one element of the second array appears in the first array; otherwise, it returns false.", // doc
    array_has_any_udf // internal function name
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns true if the array contains the element.",
    syntax_example = "array_has(array, element)",
    sql_example = r#"```sql
> select array_has([1, 2, 3], 2);
+-----------------------------+
| array_has(List([1,2,3]), 2) |
+-----------------------------+
| true                        |
+-----------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(
        name = "element",
        description = "Scalar or Array expression. Can be a constant, column, or function, and any combination of array operators."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayHas {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHas {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHas {
    pub fn new() -> Self {
        Self {
            signature: Signature::array_and_element(Volatility::Immutable),
            aliases: vec![
                String::from("list_has"),
                String::from("array_contains"),
                String::from("list_contains"),
            ],
        }
    }
}

impl ScalarUDFImpl for ArrayHas {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn simplify(
        &self,
        mut args: Vec<Expr>,
        _info: &dyn datafusion_expr::simplify::SimplifyInfo,
    ) -> Result<ExprSimplifyResult> {
        let [haystack, needle] = take_function_args(self.name(), &mut args)?;

        // if the haystack is a constant list, we can use an inlist expression which is more
        // efficient because the haystack is not varying per-row
        match haystack {
            Expr::Literal(
                // FixedSizeList gets coerced to List
                scalar @ ScalarValue::List(_) | scalar @ ScalarValue::LargeList(_),
                _,
            ) => {
                let array = scalar.to_array().unwrap(); // guarantee of ScalarValue
                if let Ok(scalar_values) =
                    ScalarValue::convert_array_to_scalar_vec(&array)
                {
                    assert_eq!(scalar_values.len(), 1);
                    let list = scalar_values
                        .into_iter()
                        .flatten()
                        .map(|v| Expr::Literal(v, None))
                        .collect();

                    return Ok(ExprSimplifyResult::Simplified(in_list(
                        std::mem::take(needle),
                        list,
                        false,
                    )));
                }
            }
            Expr::ScalarFunction(ScalarFunction { func, args })
                if func == &make_array_udf() =>
            {
                // make_array has a static set of arguments, so we can pull the arguments out from it
                return Ok(ExprSimplifyResult::Simplified(in_list(
                    std::mem::take(needle),
                    std::mem::take(args),
                    false,
                )));
            }
            _ => {}
        };
        Ok(ExprSimplifyResult::Original(args))
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let [first_arg, second_arg] = take_function_args(self.name(), &args.args)?;
        match &second_arg {
            ColumnarValue::Array(array_needle) => {
                // the needle is already an array, convert the haystack to an array of the same length
                let haystack = first_arg.to_array(array_needle.len())?;
                let array = array_has_inner_for_array(&haystack, array_needle)?;
                Ok(ColumnarValue::Array(array))
            }
            ColumnarValue::Scalar(scalar_needle) => {
                // Always return null if the second argument is null
                // i.e. array_has(array, null) -> null
                if scalar_needle.is_null() {
                    return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)));
                }

                // since the needle is a scalar, convert it to an array of size 1
                let haystack = first_arg.to_array(1)?;
                let needle = scalar_needle.to_array_of_size(1)?;
                let needle = Scalar::new(needle);
                let array = array_has_inner_for_scalar(&haystack, &needle)?;
                if let ColumnarValue::Scalar(_) = &first_arg {
                    // If both inputs are scalar, keeps output as scalar
                    let scalar_value = ScalarValue::try_from_array(&array, 0)?;
                    Ok(ColumnarValue::Scalar(scalar_value))
                } else {
                    Ok(ColumnarValue::Array(array))
                }
            }
        }
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn array_has_inner_for_scalar(
    haystack: &ArrayRef,
    needle: &dyn Datum,
) -> Result<ArrayRef> {
    let haystack = haystack.as_ref().try_into()?;
    array_has_dispatch_for_scalar(haystack, needle)
}

fn array_has_inner_for_array(haystack: &ArrayRef, needle: &ArrayRef) -> Result<ArrayRef> {
    let haystack = haystack.as_ref().try_into()?;
    array_has_dispatch_for_array(haystack, needle)
}

enum ArrayWrapper<'a> {
    FixedSizeList(&'a arrow::array::FixedSizeListArray),
    List(&'a arrow::array::GenericListArray<i32>),
    LargeList(&'a arrow::array::GenericListArray<i64>),
}

impl<'a> TryFrom<&'a dyn Array> for ArrayWrapper<'a> {
    type Error = DataFusionError;

    fn try_from(
        value: &'a dyn Array,
    ) -> std::result::Result<ArrayWrapper<'a>, Self::Error> {
        match value.data_type() {
            DataType::List(_) => {
                Ok(ArrayWrapper::List(as_generic_list_array::<i32>(value)?))
            }
            DataType::LargeList(_) => Ok(ArrayWrapper::LargeList(
                as_generic_list_array::<i64>(value)?,
            )),
            DataType::FixedSizeList(_, _) => Ok(ArrayWrapper::FixedSizeList(
                as_fixed_size_list_array(value)?,
            )),
            _ => exec_err!("array_has does not support type '{:?}'.", value.data_type()),
        }
    }
}

impl<'a> ArrayWrapper<'a> {
    fn len(&self) -> usize {
        match self {
            ArrayWrapper::FixedSizeList(arr) => arr.len(),
            ArrayWrapper::List(arr) => arr.len(),
            ArrayWrapper::LargeList(arr) => arr.len(),
        }
    }

    fn null_count(&self) -> usize {
        match self {
            ArrayWrapper::FixedSizeList(arr) => arr.null_count(),
            ArrayWrapper::List(arr) => arr.null_count(),
            ArrayWrapper::LargeList(arr) => arr.null_count(),
        }
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Option<ArrayRef>> + 'a> {
        match self {
            ArrayWrapper::FixedSizeList(arr) => Box::new(arr.iter()),
            ArrayWrapper::List(arr) => Box::new(arr.iter()),
            ArrayWrapper::LargeList(arr) => Box::new(arr.iter()),
        }
    }

    fn values(&self) -> &ArrayRef {
        match self {
            ArrayWrapper::FixedSizeList(arr) => arr.values(),
            ArrayWrapper::List(arr) => arr.values(),
            ArrayWrapper::LargeList(arr) => arr.values(),
        }
    }

    fn value_type(&self) -> DataType {
        match self {
            ArrayWrapper::FixedSizeList(arr) => arr.value_type(),
            ArrayWrapper::List(arr) => arr.value_type(),
            ArrayWrapper::LargeList(arr) => arr.value_type(),
        }
    }

    fn nulls(&self) -> Option<NullBuffer> {
        match self {
            ArrayWrapper::FixedSizeList(arr) => arr.nulls().cloned(),
            ArrayWrapper::List(arr) => arr.nulls().cloned(),
            ArrayWrapper::LargeList(arr) => arr.nulls().cloned(),
        }
    }

    fn offsets(&self) -> Box<dyn Iterator<Item = usize> + 'a> {
        match self {
            ArrayWrapper::FixedSizeList(arr) => {
                let offsets = (0..=arr.len())
                    .step_by(arr.value_length() as usize)
                    .collect::<Vec<_>>();
                Box::new(offsets.into_iter())
            }
            ArrayWrapper::List(arr) => {
                Box::new(arr.offsets().iter().map(|o| (*o) as usize))
            }
            ArrayWrapper::LargeList(arr) => {
                Box::new(arr.offsets().iter().map(|o| (*o) as usize))
            }
        }
    }
}

trait HashEqual: HashValue {
    fn equals(&self, other: &Self) -> bool;

    fn canonical_hash(&self, state: &RandomState) -> u64 {
        self.hash_one(state)
    }
}

impl<T: HashEqual + ?Sized> HashEqual for &T {
    fn equals(&self, other: &Self) -> bool {
        T::equals(self, other)
    }

    fn canonical_hash(&self, state: &RandomState) -> u64 {
        T::canonical_hash(self, state)
    }
}

macro_rules! hash_equal {
    ($($t:ty),+) => {
        $(impl HashEqual for $t {
            fn equals(&self, other: &Self) -> bool {
                self == other
            }
        })*
    };
}

hash_equal!(i8, i16, i32, i64, i128, u8, u16, u32, u64, bool, str, [u8]);

macro_rules! hash_equal_float {
    ($($t:ty),+) => {
        $(impl HashEqual for $t {
            fn equals(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    self == other
                }
            }

            fn canonical_hash(&self, state: &RandomState) -> u64 {
                if self.is_nan() {
                    state.hash_one(self.to_bits())
                } else {
                    let canonical = if *self == 0.0 { 0.0 } else { *self };
                    state.hash_one(canonical.to_bits())
                }
            }
        })*
    };
}

hash_equal_float!(f32, f64);

struct RowHashSetBuilder {
    state: RandomState,
    map: HashMap<usize, (), RandomState>,
    has_null: bool,
}

impl RowHashSetBuilder {
    fn new() -> Self {
        Self {
            state: RandomState::new(),
            map: HashMap::with_hasher(RandomState::new()),
            has_null: false,
        }
    }

    fn prepare<A>(&mut self, array: &A)
    where
        A: Array,
        for<'a> &'a A: ArrayAccessor,
        for<'a> <&'a A as ArrayAccessor>::Item: HashEqual,
    {
        self.map.clear();
        let null_count = array.null_count();
        self.has_null = null_count != 0;
        self.map.reserve(array.len() - null_count);

        let accessor = array;
        let insert_value = |idx| {
            let value = accessor.value(idx);
            let hash = value.canonical_hash(&self.state);
            if let RawEntryMut::Vacant(v) = self
                .map
                .raw_entry_mut()
                .from_hash(hash, |existing| accessor.value(*existing).equals(&value))
            {
                v.insert_with_hasher(hash, idx, (), |existing_idx| {
                    accessor.value(*existing_idx).canonical_hash(&self.state)
                });
            }
        };

        match accessor.nulls() {
            Some(nulls) => {
                BitIndexIterator::new(nulls.validity(), nulls.offset(), nulls.len())
                    .for_each(insert_value)
            }
            None => (0..accessor.len()).for_each(insert_value),
        }
    }

    fn contains<'a, A>(&self, array: &'a A, value: <&'a A as ArrayAccessor>::Item) -> bool
    where
        A: Array,
        for<'b> &'b A: ArrayAccessor,
        for<'b> <&'b A as ArrayAccessor>::Item: HashEqual,
    {
        let accessor = array;
        let hash = value.canonical_hash(&self.state);
        self.map
            .raw_entry()
            .from_hash(hash, |existing| accessor.value(*existing).equals(&value))
            .is_some()
    }

    fn has_null(&self) -> bool {
        self.has_null
    }
}

fn downcast_array_ref<'a, A: Array + 'static>(
    array: &'a ArrayRef,
    dt: &DataType,
) -> Result<&'a A> {
    array.as_any().downcast_ref::<A>().ok_or_else(|| {
        DataFusionError::Execution(format!("array_has does not support type '{dt:?}'"))
    })
}

fn array_has_non_nested_generic<A>(
    haystack: &ArrayWrapper<'_>,
    needle: &A,
) -> Result<BooleanArray>
where
    A: Array + 'static,
    for<'a> &'a A: ArrayAccessor,
    for<'a> <&'a A as ArrayAccessor>::Item: HashEqual,
{
    let mut builder = BooleanArray::builder(haystack.len());
    let mut hash_builder = RowHashSetBuilder::new();

    for (i, arr_opt) in haystack.iter().enumerate() {
        if arr_opt.is_none() || needle.is_null(i) {
            builder.append_null();
            continue;
        }

        let arr = arr_opt.unwrap();
        let typed = arr.as_any().downcast_ref::<A>().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "array_has expected value array of type '{:?}' but found '{:?}'",
                needle.data_type(),
                arr.data_type()
            ))
        })?;

        hash_builder.prepare(typed);
        let contains = hash_builder.contains(typed, needle.value(i));
        builder.append_value(contains);
    }

    Ok(builder.finish())
}

fn array_has_all_and_any_non_nested<A>(
    haystack: &ArrayWrapper<'_>,
    needle: &ArrayWrapper<'_>,
    comparison_type: ComparisonType,
) -> Result<BooleanArray>
where
    A: Array + 'static,
    for<'a> &'a A: ArrayAccessor,
    for<'a> <&'a A as ArrayAccessor>::Item: HashEqual,
{
    let mut builder = BooleanArray::builder(haystack.len());
    let mut hash_builder = RowHashSetBuilder::new();

    for (haystack_row, needle_row) in haystack.iter().zip(needle.iter()) {
        match (haystack_row, needle_row) {
            (Some(haystack_row), Some(needle_row)) => {
                let haystack_typed =
                    haystack_row.as_any().downcast_ref::<A>().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                        "array_has expected value array of type '{:?}' but found '{:?}'",
                        haystack.value_type(),
                        haystack_row.data_type()
                    ))
                    })?;

                let needle_typed =
                    needle_row.as_any().downcast_ref::<A>().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                        "array_has expected needle array of type '{:?}' but found '{:?}'",
                        needle.value_type(),
                        needle_row.data_type()
                    ))
                    })?;

                hash_builder.prepare(haystack_typed);
                let result = match comparison_type {
                    ComparisonType::All => {
                        evaluate_all(&hash_builder, haystack_typed, needle_typed)
                    }
                    ComparisonType::Any => {
                        evaluate_any(&hash_builder, haystack_typed, needle_typed)
                    }
                };
                builder.append_value(result);
            }
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish())
}

fn evaluate_all<A>(builder: &RowHashSetBuilder, haystack: &A, needle: &A) -> bool
where
    A: Array,
    for<'a> &'a A: ArrayAccessor,
    for<'a> <&'a A as ArrayAccessor>::Item: HashEqual,
{
    ArrayIter::new(needle).all(|value| match value {
        Some(value) => builder.contains(haystack, value),
        None => builder.has_null(),
    })
}

fn evaluate_any<A>(builder: &RowHashSetBuilder, haystack: &A, needle: &A) -> bool
where
    A: Array,
    for<'a> &'a A: ArrayAccessor,
    for<'a> <&'a A as ArrayAccessor>::Item: HashEqual,
{
    ArrayIter::new(needle).any(|value| match value {
        Some(value) => builder.contains(haystack, value),
        None => builder.has_null(),
    })
}

fn try_array_has_non_nested(
    haystack: &ArrayWrapper<'_>,
    needle: &ArrayRef,
) -> Result<Option<BooleanArray>> {
    let value_type = haystack.value_type();
    if value_type.is_nested()
        || matches!(
            value_type,
            DataType::Struct(_)
                | DataType::Union(_, _)
                | DataType::Map(_, _)
                | DataType::Dictionary(_, _)
                | DataType::Null
        )
    {
        return Ok(None);
    }

    macro_rules! typed_case {
        ($array_ty:ty) => {{
            let typed = downcast_array_ref::<$array_ty>(needle, &value_type)?;
            Some(array_has_non_nested_generic::<$array_ty>(haystack, typed)?)
        }};
    }

    let result = match value_type {
        DataType::Boolean => typed_case!(BooleanArray),
        DataType::Int8 => typed_case!(Int8Array),
        DataType::Int16 => typed_case!(Int16Array),
        DataType::Int32 => typed_case!(Int32Array),
        DataType::Int64 => typed_case!(Int64Array),
        DataType::UInt8 => typed_case!(UInt8Array),
        DataType::UInt16 => typed_case!(UInt16Array),
        DataType::UInt32 => typed_case!(UInt32Array),
        DataType::UInt64 => typed_case!(UInt64Array),
        DataType::Float32 => typed_case!(Float32Array),
        DataType::Float64 => typed_case!(Float64Array),
        DataType::Decimal128(_, _) => typed_case!(Decimal128Array),
        DataType::Date32 => typed_case!(Date32Array),
        DataType::Date64 => typed_case!(Date64Array),
        DataType::Time32(TimeUnit::Second) => typed_case!(Time32SecondArray),
        DataType::Time32(TimeUnit::Millisecond) => typed_case!(Time32MillisecondArray),
        DataType::Time64(TimeUnit::Microsecond) => typed_case!(Time64MicrosecondArray),
        DataType::Time64(TimeUnit::Nanosecond) => typed_case!(Time64NanosecondArray),
        DataType::Timestamp(TimeUnit::Second, _) => typed_case!(TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            typed_case!(TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            typed_case!(TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            typed_case!(TimestampNanosecondArray)
        }
        DataType::Duration(TimeUnit::Second) => typed_case!(DurationSecondArray),
        DataType::Duration(TimeUnit::Millisecond) => {
            typed_case!(DurationMillisecondArray)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            typed_case!(DurationMicrosecondArray)
        }
        DataType::Duration(TimeUnit::Nanosecond) => typed_case!(DurationNanosecondArray),
        DataType::Binary => typed_case!(BinaryArray),
        DataType::LargeBinary => typed_case!(LargeBinaryArray),
        DataType::BinaryView => typed_case!(BinaryViewArray),
        DataType::FixedSizeBinary(_) => typed_case!(FixedSizeBinaryArray),
        DataType::Utf8 => typed_case!(StringArray),
        DataType::LargeUtf8 => typed_case!(LargeStringArray),
        DataType::Utf8View => typed_case!(StringViewArray),
        _ => None,
    };

    Ok(result)
}

fn try_array_has_all_and_any_non_nested(
    haystack: &ArrayWrapper<'_>,
    needle: &ArrayWrapper<'_>,
    comparison_type: ComparisonType,
) -> Result<Option<BooleanArray>> {
    let value_type = haystack.value_type();
    if value_type != needle.value_type()
        || value_type.is_nested()
        || matches!(
            value_type,
            DataType::Struct(_)
                | DataType::Union(_, _)
                | DataType::Map(_, _)
                | DataType::Dictionary(_, _)
                | DataType::Null
        )
    {
        return Ok(None);
    }

    macro_rules! typed_case {
        ($array_ty:ty) => {{
            Some(array_has_all_and_any_non_nested::<$array_ty>(
                haystack,
                needle,
                comparison_type,
            )?)
        }};
    }

    let result = match value_type {
        DataType::Boolean => typed_case!(BooleanArray),
        DataType::Int8 => typed_case!(Int8Array),
        DataType::Int16 => typed_case!(Int16Array),
        DataType::Int32 => typed_case!(Int32Array),
        DataType::Int64 => typed_case!(Int64Array),
        DataType::UInt8 => typed_case!(UInt8Array),
        DataType::UInt16 => typed_case!(UInt16Array),
        DataType::UInt32 => typed_case!(UInt32Array),
        DataType::UInt64 => typed_case!(UInt64Array),
        DataType::Float32 => typed_case!(Float32Array),
        DataType::Float64 => typed_case!(Float64Array),
        DataType::Decimal128(_, _) => typed_case!(Decimal128Array),
        DataType::Date32 => typed_case!(Date32Array),
        DataType::Date64 => typed_case!(Date64Array),
        DataType::Time32(TimeUnit::Second) => typed_case!(Time32SecondArray),
        DataType::Time32(TimeUnit::Millisecond) => typed_case!(Time32MillisecondArray),
        DataType::Time64(TimeUnit::Microsecond) => typed_case!(Time64MicrosecondArray),
        DataType::Time64(TimeUnit::Nanosecond) => typed_case!(Time64NanosecondArray),
        DataType::Timestamp(TimeUnit::Second, _) => typed_case!(TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            typed_case!(TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            typed_case!(TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            typed_case!(TimestampNanosecondArray)
        }
        DataType::Duration(TimeUnit::Second) => typed_case!(DurationSecondArray),
        DataType::Duration(TimeUnit::Millisecond) => {
            typed_case!(DurationMillisecondArray)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            typed_case!(DurationMicrosecondArray)
        }
        DataType::Duration(TimeUnit::Nanosecond) => typed_case!(DurationNanosecondArray),
        DataType::Binary => typed_case!(BinaryArray),
        DataType::LargeBinary => typed_case!(LargeBinaryArray),
        DataType::BinaryView => typed_case!(BinaryViewArray),
        DataType::FixedSizeBinary(_) => typed_case!(FixedSizeBinaryArray),
        DataType::Utf8 => typed_case!(StringArray),
        DataType::LargeUtf8 => typed_case!(LargeStringArray),
        DataType::Utf8View => typed_case!(StringViewArray),
        _ => None,
    };

    Ok(result)
}

fn array_has_dispatch_for_array(
    haystack: ArrayWrapper<'_>,
    needle: &ArrayRef,
) -> Result<ArrayRef> {
    if let Some(result) = try_array_has_non_nested(&haystack, needle)? {
        return Ok(Arc::new(result));
    }

    let mut boolean_builder = BooleanArray::builder(haystack.len());
    for (i, arr) in haystack.iter().enumerate() {
        if arr.is_none() || needle.is_null(i) {
            boolean_builder.append_null();
            continue;
        }
        let arr = arr.unwrap();
        let is_nested = arr.data_type().is_nested();
        let needle_row = Scalar::new(needle.slice(i, 1));
        let eq_array = compare_with_eq(&arr, &needle_row, is_nested)?;
        boolean_builder.append_value(eq_array.true_count() > 0);
    }

    Ok(Arc::new(boolean_builder.finish()))
}

fn array_has_dispatch_for_scalar(
    haystack: ArrayWrapper<'_>,
    needle: &dyn Datum,
) -> Result<ArrayRef> {
    let values = haystack.values();
    let is_nested = values.data_type().is_nested();
    // If first argument is empty list (second argument is non-null), return false
    // i.e. array_has([], non-null element) -> false
    if haystack.len() == 0 {
        return Ok(Arc::new(BooleanArray::new(
            BooleanBuffer::new_unset(haystack.len()),
            None,
        )));
    }
    let eq_array = compare_with_eq(values, needle, is_nested)?;
    let mut final_contained = vec![None; haystack.len()];

    // Check validity buffer to distinguish between null and empty arrays
    let validity = match &haystack {
        ArrayWrapper::FixedSizeList(arr) => arr.nulls(),
        ArrayWrapper::List(arr) => arr.nulls(),
        ArrayWrapper::LargeList(arr) => arr.nulls(),
    };

    for (i, (start, end)) in haystack.offsets().tuple_windows().enumerate() {
        let length = end - start;

        // Check if the array at this position is null
        if let Some(validity_buffer) = validity {
            if !validity_buffer.is_valid(i) {
                final_contained[i] = None; // null array -> null result
                continue;
            }
        }

        // For non-null arrays: length is 0 for empty arrays
        if length == 0 {
            final_contained[i] = Some(false); // empty array -> false
        } else {
            let sliced_array = eq_array.slice(start, length);
            final_contained[i] = Some(sliced_array.true_count() > 0);
        }
    }

    Ok(Arc::new(BooleanArray::from(final_contained)))
}

fn array_has_all_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    array_has_all_and_any_inner(args, ComparisonType::All)
}

// General row comparison for array_has_all and array_has_any
fn general_array_has_for_all_and_any<'a>(
    haystack: &ArrayWrapper<'a>,
    needle: &ArrayWrapper<'a>,
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    let mut boolean_builder = BooleanArray::builder(haystack.len());
    let converter = RowConverter::new(vec![SortField::new(haystack.value_type())])?;

    for (arr, sub_arr) in haystack.iter().zip(needle.iter()) {
        if let (Some(arr), Some(sub_arr)) = (arr, sub_arr) {
            let arr_values = converter.convert_columns(&[arr])?;
            let sub_arr_values = converter.convert_columns(&[sub_arr])?;
            boolean_builder.append_value(general_array_has_all_and_any_kernel(
                arr_values,
                sub_arr_values,
                comparison_type,
            ));
        } else {
            boolean_builder.append_null();
        }
    }

    Ok(Arc::new(boolean_builder.finish()))
}

fn array_has_all_and_any_dispatch<'a>(
    haystack: &ArrayWrapper<'a>,
    needle: &ArrayWrapper<'a>,
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    if needle.values().is_empty() {
        let values = match comparison_type {
            ComparisonType::All => BooleanBuffer::new_set(haystack.len()),
            ComparisonType::Any => BooleanBuffer::new_unset(haystack.len()),
        };
        let nulls = match (haystack.nulls(), needle.nulls()) {
            (Some(haystack_nulls), Some(needle_nulls)) => {
                let mut builder = NullBufferBuilder::new(haystack.len());
                for i in 0..haystack.len() {
                    if haystack_nulls.is_valid(i) && needle_nulls.is_valid(i) {
                        builder.append_non_null();
                    } else {
                        builder.append_null();
                    }
                }
                Some(builder.finish())
            }
            (Some(haystack_nulls), None) => Some(haystack_nulls),
            (None, Some(needle_nulls)) => Some(needle_nulls),
            (None, None) => None,
        };
        Ok(Arc::new(BooleanArray::new(values, nulls)))
    } else if let Some(result) =
        try_array_has_all_and_any_non_nested(haystack, needle, comparison_type)?
    {
        Ok(Arc::new(result))
    } else {
        general_array_has_for_all_and_any(haystack, needle, comparison_type)
    }
}

fn array_has_all_and_any_inner(
    args: &[ArrayRef],
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    if matches!(args[0].data_type(), DataType::Null)
        || matches!(args[1].data_type(), DataType::Null)
    {
        let len = args.iter().map(|arg| arg.len()).max().unwrap_or(0);
        return Ok(Arc::new(BooleanArray::new_null(len)));
    }
    let haystack: ArrayWrapper = args[0].as_ref().try_into()?;
    let needle: ArrayWrapper = args[1].as_ref().try_into()?;
    array_has_all_and_any_dispatch(&haystack, &needle, comparison_type)
}

fn array_has_any_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    array_has_all_and_any_inner(args, ComparisonType::Any)
}

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns true if all elements of sub-array exist in array.",
    syntax_example = "array_has_all(array, sub-array)",
    sql_example = r#"```sql
> select array_has_all([1, 2, 3, 4], [2, 3]);
+--------------------------------------------+
| array_has_all(List([1,2,3,4]), List([2,3])) |
+--------------------------------------------+
| true                                       |
+--------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(
        name = "sub-array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayHasAll {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHasAll {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHasAll {
    pub fn new() -> Self {
        Self {
            signature: Signature::arrays(2, None, Volatility::Immutable),
            aliases: vec![String::from("list_has_all")],
        }
    }
}

impl ScalarUDFImpl for ArrayHasAll {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has_all"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_has_all_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns true if any elements exist in both arrays.",
    syntax_example = "array_has_any(array, sub-array)",
    sql_example = r#"```sql
> select array_has_any([1, 2, 3], [3, 4]);
+------------------------------------------+
| array_has_any(List([1,2,3]), List([3,4])) |
+------------------------------------------+
| true                                     |
+------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(
        name = "sub-array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayHasAny {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHasAny {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHasAny {
    pub fn new() -> Self {
        Self {
            signature: Signature::arrays(2, None, Volatility::Immutable),
            aliases: vec![String::from("list_has_any"), String::from("arrays_overlap")],
        }
    }
}

impl ScalarUDFImpl for ArrayHasAny {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has_any"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_has_any_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Represents the type of comparison for array_has.
#[derive(Debug, PartialEq, Clone, Copy)]
enum ComparisonType {
    // array_has_all
    All,
    // array_has_any
    Any,
}

fn general_array_has_all_and_any_kernel(
    haystack_rows: Rows,
    needle_rows: Rows,
    comparison_type: ComparisonType,
) -> bool {
    match comparison_type {
        ComparisonType::All => needle_rows.iter().all(|needle_row| {
            haystack_rows
                .iter()
                .any(|haystack_row| haystack_row == needle_row)
        }),
        ComparisonType::Any => needle_rows.iter().any(|needle_row| {
            haystack_rows
                .iter()
                .any(|haystack_row| haystack_row == needle_row)
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{
            create_array, Array, ArrayRef, AsArray, BooleanArray, Float64Array,
            Int32Array, ListArray,
        },
        buffer::OffsetBuffer,
        datatypes::{DataType, Field},
    };
    use datafusion_common::{
        config::ConfigOptions, utils::SingleRowListArrayBuilder, DataFusionError,
        ScalarValue,
    };
    use datafusion_expr::{
        col, execution_props::ExecutionProps, lit, simplify::ExprSimplifyResult,
        ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDFImpl,
    };

    use crate::expr_fn::make_array;

    use super::{array_has_inner_for_array, ArrayHas};

    #[test]
    fn test_simplify_array_has_to_in_list() {
        let haystack = lit(SingleRowListArrayBuilder::new(create_array!(
            Int32,
            [1, 2, 3]
        ))
        .build_list_scalar());
        let needle = col("c");

        let props = ExecutionProps::new();
        let context = datafusion_expr::simplify::SimplifyContext::new(&props);

        let Ok(ExprSimplifyResult::Simplified(Expr::InList(in_list))) =
            ArrayHas::new().simplify(vec![haystack, needle.clone()], &context)
        else {
            panic!("Expected simplified expression");
        };

        assert_eq!(
            in_list,
            datafusion_expr::expr::InList {
                expr: Box::new(needle),
                list: vec![lit(1), lit(2), lit(3)],
                negated: false,
            }
        );
    }

    #[test]
    fn test_simplify_array_has_with_make_array_to_in_list() {
        let haystack = make_array(vec![lit(1), lit(2), lit(3)]);
        let needle = col("c");

        let props = ExecutionProps::new();
        let context = datafusion_expr::simplify::SimplifyContext::new(&props);

        let Ok(ExprSimplifyResult::Simplified(Expr::InList(in_list))) =
            ArrayHas::new().simplify(vec![haystack, needle.clone()], &context)
        else {
            panic!("Expected simplified expression");
        };

        assert_eq!(
            in_list,
            datafusion_expr::expr::InList {
                expr: Box::new(needle),
                list: vec![lit(1), lit(2), lit(3)],
                negated: false,
            }
        );
    }

    #[test]
    fn test_array_has_complex_list_not_simplified() {
        let haystack = col("c1");
        let needle = col("c2");

        let props = ExecutionProps::new();
        let context = datafusion_expr::simplify::SimplifyContext::new(&props);

        let Ok(ExprSimplifyResult::Original(args)) =
            ArrayHas::new().simplify(vec![haystack, needle.clone()], &context)
        else {
            panic!("Expected simplified expression");
        };

        assert_eq!(args, vec![col("c1"), col("c2")],);
    }

    #[test]
    fn test_array_has_all_any_with_null_needle_rows() -> Result<(), DataFusionError> {
        let list_field: arrow::datatypes::FieldRef =
            Field::new_list_field(DataType::Int32, true).into();
        let haystack = ListArray::new(
            Arc::clone(&list_field),
            OffsetBuffer::new(vec![0, 1, 2].into()),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            None,
        );

        let needle = ListArray::new(
            list_field,
            OffsetBuffer::new(vec![0, 0, 0].into()),
            Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef,
            Some(vec![true, false].into()),
        );

        let haystack: ArrayRef = Arc::new(haystack);
        let needle: ArrayRef = Arc::new(needle);

        let any_result = super::array_has_all_and_any_inner(
            &[Arc::clone(&haystack), Arc::clone(&needle)],
            super::ComparisonType::Any,
        )?;
        let any_result = any_result.as_boolean();
        assert_eq!(any_result.len(), 2);
        assert!(!any_result.value(0));
        assert!(any_result.is_null(1));

        let all_result = super::array_has_all_and_any_inner(
            &[haystack, needle],
            super::ComparisonType::All,
        )?;
        let all_result = all_result.as_boolean();
        assert_eq!(all_result.len(), 2);
        assert!(all_result.value(0));
        assert!(all_result.is_null(1));

        Ok(())
    }

    #[test]
    fn test_array_has_all_any_empty_needle_propagates_nulls(
    ) -> Result<(), DataFusionError> {
        let list_field: arrow::datatypes::FieldRef =
            Field::new_list_field(DataType::Int32, true).into();
        let haystack = ListArray::new(
            Arc::clone(&list_field),
            OffsetBuffer::new(vec![0, 0, 1].into()),
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            Some(vec![false, true].into()),
        );

        let needle = ListArray::new(
            list_field,
            OffsetBuffer::new(vec![0, 0, 0].into()),
            Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef,
            None,
        );

        let haystack: ArrayRef = Arc::new(haystack);
        let needle: ArrayRef = Arc::new(needle);

        let any_result = super::array_has_all_and_any_inner(
            &[Arc::clone(&haystack), Arc::clone(&needle)],
            super::ComparisonType::Any,
        )?;
        let any_result = any_result.as_boolean();
        assert_eq!(any_result.len(), 2);
        assert!(any_result.is_null(0));
        assert!(!any_result.value(1));

        let all_result = super::array_has_all_and_any_inner(
            &[haystack, needle],
            super::ComparisonType::All,
        )?;
        let all_result = all_result.as_boolean();
        assert_eq!(all_result.len(), 2);
        assert!(all_result.is_null(0));
        assert!(all_result.value(1));

        Ok(())
    }

    #[test]
    fn test_array_has_all_any_with_null_scalar_needle() -> Result<(), DataFusionError> {
        let list_field: arrow::datatypes::FieldRef =
            Field::new_list_field(DataType::Int32, true).into();
        let haystack = ListArray::new(
            Arc::clone(&list_field),
            OffsetBuffer::new(vec![0, 1, 2].into()),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            None,
        );
        let haystack_len = haystack.len();

        let needle = ScalarValue::Null.to_array_of_size(haystack_len)?;

        let haystack: ArrayRef = Arc::new(haystack);

        let any_result = super::array_has_all_and_any_inner(
            &[Arc::clone(&haystack), Arc::clone(&needle)],
            super::ComparisonType::Any,
        )?;
        let any_result = any_result.as_boolean();
        assert_eq!(any_result.len(), haystack_len);
        assert_eq!(any_result.null_count(), haystack_len);

        let all_result = super::array_has_all_and_any_inner(
            &[haystack, needle],
            super::ComparisonType::All,
        )?;
        let all_result = all_result.as_boolean();
        assert_eq!(all_result.len(), haystack_len);
        assert_eq!(all_result.null_count(), haystack_len);

        Ok(())
    }

    #[test]
    fn test_array_has_list_empty_child() -> Result<(), DataFusionError> {
        let haystack_field = Arc::new(Field::new_list(
            "haystack",
            Field::new_list("", Field::new("", DataType::Int32, true), true),
            true,
        ));
        let needle_field = Arc::new(Field::new("needle", DataType::Int32, true));
        let return_field = Arc::new(Field::new_list(
            "return",
            Field::new("", DataType::Boolean, true),
            true,
        ));

        let haystack = ListArray::new(
            Field::new_list_field(DataType::Int32, true).into(),
            OffsetBuffer::new(vec![0, 0].into()),
            Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef,
            Some(vec![false].into()),
        );

        let haystack = ColumnarValue::Array(Arc::new(haystack));
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));

        let result = ArrayHas::new().invoke_with_args(ScalarFunctionArgs {
            args: vec![haystack, needle],
            arg_fields: vec![haystack_field, needle_field],
            number_rows: 1,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        })?;

        let output = result.into_array(1)?;
        let output = output.as_boolean();
        assert_eq!(output.len(), 1);
        assert!(output.is_null(0));

        Ok(())
    }

    #[test]
    fn test_array_has_float_signed_zero() -> Result<(), DataFusionError> {
        let haystack_values = Arc::new(Float64Array::from(vec![0.0])) as ArrayRef;
        let haystack = Arc::new(ListArray::new(
            Field::new_list_field(DataType::Float64, true).into(),
            OffsetBuffer::new(vec![0, 1].into()),
            haystack_values,
            None,
        )) as ArrayRef;

        let needle = Arc::new(Float64Array::from(vec![-0.0])) as ArrayRef;

        let result = array_has_inner_for_array(&haystack, &needle)?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean array");

        assert_eq!(result.len(), 1);
        assert!(result.value(0));

        Ok(())
    }

    #[test]
    fn test_array_has_float_nan() -> Result<(), DataFusionError> {
        let haystack_values = Arc::new(Float64Array::from(vec![f64::NAN])) as ArrayRef;
        let haystack = Arc::new(ListArray::new(
            Field::new_list_field(DataType::Float64, true).into(),
            OffsetBuffer::new(vec![0, 1].into()),
            haystack_values,
            None,
        )) as ArrayRef;

        let needle = Arc::new(Float64Array::from(vec![f64::NAN])) as ArrayRef;

        let result = array_has_inner_for_array(&haystack, &needle)?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean array");

        assert_eq!(result.len(), 1);
        assert!(!result.value(0));

        Ok(())
    }
}
