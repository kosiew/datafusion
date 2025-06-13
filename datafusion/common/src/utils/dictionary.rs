use crate::cast::as_dictionary_array;
use arrow::array::{Array, ArrayRef, NullBufferBuilder};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{
    ArrowDictionaryKeyType, DataType, Int16Type, Int32Type, Int64Type, Int8Type,
    UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};

/// Return the logical null buffer for `array`, combining dictionary
/// key nulls with nulls from the values array when applicable.
pub fn combined_dictionary_nulls(array: &ArrayRef) -> Option<NullBuffer> {
    match array.data_type() {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::Int8 => combine::<Int8Type>(array),
            DataType::Int16 => combine::<Int16Type>(array),
            DataType::Int32 => combine::<Int32Type>(array),
            DataType::Int64 => combine::<Int64Type>(array),
            DataType::UInt8 => combine::<UInt8Type>(array),
            DataType::UInt16 => combine::<UInt16Type>(array),
            DataType::UInt32 => combine::<UInt32Type>(array),
            DataType::UInt64 => combine::<UInt64Type>(array),
            _ => array.logical_nulls().as_ref().cloned(),
        },
        _ => array.logical_nulls().as_ref().cloned(),
    }
}

fn combine<K: ArrowDictionaryKeyType>(array: &ArrayRef) -> Option<NullBuffer> {
    let dict = match as_dictionary_array::<K>(array.as_ref()) {
        Ok(dict) => dict,
        Err(_) => return None,
    };

    let values_nulls = dict.values().logical_nulls();
    if dict.null_count() == 0 && values_nulls.is_none() {
        return None;
    }

    let mut builder = NullBufferBuilder::new(dict.len());
    for i in 0..dict.len() {
        match dict.key(i) {
            None => builder.append_null(),
            Some(value_idx) => {
                let is_null = values_nulls
                    .as_ref()
                    .map(|n| !n.is_valid(value_idx))
                    .unwrap_or(false);
                if is_null {
                    builder.append_null();
                } else {
                    builder.append_non_null();
                }
            }
        }
    }

    builder.finish()
}
