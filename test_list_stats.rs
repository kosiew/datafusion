// Quick test to check if Parquet writes statistics for List<String> columns

use arrow::array::{ArrayRef, ListArray, StringArray};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use std::fs::File;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "list_col",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));

    let file = File::create("/tmp/test_list_stats.parquet")?;

    let props = WriterProperties::builder()
        .set_max_row_group_size(100)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Row group 1: values starting with "aaa"
    let mut all_values = Vec::new();
    let mut offsets = vec![0i32];
    for i in 0..100 {
        all_values.push(format!("aaa_{}", i));
        all_values.push(format!("aaa_{}_b", i));
        offsets.push((offsets.last().unwrap() + 2) as i32);
    }

    let id_array =
        Arc::new(arrow::array::Int64Array::from_iter_values(0..100)) as ArrayRef;
    let values_array =
        Arc::new(StringArray::from_iter_values(all_values.iter())) as ArrayRef;
    let scalar_buffer: ScalarBuffer<i32> = offsets.into();
    let offset_buffer = OffsetBuffer::new(scalar_buffer);
    let list_array = Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        offset_buffer,
        values_array,
        None,
    )) as ArrayRef;

    let batch1 = RecordBatch::try_new(schema.clone(), vec![id_array, list_array])?;
    writer.write(&batch1)?;

    // Row group 2: values starting with "zzz"
    let mut all_values = Vec::new();
    let mut offsets = vec![0i32];
    for i in 0..100 {
        all_values.push(format!("zzz_{}", i));
        all_values.push(format!("zzz_{}_b", i));
        offsets.push((offsets.last().unwrap() + 2) as i32);
    }

    let id_array =
        Arc::new(arrow::array::Int64Array::from_iter_values(100..200)) as ArrayRef;
    let values_array =
        Arc::new(StringArray::from_iter_values(all_values.iter())) as ArrayRef;
    let scalar_buffer: ScalarBuffer<i32> = offsets.into();
    let offset_buffer = OffsetBuffer::new(scalar_buffer);
    let list_array = Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        offset_buffer,
        values_array,
        None,
    )) as ArrayRef;

    let batch2 = RecordBatch::try_new(schema.clone(), vec![id_array, list_array])?;
    writer.write(&batch2)?;

    writer.finish()?;

    println!("✅ Created test parquet file: /tmp/test_list_stats.parquet");
    println!(
        "Check statistics with: cargo run --example parquet-read-statistics /tmp/test_list_stats.parquet"
    );

    Ok(())
}
