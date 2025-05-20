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

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_catalog::TableProvider;
use datafusion_common::assert_batches_sorted_eq;
use datafusion_common::record_batch;
use datafusion_common::Result;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::listing::PartitionedFile;
use datafusion_datasource::physical_plan::FileScanConfig;
use datafusion_datasource::schema_adapter::{
    SchemaAdapter, SchemaAdapterFactory, SchemaMapper,
};
use datafusion_physical_plan::collect;
use object_store::{path::Path, ObjectMeta};
use tempfile::TempDir;

use crate::datasource::file_format::parquet::ParquetFormat;
use crate::datasource::listing::table::{FileSourceExt, ListingTable};
use crate::datasource::listing::{ListingOptions, ListingTableConfig, ListingTableUrl};
use crate::execution::context::SessionContext;
use crate::test::object_store::register_test_store;

#[tokio::test]
async fn test_listing_table_with_schema_adapter_factory() -> Result<()> {
    let ctx = SessionContext::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]));

    // Create a test table path
    let table_path = ListingTableUrl::parse("memory:///table/").unwrap();

    // Create config with schema adapter
    let config = ListingTableConfig::new(table_path.clone())
        .with_schema(schema.clone())
        .with_listing_options(ListingOptions::new(Arc::new(ParquetFormat::default())))
        .with_schema_adapter_factory(Arc::new(TestSchemaAdapterFactory));

    // Create table with the config
    let table = ListingTable::try_new(config)?;

    // Verify the schema adapter factory was set correctly
    assert!(table.schema_adapter_factory.is_some());

    Ok(())
}

#[tokio::test]
async fn test_file_source_with_schema_adapter() -> Result<()> {
    // Create a file source
    let source = Arc::new(ParquetFormat::default().file_source());

    // Add schema adapter
    let source_with_adapter =
        source.with_schema_adapter(Some(Arc::new(TestSchemaAdapterFactory)));

    // The source should still be the same type (we're just adding capabilities)
    assert_eq!(
        source.as_any().type_id(),
        source_with_adapter.as_any().type_id()
    );

    // If no adapter is provided, should return original source
    let same_source = source.with_schema_adapter(None);
    assert!(Arc::ptr_eq(&source, &same_source));

    Ok(())
}

#[cfg(feature = "parquet")]
#[tokio::test]
async fn test_parquet_file_reader_preserves_schema_adapter() -> Result<()> {
    use datafusion_datasource_parquet::source::ParquetSource;
    use parquet::arrow::ArrowWriter;
    use std::fs;

    // Create a temporary directory and parquet file
    let tmp_dir = TempDir::new()?;
    let file_path = tmp_dir.path().join("test.parquet");

    // Create schema and data
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));

    // Create parquet file
    {
        let file = fs::File::create(&file_path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;

        writer.write(&batch)?;
        writer.close()?;
    }

    // Create file metadata
    let metadata = fs::metadata(&file_path)?;
    let meta = ObjectMeta {
        location: Path::from(file_path.to_string_lossy().as_ref()),
        last_modified: metadata.modified().map(chrono::DateTime::from)?,
        size: metadata.len(),
        e_tag: None,
        version: None,
    };

    // Create partitioned file
    let file = PartitionedFile {
        object_meta: meta,
        partition_values: vec![],
        range: None,
        statistics: None,
        extensions: None,
        metadata_size_hint: None,
    };

    // Create source with schema adapter
    let source = ParquetSource::default()
        .with_schema_adapter_factory(Arc::new(TestSchemaAdapterFactory));

    // Create scan config
    let config =
        FileScanConfig::new("file://tmp".into(), schema.clone(), Arc::new(source))
            .with_file(file);

    // Create new source from existing config in a way similar to how preserve_conf_schema_adapter_factory works
    let new_source = ParquetSource::default();
    let original_source = config
        .file_source()
        .as_any()
        .downcast_ref::<ParquetSource>()
        .unwrap();
    let factory = original_source.schema_adapter_factory().cloned();

    // Check factory was correctly retrieved
    assert!(factory.is_some());

    // Apply factory to new source
    let new_source_with_adapter =
        new_source.with_schema_adapter_factory(factory.unwrap());

    // Check new source has the adapter
    assert!(new_source_with_adapter.schema_adapter_factory().is_some());

    Ok(())
}

// Test schema adapter factory implementation
#[derive(Debug)]
struct TestSchemaAdapterFactory;

impl SchemaAdapterFactory for TestSchemaAdapterFactory {
    fn create(
        &self,
        projected_table_schema: SchemaRef,
        _table_schema: SchemaRef,
    ) -> Box<dyn SchemaAdapter> {
        Box::new(TestSchemaAdapter {
            table_schema: projected_table_schema,
        })
    }
}

struct TestSchemaAdapter {
    table_schema: SchemaRef,
}

impl SchemaAdapter for TestSchemaAdapter {
    fn map_column_index(&self, index: usize, file_schema: &Schema) -> Option<usize> {
        let field = self.table_schema.field(index);
        file_schema
            .fields()
            .iter()
            .position(|f| f.name() == field.name())
    }

    fn map_schema(
        &self,
        file_schema: &Schema,
    ) -> datafusion_common::Result<(Arc<dyn SchemaMapper>, Vec<usize>)> {
        let mut projection = Vec::with_capacity(file_schema.fields().len());

        for (file_idx, file_field) in file_schema.fields().iter().enumerate() {
            if self
                .table_schema
                .fields()
                .iter()
                .any(|f| f.name() == file_field.name())
            {
                projection.push(file_idx);
            }
        }

        Ok((Arc::new(TestSchemaMapping {}), projection))
    }
}

#[derive(Debug)]
struct TestSchemaMapping {}

impl SchemaMapper for TestSchemaMapping {
    fn map_batch(&self, batch: RecordBatch) -> datafusion_common::Result<RecordBatch> {
        // Add an extra "name" column with "test" values
        let mut fields = batch.schema().fields().clone();
        fields.push(Field::new("name", DataType::Utf8, true));

        let schema = Arc::new(Schema::new(fields));

        let mut columns = batch.columns().to_vec();
        let extra_column =
            Arc::new(StringArray::from_iter(vec!["test"; batch.num_rows()]));
        columns.push(extra_column);

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    fn map_column_statistics(
        &self,
        file_col_statistics: &[datafusion_common::stats::ColumnStatistics],
    ) -> datafusion_common::Result<Vec<datafusion_common::stats::ColumnStatistics>> {
        // Just pass through the statistics
        Ok(file_col_statistics.to_vec())
    }
}

#[cfg(feature = "parquet")]
#[tokio::test]
async fn test_listing_table_uses_schema_adapter() -> Result<()> {
    use crate::test::object_store::make_test_store_and_state;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::file_format::parquet::ParquetFormat;
    use datafusion_physical_plan::ExecutionPlanProperties;
    use parquet::arrow::ArrowWriter;
    use std::fs;

    // Create a temporary directory and parquet file
    let tmp_dir = TempDir::new()?;
    let file_path = tmp_dir.path().join("test.parquet");

    // Create schema and data
    let file_schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]));

    // Create parquet file with just the id column
    {
        let file = fs::File::create(&file_path)?;
        let mut writer = ArrowWriter::try_new(file, file_schema.clone(), None)?;

        let batch = RecordBatch::try_new(
            file_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;

        writer.write(&batch)?;
        writer.close()?;
    }

    // Setup test store
    let ctx = SessionContext::new();
    let (store, state) = make_test_store_and_state(&ctx)?;

    // Register the file in the store
    let url = format!("test://{}", file_path.to_string_lossy());
    let data_path = Path::from(file_path.to_string_lossy().as_ref());
    let meta = fs::metadata(&file_path)?;
    let object_meta = ObjectMeta {
        location: data_path,
        last_modified: meta.modified().map(chrono::DateTime::from)?,
        size: meta.len(),
        e_tag: None,
        version: None,
    };
    let expected_location = object_meta.location.clone();
    store.add_meta(object_meta);

    // Create listing table with schema adapter
    let table_path = ListingTableUrl::parse("test:///").unwrap();
    let config = ListingTableConfig::new(table_path)
        .with_schema(table_schema.clone())
        .with_listing_options(ListingOptions::new(Arc::new(ParquetFormat::default())))
        .with_schema_adapter_factory(Arc::new(TestSchemaAdapterFactory));

    let table = ListingTable::try_new(config)?;

    // We need to use the TableProvider trait for the scan method
    use datafusion_catalog::TableProvider;
    // Scan the table
    let exec = table.scan(&state, None, &[], None).await?;

    // Execute the plan
    let batches = collect(exec, ctx.task_ctx()).await?;

    // Check the result - should have both id and name columns
    // where name column was added by our schema adapter
    let expected = [
        "+----+------+",
        "| id | name |",
        "+----+------+",
        "| 1  | test |",
        "| 2  | test |",
        "| 3  | test |",
        "+----+------+",
    ];

    assert_batches_sorted_eq!(expected, &batches);

    Ok(())
}
