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

//! [`ParquetOpener`] for opening Parquet files

use std::sync::Arc;

use crate::datasource::file_format::{
    coerce_file_schema_to_string_type, coerce_file_schema_to_view_type,
};
use crate::datasource::physical_plan::parquet::page_filter::PagePruningAccessPlanFilter;
use crate::datasource::physical_plan::parquet::row_group_filter::RowGroupAccessPlanFilter;
use crate::datasource::physical_plan::parquet::{
    row_filter, should_enable_page_index, ParquetAccessPlan,
};
use crate::datasource::physical_plan::{
    FileMeta, FileOpenFuture, FileOpener, ParquetFileMetrics, ParquetFileReaderFactory,
};
use crate::datasource::schema_adapter::SchemaAdapterFactory;

use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::{exec_err, Result};
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_optimizer::pruning::PruningPredicate;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;

use futures::{StreamExt, TryStreamExt};
use log::debug;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};

/// Implements [`FileOpener`] for a parquet file
pub(super) struct ParquetOpener {
    /// Execution partition index
    pub partition_index: usize,
    /// Column indexes in `table_schema` needed by the query
    pub projection: Arc<[usize]>,
    /// Target number of rows in each output RecordBatch
    pub batch_size: usize,
    /// Optional limit on the number of rows to read
    pub limit: Option<usize>,
    /// Optional predicate to apply during the scan
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Optional pruning predicate applied to row group statistics
    pub pruning_predicate: Option<Arc<PruningPredicate>>,
    /// Optional pruning predicate applied to data page statistics
    pub page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    /// Schema of the output table
    pub table_schema: SchemaRef,
    /// Optional hint for how large the initial request to read parquet metadata
    /// should be
    pub metadata_size_hint: Option<usize>,
    /// Metrics for reporting
    pub metrics: ExecutionPlanMetricsSet,
    /// Factory for instantiating parquet reader
    pub parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
    /// Should the filters be evaluated during the parquet scan using
    /// [`DataFusionArrowPredicate`](row_filter::DatafusionArrowPredicate)?
    pub pushdown_filters: bool,
    /// Should the filters be reordered to optimize the scan?
    pub reorder_filters: bool,
    /// Should the page index be read from parquet files, if present, to skip
    /// data pages
    pub enable_page_index: bool,
    /// Should the bloom filter be read from parquet, if present, to skip row
    /// groups
    pub enable_bloom_filter: bool,
    /// Schema adapter factory
    pub schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
}

impl FileOpener for ParquetOpener {
    fn open(&self, file_meta: FileMeta) -> Result<FileOpenFuture> {
        let file_range = file_meta.range.clone();
        let extensions = file_meta.extensions.clone();
        let file_name = file_meta.location().to_string();
        println!("==> Opening file: {file_name}");
        let file_metrics =
            ParquetFileMetrics::new(self.partition_index, &file_name, &self.metrics);

        let metadata_size_hint = file_meta.metadata_size_hint.or(self.metadata_size_hint);

        let mut reader: Box<dyn AsyncFileReader> =
            self.parquet_file_reader_factory.create_reader(
                self.partition_index,
                file_meta,
                metadata_size_hint,
                &self.metrics,
            )?;
        println!("==> Created reader for file: {file_name}");

        let batch_size = self.batch_size;

        let projected_schema = match self.table_schema.project(&self.projection) {
            Ok(schema) => {
                println!("==> Successfully projected schema");
                SchemaRef::from(schema)
            }
            Err(e) => {
                println!("==> Error projecting schema: {e}");
                return Err(e.into());
            }
        };

        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, Arc::clone(&self.table_schema));
        println!("==> Created schema adapter");

        let predicate = self.predicate.clone();
        println!("==> Cloned predicate");

        let pruning_predicate = self.pruning_predicate.clone();
        println!("==> Cloned pruning predicate");

        let page_pruning_predicate = self.page_pruning_predicate.clone();
        println!("==> Cloned page pruning predicate");

        let table_schema = Arc::clone(&self.table_schema);
        println!("==> Cloned table schema");

        let reorder_predicates = self.reorder_filters;
        println!("==> Set reorder predicates");

        let pushdown_filters = self.pushdown_filters;
        println!("==> Set pushdown filters");

        let enable_page_index = should_enable_page_index(
            self.enable_page_index,
            &self.page_pruning_predicate,
        );
        println!("==> Determined if page index should be enabled");

        let enable_bloom_filter = self.enable_bloom_filter;
        println!("==> Set enable bloom filter");

        let limit = self.limit;
        println!("==> Set limit");

                let result = async move {
            let options = ArrowReaderOptions::new().with_page_index(enable_page_index);
        
            let mut metadata_timer = file_metrics.metadata_load_time.timer();
            let metadata = match ArrowReaderMetadata::load_async(&mut reader, options.clone()).await {
                Ok(metadata) => {
                    println!("==> Loaded metadata for file: {file_name}");
                    metadata
                }
                Err(e) => {
                    println!("==> Error loading metadata for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
            let mut schema = Arc::clone(metadata.schema());
        
            if let Some(merged) = coerce_file_schema_to_string_type(&table_schema, &schema) {
                schema = Arc::new(merged);
            }
        
            // read with view types
            if let Some(merged) = coerce_file_schema_to_view_type(&table_schema, &schema) {
                schema = Arc::new(merged);
            }
        
            let options = ArrowReaderOptions::new()
                .with_page_index(enable_page_index)
                .with_schema(Arc::clone(&schema));
            let metadata = match ArrowReaderMetadata::try_new(Arc::clone(metadata.metadata()), options) {
                Ok(metadata) => {
                    println!("==> Created ArrowReaderMetadata for file: {file_name}");
                    metadata
                }
                Err(e) => {
                    println!("==> Error creating ArrowReaderMetadata for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
        
            metadata_timer.stop();
        
            let mut builder = match ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata) {
                Ok(builder) => {
                    println!("==> Created ParquetRecordBatchStreamBuilder for file: {file_name}");
                    builder
                }
                Err(e) => {
                    println!("==> Error creating ParquetRecordBatchStreamBuilder for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
        
            let file_schema = Arc::clone(builder.schema());
        
            let (schema_mapping, adapted_projections) = match schema_adapter.map_schema(&file_schema) {
                Ok(result) => {
                    println!("==> Mapped schema for file: {file_name}");
                    result
                }
                Err(e) => {
                    println!("==> Error mapping schema for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
        
            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                adapted_projections.iter().cloned(),
            );
        
            // Filter pushdown: evaluate predicates during scan
            if let Some(predicate) = pushdown_filters.then_some(predicate).flatten() {
                let row_filter = row_filter::build_row_filter(
                    &predicate,
                    &file_schema,
                    &table_schema,
                    builder.metadata(),
                    reorder_predicates,
                    &file_metrics,
                    Arc::clone(&schema_mapping),
                );
        
                match row_filter {
                    Ok(Some(filter)) => {
                        builder = builder.with_row_filter(filter);
                        println!("==> Applied row filter for file: {file_name}");
                    }
                    Ok(None) => {}
                    Err(e) => {
                        println!("==> Ignoring error building row filter for '{:?}': {}", predicate, e);
                        debug!(
                            "Ignoring error building row filter for '{:?}': {}",
                            predicate, e
                        );
                    }
                };
            };
        
            // Determine which row groups to actually read. The idea is to skip
            // as many row groups as possible based on the metadata and query
            let file_metadata = Arc::clone(builder.metadata());
            let predicate = pruning_predicate.as_ref().map(|p| p.as_ref());
            let rg_metadata = file_metadata.row_groups();
            // track which row groups to actually read
            let access_plan = match create_initial_plan(&file_name, extensions, rg_metadata.len()) {
                Ok(plan) => {
                    println!("==> Created initial access plan for file: {file_name}");
                    plan
                }
                Err(e) => {
                    println!("==> Error creating initial access plan for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
            let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
            // if there is a range restricting what parts of the file to read
            if let Some(range) = file_range.as_ref() {
                row_groups.prune_by_range(rg_metadata, range);
                println!("==> Pruned row groups by range for file: {file_name}");
            }
            // If there is a predicate that can be evaluated against the metadata
            if let Some(predicate) = predicate.as_ref() {
                row_groups.prune_by_statistics(
                    &file_schema,
                    builder.parquet_schema(),
                    rg_metadata,
                    predicate,
                    &file_metrics,
                );
                println!("==> Pruned row groups by statistics for file: {file_name}");
        
                if enable_bloom_filter && !row_groups.is_empty() {
                    row_groups
                        .prune_by_bloom_filters(
                            &file_schema,
                            &mut builder,
                            predicate,
                            &file_metrics,
                        )
                        .await;
                    println!("==> Pruned row groups by bloom filters for file: {file_name}");
                }
            }
        
            let mut access_plan = row_groups.build();
            println!("==> Built access plan for file: {file_name}");
        
            // page index pruning: if all data on individual pages can
            // be ruled using page metadata, rows from other columns
            // with that range can be skipped as well
            if enable_page_index && !access_plan.is_empty() {
                if let Some(p) = page_pruning_predicate {
                    access_plan = p.prune_plan_with_page_index(
                        access_plan,
                        &file_schema,
                        builder.parquet_schema(),
                        file_metadata.as_ref(),
                        &file_metrics,
                    );
                    println!("==> Pruned access plan with page index for file: {file_name}");
                }
            }
        
            let row_group_indexes = access_plan.row_group_indexes();
            if let Some(row_selection) = access_plan.into_overall_row_selection(rg_metadata)? {
                builder = builder.with_row_selection(row_selection);
                println!("==> Applied row selection for file: {file_name}");
            }
        
            if let Some(limit) = limit {
                builder = builder.with_limit(limit);
                println!("==> Applied limit for file: {file_name}");
            }
        
            let stream = match builder
                .with_projection(mask)
                .with_batch_size(batch_size)
                .with_row_groups(row_group_indexes)
                .build() {
                Ok(stream) => {
                    println!("==> Built ParquetRecordBatchStream for file: {file_name}");
                    stream
                }
                Err(e) => {
                    println!("==> Error building ParquetRecordBatchStream for file: {file_name}, error: {e}");
                    return Err(e);
                }
            };
        
            let adapted = stream
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
                .map(move |maybe_batch| {
                    maybe_batch
                        .and_then(|b| schema_mapping.map_batch(b).map_err(Into::into))
                });
        
            println!("==> Successfully created adapted stream for file: {file_name}");
            Ok(adapted.boxed())
        }.await;
        
        result.map_err(Into::into)
    }
}

/// Return the initial [`ParquetAccessPlan`]
///
/// If the user has supplied one as an extension, use that
/// otherwise return a plan that scans all row groups
///
/// Returns an error if an invalid `ParquetAccessPlan` is provided
///
/// Note: file_name is only used for error messages
fn create_initial_plan(
    file_name: &str,
    extensions: Option<Arc<dyn std::any::Any + Send + Sync>>,
    row_group_count: usize,
) -> Result<ParquetAccessPlan> {
    if let Some(extensions) = extensions {
        if let Some(access_plan) = extensions.downcast_ref::<ParquetAccessPlan>() {
            let plan_len = access_plan.len();
            if plan_len != row_group_count {
                return exec_err!(
                    "Invalid ParquetAccessPlan for {file_name}. Specified {plan_len} row groups, but file has {row_group_count}"
                );
            }

            // check row group count matches the plan
            return Ok(access_plan.clone());
        } else {
            debug!("ParquetExec Ignoring unknown extension specified for {file_name}");
        }
    }

    // default to scanning all row groups
    Ok(ParquetAccessPlan::new_all(row_group_count))
}
