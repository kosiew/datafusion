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

//! Defines the Range Merge join execution plan.
//! A Range Merge join plan consumes two sorted input plans and produces
//! joined output based on non-equality conditions (<, <=, >, >=).

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::fmt::Formatter;
use std::mem::size_of;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::*;
use arrow::compute::{concat_batches, filter_record_batch, take, SortOptions};
use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

use datafusion_common::{
    exec_err, internal_err, not_impl_err, plan_err, DataFusionError, JoinSide, JoinType,
    Result, ScalarValue, ToDFSchema,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, LexRequirement};
use futures::{Stream, StreamExt};

use crate::execution_plan::{boundedness_from_children, EmissionType};
use crate::expressions::PhysicalSortExpr;
use crate::joins::utils::{
    build_join_schema, check_join_is_valid, estimate_join_statistics, JoinOn,
};
use crate::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, SpillMetrics,
};
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PhysicalExpr as _, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
    Statistics,
};

/// Supported comparison operators for range joins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOperator {
    /// <
    LessThan,
    /// <=
    LessThanOrEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterThanOrEqual,
}

impl RangeOperator {
    /// Convert from datafusion_expr::Operator
    pub fn from_operator(op: Operator) -> Option<Self> {
        match op {
            Operator::Lt => Some(RangeOperator::LessThan),
            Operator::LtEq => Some(RangeOperator::LessThanOrEqual),
            Operator::Gt => Some(RangeOperator::GreaterThan),
            Operator::GtEq => Some(RangeOperator::GreaterThanOrEqual),
            _ => None,
        }
    }

    /// Returns true if the comparison is satisfied
    pub fn compare<T: PartialOrd>(&self, left: &T, right: &T) -> bool {
        match self {
            RangeOperator::LessThan => left < right,
            RangeOperator::LessThanOrEqual => left <= right,
            RangeOperator::GreaterThan => left > right,
            RangeOperator::GreaterThanOrEqual => left >= right,
        }
    }

    /// Returns the required sort order for inputs
    pub fn required_sort_order(&self) -> SortOptions {
        // For < and <=, we want ascending order
        // For > and >=, we want descending order
        match self {
            RangeOperator::LessThan | RangeOperator::LessThanOrEqual => SortOptions {
                descending: false,
                nulls_first: false,
            },
            RangeOperator::GreaterThan | RangeOperator::GreaterThanOrEqual => {
                SortOptions {
                    descending: true,
                    nulls_first: true,
                }
            }
        }
    }

    /// Returns a string representation of the operator
    pub fn as_str(&self) -> &'static str {
        match self {
            RangeOperator::LessThan => "<",
            RangeOperator::LessThanOrEqual => "<=",
            RangeOperator::GreaterThan => ">",
            RangeOperator::GreaterThanOrEqual => ">=",
        }
    }
}

/// Join execution plan that executes range join predicates on sorted inputs
/// (e.g., `t1.col < t2.col`).
///
/// # Range Join Condition
///
/// Join predicate expressions are represented by [`Self::on`] and must be a
/// non-equality range condition (<, <=, >, >=).
///
/// # Sorting
///
/// This plan assumes that both inputs are sorted appropriately for the range
/// condition. Different operators require different sort orders:
/// - For < and <=: Ascending order
/// - For > and >=: Descending order
///
/// The implementation uses a streaming approach with two pointers to track and join
/// matching rows from left and right inputs based on the range condition.
///
/// # LIMIT optimization
///
/// Optionally supports early termination when a limit is specified, stopping the
/// emission of rows once the specified number of rows has been produced.
#[derive(Debug, Clone)]
pub struct RangeMergeJoinExec {
    /// Left sorted input execution plan
    left: Arc<dyn ExecutionPlan>,
    /// Right sorted input execution plan
    right: Arc<dyn ExecutionPlan>,
    /// Set of columns used to join with range condition
    on: JoinOn,
    /// Range operator to use for comparison (e.g., <, <=, >, >=)
    operator: RangeOperator,
    /// How the join is performed (currently only Inner is supported)
    join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The left sort expressions
    left_sort_exprs: LexOrdering,
    /// The right sort expressions
    right_sort_exprs: LexOrdering,
    /// Optional limit for early termination
    limit: Option<usize>,
    /// Cache holding plan properties
    cache: PlanProperties,
}

impl RangeMergeJoinExec {
    /// Tries to create a new [RangeMergeJoinExec].
    /// The inputs must be sorted appropriately for the range operator.
    ///
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        operator: RangeOperator,
        join_type: JoinType,
        limit: Option<usize>,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        // Currently only supporting INNER join type
        if join_type != JoinType::Inner {
            return not_impl_err!("RangeMergeJoinExec only supports JoinType::Inner");
        }

        // Currently only supporting one comparison column pair
        if on.len() != 1 {
            return plan_err!(
                "RangeMergeJoinExec currently only supports a single join condition"
            );
        }

        check_join_is_valid(&left_schema, &right_schema, &on)?;

        // Determine sort order based on the operator
        let sort_options = operator.required_sort_order();

        // Create sort expressions for the left and right inputs
        let (left_sort_exprs, right_sort_exprs): (Vec<_>, Vec<_>) = on
            .iter()
            .map(|((l, r))| {
                let left = PhysicalSortExpr {
                    expr: Arc::clone(l),
                    options: sort_options,
                };
                let right = PhysicalSortExpr {
                    expr: Arc::clone(r),
                    options: sort_options,
                };
                (left, right)
            })
            .unzip();

        let schema =
            Arc::new(build_join_schema(&left_schema, &right_schema, &join_type).0);
        let cache =
            Self::compute_properties(&left, &right, Arc::clone(&schema), join_type, &on);

        Ok(Self {
            left,
            right,
            on,
            operator,
            join_type,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            left_sort_exprs: LexOrdering::new(left_sort_exprs),
            right_sort_exprs: LexOrdering::new(right_sort_exprs),
            limit,
            cache,
        })
    }

    /// This function creates the cache object that stores the plan properties.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        _join_on: &JoinOn,
    ) -> PlanProperties {
        // For range joins, we conservatively say we don't preserve any ordering
        let maintains_input_order = vec![false, false];

        // Use a similar approach to SortMergeJoin for output partitioning
        let output_partitioning = match (
            left.output_partitioning().partition_count(),
            right.output_partitioning().partition_count(),
        ) {
            (1, 1) => left.output_partitioning().clone(),
            (_, 1) if join_type == JoinType::Left || join_type == JoinType::Inner => {
                left.output_partitioning().clone()
            }
            (1, _) if join_type == JoinType::Right || join_type == JoinType::Inner => {
                right.output_partitioning().clone()
            }
            _ => crate::physical_plan::partitioning::Partitioning::UnknownPartitioning(1),
        };

        PlanProperties::new(
            // For now, don't track any equivalence properties
            datafusion_physical_expr::equivalence::EquivalenceProperties::new(schema),
            output_partitioning,
            EmissionType::Incremental,
            boundedness_from_children([left, right]),
        )
    }

    /// Get the join type
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Get reference to the left input
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// Get reference to the right input
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of columns used to join on with range condition
    pub fn on(&self) -> &[(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)] {
        &self.on
    }

    /// Get the range operator
    pub fn operator(&self) -> RangeOperator {
        self.operator
    }

    /// Get the optional limit
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }
}

impl DisplayAs for RangeMergeJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({} {} {})", c1, self.operator.as_str(), c2))
                    .collect::<Vec<String>>()
                    .join(", ");
                let limit_str = self
                    .limit
                    .map_or("".to_string(), |l| format!(", limit={}", l));

                write!(
                    f,
                    "RangeMergeJoin: join_type={:?}, on=[{}]{}",
                    self.join_type, on, limit_str
                )
            }
            DisplayFormatType::TreeRender => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({} {} {})", c1, self.operator.as_str(), c2))
                    .collect::<Vec<String>>()
                    .join(", ");

                let limit_str = self
                    .limit
                    .map_or("".to_string(), |l| format!(", limit={}", l));
                write!(f, "on={}{}", on, limit_str)
            }
        }
    }
}

impl ExecutionPlan for RangeMergeJoinExec {
    fn name(&self) -> &'static str {
        "RangeMergeJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // For range joins, we need the data to be on the same partition
        vec![Distribution::SinglePartition, Distribution::SinglePartition]
    }

    fn required_input_ordering(&self) -> Vec<Option<LexRequirement>> {
        vec![
            Some(LexRequirement::from(self.left_sort_exprs.clone())),
            Some(LexRequirement::from(self.right_sort_exprs.clone())),
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Range joins don't preserve input order
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &children[..] {
            [left, right] => Ok(Arc::new(RangeMergeJoinExec::try_new(
                Arc::clone(left),
                Arc::clone(right),
                self.on.clone(),
                self.operator,
                self.join_type,
                self.limit,
            )?)),
            _ => internal_err!("RangeMergeJoinExec wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("RangeMergeJoinExec only supports a single partition");
        }

        // Get the column expressions to join on
        let (left_expr, right_expr) = match &self.on[..] {
            [(left_expr, right_expr)] => (Arc::clone(left_expr), Arc::clone(right_expr)),
            _ => unreachable!("RangeMergeJoinExec should only have one join condition"),
        };

        // Execute the children plans
        let left_stream = self.left.execute(partition, Arc::clone(&context))?;
        let right_stream = self.right.execute(partition, Arc::clone(&context))?;

        // Create output buffer with configured batch size
        let batch_size = context.session_config().batch_size();

        // Create memory reservation for the stream
        let reservation =
            MemoryConsumer::new(format!("RangeMergeJoinStream[{partition}]"))
                .register(context.memory_pool());

        // Create the range merge join stream
        Ok(Box::pin(RangeMergeJoinStream::try_new(
            self.schema.clone(),
            left_stream,
            right_stream,
            left_expr,
            right_expr,
            self.operator,
            batch_size,
            self.limit,
            RangeMergeJoinMetrics::new(partition, &self.metrics),
            reservation,
        )?))
    }

    fn statistics(&self) -> Result<Statistics> {
        // Use the same logic as SortMergeJoinExec for statistics
        estimate_join_statistics(
            Arc::clone(&self.left),
            Arc::clone(&self.right),
            self.on.clone(),
            &self.join_type,
            &self.schema,
        )
    }
}

/// Metrics for RangeMergeJoinExec
struct RangeMergeJoinMetrics {
    /// Total time for joining batches
    join_time: crate::metrics::Time,
    /// Number of batches consumed by this operator
    input_batches: Count,
    /// Number of rows consumed by this operator
    input_rows: Count,
    /// Number of batches produced by this operator
    output_batches: Count,
    /// Number of rows produced by this operator
    output_rows: Count,
}

impl RangeMergeJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);
        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);
        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            join_time,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

/// Stream implementing the range merge join operation
struct RangeMergeJoinStream {
    /// Output schema
    schema: SchemaRef,
    /// Left input stream
    left: SendableRecordBatchStream,
    /// Right input stream
    right: SendableRecordBatchStream,
    /// Left join column expression
    left_expr: Arc<dyn PhysicalExpr>,
    /// Right join column expression
    right_expr: Arc<dyn PhysicalExpr>,
    /// Range operator
    operator: RangeOperator,
    /// Target output batch size
    batch_size: usize,
    /// Optional limit for early termination
    limit: Option<usize>,
    /// Current state of the join process
    state: RangeMergeJoinState,
    /// Buffer for left batches
    left_batches: Vec<RecordBatch>,
    /// Buffer for right batches
    right_batches: Vec<RecordBatch>,
    /// Current index in left batch being processed
    left_idx: usize,
    /// Current index in right batch being processed
    right_idx: usize,
    /// Current left batch index
    left_batch_idx: usize,
    /// Current right batch index
    right_batch_idx: usize,
    /// Join result rows being accumulated
    result_rows: Vec<(usize, usize)>, // (left_idx, right_idx)
    /// Number of rows output so far (for LIMIT)
    rows_produced: usize,
    /// Metrics
    metrics: RangeMergeJoinMetrics,
    /// Memory reservation
    reservation: MemoryReservation,
}

/// State of range merge join stream
#[derive(Debug, PartialEq, Eq)]
enum RangeMergeJoinState {
    /// Reading inputs
    Reading,
    /// Producing output from buffered inputs
    Producing,
    /// No more data to process
    Finished,
}

impl RangeMergeJoinStream {
    pub fn try_new(
        schema: SchemaRef,
        left: SendableRecordBatchStream,
        right: SendableRecordBatchStream,
        left_expr: Arc<dyn PhysicalExpr>,
        right_expr: Arc<dyn PhysicalExpr>,
        operator: RangeOperator,
        batch_size: usize,
        limit: Option<usize>,
        metrics: RangeMergeJoinMetrics,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        Ok(Self {
            schema,
            left,
            right,
            left_expr,
            right_expr,
            operator,
            batch_size,
            limit,
            state: RangeMergeJoinState::Reading,
            left_batches: vec![],
            right_batches: vec![],
            left_idx: 0,
            right_idx: 0,
            left_batch_idx: 0,
            right_batch_idx: 0,
            result_rows: Vec::new(),
            rows_produced: 0,
            metrics,
            reservation,
        })
    }

    /// Evaluates the join key expressions on the given batch
    fn evaluate_expr(
        &self,
        batch: &RecordBatch,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Result<ArrayRef> {
        let num_rows = batch.num_rows();
        let result = expr.evaluate(batch)?;
        Ok(result.into_array(num_rows)?)
    }

    /// Returns true if we've produced enough rows to satisfy the LIMIT
    fn limit_reached(&self) -> bool {
        if let Some(limit) = self.limit {
            self.rows_produced >= limit
        } else {
            false
        }
    }

    /// Process the current left and right batches, finding matching rows
    fn process_batches(&mut self) -> Result<()> {
        // Clear previous results
        self.result_rows.clear();

        if self.left_batches.is_empty() || self.right_batches.is_empty() {
            return Ok(());
        }

        let left_batch = &self.left_batches[self.left_batch_idx];
        let right_batch = &self.right_batches[self.right_batch_idx];

        // Evaluate the join expressions for the current batches
        let left_values = self.evaluate_expr(left_batch, &self.left_expr)?;
        let right_values = self.evaluate_expr(right_batch, &self.right_expr)?;

        // Use a two-pointer approach to find matching rows
        let mut left_idx = self.left_idx;
        let mut right_idx = self.right_idx;

        // Processing logic depends on the operator
        match self.operator {
            RangeOperator::LessThan | RangeOperator::LessThanOrEqual => {
                // For < or <=, we scan right side for each left element
                while left_idx < left_batch.num_rows() && !self.limit_reached() {
                    // Process all right elements for current left element
                    while right_idx < right_batch.num_rows() && !self.limit_reached() {
                        if compare_arrays(
                            &left_values,
                            left_idx,
                            &right_values,
                            right_idx,
                            self.operator,
                        )? {
                            // Found a match, save it
                            self.result_rows.push((left_idx, right_idx));

                            // Stop if we've reached the limit
                            if let Some(limit) = self.limit {
                                if self.rows_produced + self.result_rows.len() >= limit {
                                    break;
                                }
                            }
                        } else {
                            // If left is not < right, then we can break the inner loop
                            // since right values are sorted in ascending order
                            break;
                        }
                        right_idx += 1;
                    }

                    // Move to next left row
                    left_idx += 1;

                    // Reset right_idx for the next left value if we haven't exhausted right
                    // For range joins, we need to rescan from beginning for each left value
                    right_idx = 0;
                }
            }
            RangeOperator::GreaterThan | RangeOperator::GreaterThanOrEqual => {
                // For > or >=, similar logic but remember inputs are in descending order
                while left_idx < left_batch.num_rows() && !self.limit_reached() {
                    // Process all right elements for current left element
                    while right_idx < right_batch.num_rows() && !self.limit_reached() {
                        if compare_arrays(
                            &left_values,
                            left_idx,
                            &right_values,
                            right_idx,
                            self.operator,
                        )? {
                            // Found a match, save it
                            self.result_rows.push((left_idx, right_idx));

                            // Stop if we've reached the limit
                            if let Some(limit) = self.limit {
                                if self.rows_produced + self.result_rows.len() >= limit {
                                    break;
                                }
                            }
                        } else {
                            // If left is not > right, then we can break the inner loop
                            break;
                        }
                        right_idx += 1;
                    }

                    // Move to next left row
                    left_idx += 1;

                    // Reset right_idx for the next left value
                    right_idx = 0;
                }
            }
        }

        // Update indices for next time
        self.left_idx = left_idx;
        self.right_idx = right_idx;

        // If we processed all rows in the left batch, move to the next left batch
        if left_idx >= left_batch.num_rows() {
            self.left_batch_idx += 1;
            self.left_idx = 0;
        }

        Ok(())
    }

    /// Creates a joined output batch from the collected result rows
    fn create_output_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.result_rows.is_empty() {
            return Ok(None);
        }

        let left_batch = &self.left_batches[self.left_batch_idx];
        let right_batch = &self.right_batches[self.right_batch_idx];

        // Create arrays of indices for left and right batches
        let left_indices: UInt64Array = UInt64Array::from_iter_values(
            self.result_rows.iter().map(|(l, _)| *l as u64),
        );
        let right_indices: UInt64Array = UInt64Array::from_iter_values(
            self.result_rows.iter().map(|(_, r)| *r as u64),
        );

        // For each column in left batch, take rows by left_indices
        let left_columns: Result<Vec<ArrayRef>, ArrowError> = left_batch
            .columns()
            .iter()
            .map(|col| take(col, &left_indices, None))
            .collect();

        // For each column in right batch, take rows by right_indices
        let right_columns: Result<Vec<ArrayRef>, ArrowError> = right_batch
            .columns()
            .iter()
            .map(|col| take(col, &right_indices, None))
            .collect();

        // Combine left and right columns
        let mut columns = left_columns?;
        columns.extend(right_columns?);

        // Create output batch
        let batch_size = self.result_rows.len();
        let output_batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;

        // Update metrics
        self.metrics.output_batches.add(1);
        self.metrics.output_rows.add(batch_size);
        self.rows_produced += batch_size;

        // Clear result rows after creating the batch
        self.result_rows.clear();

        Ok(Some(output_batch))
    }
}

impl RecordBatchStream for RangeMergeJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for RangeMergeJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let timer = self.metrics.join_time.timer();
        let mut timer = std::mem::ManuallyDrop::new(timer);

        loop {
            match self.state {
                RangeMergeJoinState::Reading => {
                    // Poll left stream
                    if self.left_batch_idx >= self.left_batches.len() {
                        match self.left.poll_next_unpin(cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                self.metrics.input_batches.add(1);
                                self.metrics.input_rows.add(batch.num_rows());
                                if batch.num_rows() > 0 {
                                    self.left_batches.push(batch);
                                }
                            }
                            Poll::Ready(Some(Err(e))) => {
                                return Poll::Ready(Some(Err(e)))
                            }
                            Poll::Ready(None) => {}
                            Poll::Pending => return Poll::Pending,
                        }
                    }

                    // Poll right stream
                    if self.right_batch_idx >= self.right_batches.len() {
                        match self.right.poll_next_unpin(cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                self.metrics.input_batches.add(1);
                                self.metrics.input_rows.add(batch.num_rows());
                                if batch.num_rows() > 0 {
                                    self.right_batches.push(batch);
                                }
                            }
                            Poll::Ready(Some(Err(e))) => {
                                return Poll::Ready(Some(Err(e)))
                            }
                            Poll::Ready(None) => {}
                            Poll::Pending => return Poll::Pending,
                        }
                    }

                    // Check if we've read all inputs
                    let left_done = self.left_batch_idx >= self.left_batches.len()
                        && self.left.is_exhausted();
                    let right_done = self.right_batch_idx >= self.right_batches.len()
                        && self.right.is_exhausted();

                    if left_done || right_done {
                        if self.left_batches.is_empty() || self.right_batches.is_empty() {
                            // No data to join
                            self.state = RangeMergeJoinState::Finished;
                        } else {
                            // Move to producing output
                            self.state = RangeMergeJoinState::Producing;
                        }
                    } else if self.left_batch_idx < self.left_batches.len()
                        && self.right_batch_idx < self.right_batches.len()
                    {
                        // We have data to process
                        self.state = RangeMergeJoinState::Producing;
                    }
                }
                RangeMergeJoinState::Producing => {
                    // Check if we've reached the limit
                    if self.limit_reached() {
                        self.state = RangeMergeJoinState::Finished;
                        continue;
                    }

                    // Process current batches to find matches
                    self.process_batches()?;

                    // Create output batch from collected results
                    if let Some(batch) = self.create_output_batch()? {
                        return Poll::Ready(Some(Ok(batch)));
                    }

                    // If no more results from current batches, move back to reading
                    self.state = RangeMergeJoinState::Reading;
                }
                RangeMergeJoinState::Finished => {
                    // Clean up any resources
                    self.left_batches.clear();
                    self.right_batches.clear();
                    self.result_rows.clear();

                    // Send end-of-stream
                    return Poll::Ready(None);
                }
            }
        }
    }
}

/// Compare values from two arrays using the given range operator
fn compare_arrays(
    left_array: &ArrayRef,
    left_idx: usize,
    right_array: &ArrayRef,
    right_idx: usize,
    operator: RangeOperator,
) -> Result<bool> {
    macro_rules! compare_values {
        ($array_type:ty) => {{
            let left_array = left_array.as_any().downcast_ref::<$array_type>().unwrap();
            let right_array = right_array.as_any().downcast_ref::<$array_type>().unwrap();

            if left_array.is_null(left_idx) || right_array.is_null(right_idx) {
                // Null values don't satisfy any range condition
                return Ok(false);
            }

            let left_value = left_array.value(left_idx);
            let right_value = right_array.value(right_idx);

            Ok(operator.compare(&left_value, &right_value))
        }};
    }

    match left_array.data_type() {
        DataType::Boolean => compare_values!(BooleanArray),
        DataType::Int8 => compare_values!(Int8Array),
        DataType::Int16 => compare_values!(Int16Array),
        DataType::Int32 => compare_values!(Int32Array),
        DataType::Int64 => compare_values!(Int64Array),
        DataType::UInt8 => compare_values!(UInt8Array),
        DataType::UInt16 => compare_values!(UInt16Array),
        DataType::UInt32 => compare_values!(UInt32Array),
        DataType::UInt64 => compare_values!(UInt64Array),
        DataType::Float32 => compare_values!(Float32Array),
        DataType::Float64 => compare_values!(Float64Array),
        DataType::Utf8 => compare_values!(StringArray),
        DataType::LargeUtf8 => compare_values!(LargeStringArray),
        DataType::Decimal128(..) => compare_values!(Decimal128Array),
        DataType::Date32 => compare_values!(Date32Array),
        DataType::Date64 => compare_values!(Date64Array),
        DataType::Timestamp(TimeUnit::Second, _) => {
            compare_values!(TimestampSecondArray)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            compare_values!(TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            compare_values!(TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            compare_values!(TimestampNanosecondArray)
        }
        other => not_impl_err!("Range join not implemented for data type: {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{build_table_i32, exec_plan};
    use arrow::datatypes::{Field, Schema};
    use datafusion_common::{assert_batches_eq, Result};
    use datafusion_physical_expr::expressions::Column;

    /// Helper function to build test tables
    fn create_test_table(
        a_values: &[i32],
        b_values: &[i32],
        c_values: &[i32],
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(
            ("a", &a_values.to_vec()),
            ("b", &b_values.to_vec()),
            ("c", &c_values.to_vec()),
        );

        crate::memory::MemoryExec::try_new(&[vec![batch]], batch.schema(), None).unwrap()
    }

    #[tokio::test]
    async fn test_range_merge_join_less_than() -> Result<()> {
        let left = create_test_table(
            &[1, 2, 3, 4, 5],
            &[5, 10, 15, 20, 25],
            &[100, 200, 300, 400, 500],
        );

        let right = create_test_table(
            &[10, 20, 30, 40, 50],
            &[6, 12, 18, 24, 30],
            &[1000, 2000, 3000, 4000, 5000],
        );

        let left_expr = Arc::new(Column::new("b", 0));
        let right_expr = Arc::new(Column::new("b", 0));

        let join_on = vec![(left_expr.clone(), right_expr.clone())];

        // Create range merge join with b < b
        let range_join = Arc::new(RangeMergeJoinExec::try_new(
            left,
            right,
            join_on,
            RangeOperator::LessThan,
            JoinType::Inner,
            None,
        )?);

        // Execute the plan
        let result = exec_plan(range_join).await?;

        let expected = vec![
            "+---+----+-----+----+----+------+",
            "| a | b  | c   | a  | b  | c    |",
            "+---+----+-----+----+----+------+",
            "| 1 | 5  | 100 | 10 | 6  | 1000 |",
            "| 1 | 5  | 100 | 20 | 12 | 2000 |",
            "| 1 | 5  | 100 | 30 | 18 | 3000 |",
            "| 1 | 5  | 100 | 40 | 24 | 4000 |",
            "| 1 | 5  | 100 | 50 | 30 | 5000 |",
            "| 2 | 10 | 200 | 20 | 12 | 2000 |",
            "| 2 | 10 | 200 | 30 | 18 | 3000 |",
            "| 2 | 10 | 200 | 40 | 24 | 4000 |",
            "| 2 | 10 | 200 | 50 | 30 | 5000 |",
            "| 3 | 15 | 300 | 30 | 18 | 3000 |",
            "| 3 | 15 | 300 | 40 | 24 | 4000 |",
            "| 3 | 15 | 300 | 50 | 30 | 5000 |",
            "| 4 | 20 | 400 | 40 | 24 | 4000 |",
            "| 4 | 20 | 400 | 50 | 30 | 5000 |",
            "| 5 | 25 | 500 | 50 | 30 | 5000 |",
            "+---+----+-----+----+----+------+",
        ];

        assert_batches_eq!(expected, &result);

        Ok(())
    }

    #[tokio::test]
    async fn test_range_merge_join_less_than_or_equal() -> Result<()> {
        let left = create_test_table(&[1, 2, 3], &[10, 20, 30], &[100, 200, 300]);

        let right = create_test_table(&[10, 20, 30], &[10, 20, 30], &[1000, 2000, 3000]);

        let left_expr = Arc::new(Column::new("b", 0));
        let right_expr = Arc::new(Column::new("b", 0));

        let join_on = vec![(left_expr.clone(), right_expr.clone())];

        // Create range merge join with b <= b
        let range_join = Arc::new(RangeMergeJoinExec::try_new(
            left,
            right,
            join_on,
            RangeOperator::LessThanOrEqual,
            JoinType::Inner,
            None,
        )?);

        // Execute the plan
        let result = exec_plan(range_join).await?;

        let expected = vec![
            "+---+----+-----+----+----+------+",
            "| a | b  | c   | a  | b  | c    |",
            "+---+----+-----+----+----+------+",
            "| 1 | 10 | 100 | 10 | 10 | 1000 |",
            "| 1 | 10 | 100 | 20 | 20 | 2000 |",
            "| 1 | 10 | 100 | 30 | 30 | 3000 |",
            "| 2 | 20 | 200 | 20 | 20 | 2000 |",
            "| 2 | 20 | 200 | 30 | 30 | 3000 |",
            "| 3 | 30 | 300 | 30 | 30 | 3000 |",
            "+---+----+-----+----+----+------+",
        ];

        assert_batches_eq!(expected, &result);

        Ok(())
    }

    #[tokio::test]
    async fn test_range_merge_join_greater_than() -> Result<()> {
        let left = create_test_table(
            &[5, 4, 3, 2, 1], // Note: values in descending order
            &[30, 24, 18, 12, 6],
            &[500, 400, 300, 200, 100],
        );

        let right = create_test_table(
            &[50, 40, 30, 20, 10], // Note: values in descending order
            &[25, 20, 15, 10, 5],
            &[5000, 4000, 3000, 2000, 1000],
        );

        let left_expr = Arc::new(Column::new("b", 0));
        let right_expr = Arc::new(Column::new("b", 0));

        let join_on = vec![(left_expr.clone(), right_expr.clone())];

        // Create range merge join with b > b
        let range_join = Arc::new(RangeMergeJoinExec::try_new(
            left,
            right,
            join_on,
            RangeOperator::GreaterThan,
            JoinType::Inner,
            None,
        )?);

        // Execute the plan
        let result = exec_plan(range_join).await?;

        let expected = vec![
            "+---+----+-----+----+----+------+",
            "| a | b  | c   | a  | b  | c    |",
            "+---+----+-----+----+----+------+",
            "| 5 | 30 | 500 | 50 | 25 | 5000 |",
            "| 5 | 30 | 500 | 40 | 20 | 4000 |",
            "| 5 | 30 | 500 | 30 | 15 | 3000 |",
            "| 5 | 30 | 500 | 20 | 10 | 2000 |",
            "| 5 | 30 | 500 | 10 | 5  | 1000 |",
            "| 4 | 24 | 400 | 40 | 20 | 4000 |",
            "| 4 | 24 | 400 | 30 | 15 | 3000 |",
            "| 4 | 24 | 400 | 20 | 10 | 2000 |",
            "| 4 | 24 | 400 | 10 | 5  | 1000 |",
            "| 3 | 18 | 300 | 30 | 15 | 3000 |",
            "| 3 | 18 | 300 | 20 | 10 | 2000 |",
            "| 3 | 18 | 300 | 10 | 5  | 1000 |",
            "| 2 | 12 | 200 | 20 | 10 | 2000 |",
            "| 2 | 12 | 200 | 10 | 5  | 1000 |",
            "| 1 | 6  | 100 | 10 | 5  | 1000 |",
            "+---+----+-----+----+----+------+",
        ];

        assert_batches_eq!(expected, &result);

        Ok(())
    }

    #[tokio::test]
    async fn test_range_merge_join_with_limit() -> Result<()> {
        let left = create_test_table(
            &[1, 2, 3, 4, 5],
            &[5, 10, 15, 20, 25],
            &[100, 200, 300, 400, 500],
        );

        let right = create_test_table(
            &[10, 20, 30, 40, 50],
            &[6, 12, 18, 24, 30],
            &[1000, 2000, 3000, 4000, 5000],
        );

        let left_expr = Arc::new(Column::new("b", 0));
        let right_expr = Arc::new(Column::new("b", 0));

        let join_on = vec![(left_expr.clone(), right_expr.clone())];

        // Create range merge join with b < b and a limit of 5
        let range_join = Arc::new(RangeMergeJoinExec::try_new(
            left,
            right,
            join_on,
            RangeOperator::LessThan,
            JoinType::Inner,
            Some(5),
        )?);

        // Execute the plan
        let result = exec_plan(range_join).await?;

        // Should only return the first 5 rows
        let expected = vec![
            "+---+---+-----+----+---+------+",
            "| a | b | c   | a  | b | c    |",
            "+---+---+-----+----+---+------+",
            "| 1 | 5 | 100 | 10 | 6 | 1000 |",
            "| 1 | 5 | 100 | 20 | 12| 2000 |",
            "| 1 | 5 | 100 | 30 | 18| 3000 |",
            "| 1 | 5 | 100 | 40 | 24| 4000 |",
            "| 1 | 5 | 100 | 50 | 30| 5000 |",
            "+---+---+-----+----+---+------+",
        ];

        assert_batches_eq!(expected, &result);

        Ok(())
    }
}
