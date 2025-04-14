// ... existing code ...

pub struct HashJoinExec {
    // ... existing fields ...
    /// Optional limit on the number of rows to produce
    limit: Option<usize>,

    /// Optional number of rows to skip
    skip: usize,
}

impl HashJoinExec {
    // ... existing code ...

    /// Set a limit on the number of rows to produce
    pub fn with_limit(mut self, limit: usize, skip: usize) -> Self {
        self.limit = Some(limit);
        self.skip = skip;
        self
    }

    // ... existing code ...
}

// Modify the execute method to respect the limit
impl ExecutionPlan for HashJoinExec {
    // ... existing code ...

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // ... existing code ...

        // Create a stream that is limit-aware
        let stream = HashJoinStream::new(
            left_stream,
            right_stream,
            self.on_left.clone(),
            self.on_right.clone(),
            self.filter.clone(),
            self.join_type,
            self.null_equals_null,
            self.schema.clone(),
            metrics,
            self.limit,
            self.skip,
        )?;

        Ok(Box::pin(stream))
    }

    // ... existing code ...
}

// Update the HashJoinStream to respect limits during execution
struct HashJoinStream {
    // ... existing fields ...
    /// Optional limit on the number of rows to produce
    limit: Option<usize>,

    /// Number of rows to skip
    skip: usize,

    /// Number of rows produced so far (after skipping)
    rows_produced: usize,
}

impl HashJoinStream {
    // ... existing code ...

    fn new(
        // ... existing parameters ...
        limit: Option<usize>,
        skip: usize,
    ) -> Result<Self> {
        // ... existing code ...

        Ok(Self {
            // ... existing fields ...
            limit,
            skip,
            rows_produced: 0,
        })
    }
}

impl Stream for HashJoinStream {
    // ... existing code ...

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        // Check if we've reached the limit
        if let Some(limit) = self.limit {
            if self.rows_produced >= limit {
                return Poll::Ready(None);
            }
        }

        // ... existing polling logic ...

        // When producing a batch, check if it exceeds the limit
        match self.process_batch(/* ... */) {
            Poll::Ready(Some(Ok(batch))) => {
                let batch_size = batch.num_rows();

                // Skip rows if needed
                if self.skip > 0 {
                    let rows_to_skip = self.skip.min(batch_size);
                    self.skip -= rows_to_skip;

                    if rows_to_skip == batch_size {
                        // Skip the entire batch
                        return self.poll_next(cx);
                    }

                    // Create a new batch with only the non-skipped rows
                    // ... skipping logic ...
                }

                // Apply limit if needed
                if let Some(limit) = self.limit {
                    let remaining = limit - self.rows_produced;
                    if batch_size > remaining {
                        // Truncate the batch to the remaining limit
                        // ... truncation logic ...
                    }
                    self.rows_produced += batch.num_rows();
                }

                Poll::Ready(Some(Ok(batch)))
            }
            other => other,
        }

        // ... existing code ...
    }
}
