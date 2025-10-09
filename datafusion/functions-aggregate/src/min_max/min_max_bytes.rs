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

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryBuilder, BinaryViewBuilder, BooleanArray,
    LargeBinaryBuilder, LargeStringBuilder, StringBuilder, StringViewBuilder,
};
use arrow::datatypes::DataType;
use datafusion_common::{internal_datafusion_err, internal_err, Result};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::apply_filter_as_nulls;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::mem::{size_of, take};
use std::ptr::NonNull;
use std::sync::Arc;

const LEARNING_WINDOW: usize = 3;
const DENSE_THRESHOLD_PPM: u32 = 550_000;
const SIMPLE_THRESHOLD_PPM: u32 = 120_000;
const MAX_GROWTH_EVENTS: u32 = 2;
const DENSE_TOUCHED_INLINE_CAPACITY: usize = 32;
const SIMPLE_SCRATCH_SIZE: usize = 8_192;
const SIMPLE_TOUCHED_INLINE_CAPACITY: usize = 32;

/// Implements fast Min/Max [`GroupsAccumulator`] for "bytes" types ([`StringArray`],
/// [`BinaryArray`], [`StringViewArray`], etc)
///
/// This implementation dispatches to the appropriate specialized code in
/// [`MinMaxBytesState`] based on data type and comparison function
///
/// [`StringArray`]: arrow::array::StringArray
/// [`BinaryArray`]: arrow::array::BinaryArray
/// [`StringViewArray`]: arrow::array::StringViewArray
#[derive(Debug)]
pub(crate) struct MinMaxBytesAccumulator {
    /// Inner data storage.
    inner: MinMaxBytesState,
    /// if true, is `MIN` otherwise is `MAX`
    is_min: bool,
}

impl MinMaxBytesAccumulator {
    /// Create a new accumulator for computing `min(val)`
    pub fn new_min(data_type: DataType) -> Self {
        Self {
            inner: MinMaxBytesState::new(data_type),
            is_min: true,
        }
    }

    /// Create a new accumulator fo computing `max(val)`
    pub fn new_max(data_type: DataType) -> Self {
        Self {
            inner: MinMaxBytesState::new(data_type),
            is_min: false,
        }
    }
}

impl GroupsAccumulator for MinMaxBytesAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let array = &values[0];
        assert_eq!(array.len(), group_indices.len());
        assert_eq!(array.data_type(), &self.inner.data_type);

        // apply filter if needed
        let array = apply_filter_as_nulls(array, opt_filter)?;

        // dispatch to appropriate kernel / specialized implementation
        fn string_min(a: &[u8], b: &[u8]) -> bool {
            // safety: only called from this function, which ensures a and b come
            // from an array with valid utf8 data
            unsafe {
                let a = std::str::from_utf8_unchecked(a);
                let b = std::str::from_utf8_unchecked(b);
                a < b
            }
        }
        fn string_max(a: &[u8], b: &[u8]) -> bool {
            // safety: only called from this function, which ensures a and b come
            // from an array with valid utf8 data
            unsafe {
                let a = std::str::from_utf8_unchecked(a);
                let b = std::str::from_utf8_unchecked(b);
                a > b
            }
        }
        fn binary_min(a: &[u8], b: &[u8]) -> bool {
            a < b
        }

        fn binary_max(a: &[u8], b: &[u8]) -> bool {
            a > b
        }

        fn str_to_bytes<'a>(
            it: impl Iterator<Item = Option<&'a str>>,
        ) -> impl Iterator<Item = Option<&'a [u8]>> {
            it.map(|s| s.map(|s| s.as_bytes()))
        }

        match (self.is_min, &self.inner.data_type) {
            // Utf8/LargeUtf8/Utf8View Min
            (true, &DataType::Utf8) => self.inner.update_batch(
                str_to_bytes(array.as_string::<i32>().iter()),
                group_indices,
                total_num_groups,
                string_min,
            ),
            (true, &DataType::LargeUtf8) => self.inner.update_batch(
                str_to_bytes(array.as_string::<i64>().iter()),
                group_indices,
                total_num_groups,
                string_min,
            ),
            (true, &DataType::Utf8View) => self.inner.update_batch(
                str_to_bytes(array.as_string_view().iter()),
                group_indices,
                total_num_groups,
                string_min,
            ),

            // Utf8/LargeUtf8/Utf8View Max
            (false, &DataType::Utf8) => self.inner.update_batch(
                str_to_bytes(array.as_string::<i32>().iter()),
                group_indices,
                total_num_groups,
                string_max,
            ),
            (false, &DataType::LargeUtf8) => self.inner.update_batch(
                str_to_bytes(array.as_string::<i64>().iter()),
                group_indices,
                total_num_groups,
                string_max,
            ),
            (false, &DataType::Utf8View) => self.inner.update_batch(
                str_to_bytes(array.as_string_view().iter()),
                group_indices,
                total_num_groups,
                string_max,
            ),

            // Binary/LargeBinary/BinaryView Min
            (true, &DataType::Binary) => self.inner.update_batch(
                array.as_binary::<i32>().iter(),
                group_indices,
                total_num_groups,
                binary_min,
            ),
            (true, &DataType::LargeBinary) => self.inner.update_batch(
                array.as_binary::<i64>().iter(),
                group_indices,
                total_num_groups,
                binary_min,
            ),
            (true, &DataType::BinaryView) => self.inner.update_batch(
                array.as_binary_view().iter(),
                group_indices,
                total_num_groups,
                binary_min,
            ),

            // Binary/LargeBinary/BinaryView Max
            (false, &DataType::Binary) => self.inner.update_batch(
                array.as_binary::<i32>().iter(),
                group_indices,
                total_num_groups,
                binary_max,
            ),
            (false, &DataType::LargeBinary) => self.inner.update_batch(
                array.as_binary::<i64>().iter(),
                group_indices,
                total_num_groups,
                binary_max,
            ),
            (false, &DataType::BinaryView) => self.inner.update_batch(
                array.as_binary_view().iter(),
                group_indices,
                total_num_groups,
                binary_max,
            ),

            _ => internal_err!(
                "Unexpected combination for MinMaxBytesAccumulator: ({:?}, {:?})",
                self.is_min,
                self.inner.data_type
            ),
        }
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let (data_capacity, min_maxes) = self.inner.emit_to(emit_to);

        // Convert the Vec of bytes to a vec of Strings (at no cost)
        fn bytes_to_str(
            min_maxes: Vec<Option<Vec<u8>>>,
        ) -> impl Iterator<Item = Option<String>> {
            min_maxes.into_iter().map(|opt| {
                opt.map(|bytes| {
                    // Safety: only called on data added from update_batch which ensures
                    // the input type matched the output type
                    unsafe { String::from_utf8_unchecked(bytes) }
                })
            })
        }

        let result: ArrayRef = match self.inner.data_type {
            DataType::Utf8 => {
                let mut builder =
                    StringBuilder::with_capacity(min_maxes.len(), data_capacity);
                for opt in bytes_to_str(min_maxes) {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_str()),
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeUtf8 => {
                let mut builder =
                    LargeStringBuilder::with_capacity(min_maxes.len(), data_capacity);
                for opt in bytes_to_str(min_maxes) {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_str()),
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Utf8View => {
                let block_size = capacity_to_view_block_size(data_capacity);

                let mut builder = StringViewBuilder::with_capacity(min_maxes.len())
                    .with_fixed_block_size(block_size);
                for opt in bytes_to_str(min_maxes) {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_str()),
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Binary => {
                let mut builder =
                    BinaryBuilder::with_capacity(min_maxes.len(), data_capacity);
                for opt in min_maxes {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_ref() as &[u8]),
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeBinary => {
                let mut builder =
                    LargeBinaryBuilder::with_capacity(min_maxes.len(), data_capacity);
                for opt in min_maxes {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_ref() as &[u8]),
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::BinaryView => {
                let block_size = capacity_to_view_block_size(data_capacity);

                let mut builder = BinaryViewBuilder::with_capacity(min_maxes.len())
                    .with_fixed_block_size(block_size);
                for opt in min_maxes {
                    match opt {
                        None => builder.append_null(),
                        Some(s) => builder.append_value(s.as_ref() as &[u8]),
                    }
                }
                Arc::new(builder.finish())
            }
            _ => {
                return internal_err!(
                    "Unexpected data type for MinMaxBytesAccumulator: {:?}",
                    self.inner.data_type
                );
            }
        };

        assert_eq!(&self.inner.data_type, result.data_type());
        Ok(result)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        // min/max are their own states (no transition needed)
        self.evaluate(emit_to).map(|arr| vec![arr])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // min/max are their own states (no transition needed)
        self.update_batch(values, group_indices, opt_filter, total_num_groups)
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        // Min/max do not change the values as they are their own states
        // apply the filter by combining with the null mask, if any
        let output = apply_filter_as_nulls(&values[0], opt_filter)?;
        Ok(vec![output])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Returns the block size in (contiguous buffer size) to use
/// for a given data capacity (total string length)
///
/// This is a heuristic to avoid allocating too many small buffers
fn capacity_to_view_block_size(data_capacity: usize) -> u32 {
    let max_block_size = 2 * 1024 * 1024;
    // Avoid block size equal to zero when calling `with_fixed_block_size()`.
    if data_capacity == 0 {
        return 1;
    }
    if let Ok(block_size) = u32::try_from(data_capacity) {
        block_size.min(max_block_size)
    } else {
        max_block_size
    }
}

/// Stores internal Min/Max state for "bytes" types.
///
/// This implementation is general and stores the minimum/maximum for each
/// groups in an individual byte array, which balances allocations and memory
/// fragmentation (aka garbage).
///
/// ```text
///                    ┌─────────────────────────────────┐
///   ┌─────┐    ┌────▶│Option<Vec<u8>> (["A"])          │───────────▶   "A"
///   │  0  │────┘     └─────────────────────────────────┘
///   ├─────┤          ┌─────────────────────────────────┐
///   │  1  │─────────▶│Option<Vec<u8>> (["Z"])          │───────────▶   "Z"
///   └─────┘          └─────────────────────────────────┘               ...
///     ...               ...
///   ┌─────┐          ┌────────────────────────────────┐
///   │ N-2 │─────────▶│Option<Vec<u8>> (["A"])         │────────────▶   "A"
///   ├─────┤          └────────────────────────────────┘
///   │ N-1 │────┐     ┌────────────────────────────────┐
///   └─────┘    └────▶│Option<Vec<u8>> (["Q"])         │────────────▶   "Q"
///                    └────────────────────────────────┘
///
///                      min_max: Vec<Option<Vec<u8>>
/// ```
///
/// Note that for `StringViewArray` and `BinaryViewArray`, there are potentially
/// more efficient implementations (e.g. by managing a string data buffer
/// directly), but then garbage collection, memory management, and final array
/// construction becomes more complex.
///
/// See discussion on <https://github.com/apache/datafusion/issues/6906>
#[derive(Clone, Copy)]
struct DenseScratchSlot {
    epoch_and_flags: u32,
    batch_index: u32,
}

#[derive(Debug, Clone, Copy)]
enum DenseLocation {
    Untouched,
    Existing,
    BatchIndex(u32),
}

impl DenseScratchSlot {
    const LOCATION_BITS: u32 = 2;
    const LOCATION_MASK: u32 = (1 << Self::LOCATION_BITS) - 1;
    const FLAG_UNTOUCHED: u32 = 0;
    const FLAG_EXISTING: u32 = 1;
    const FLAG_BATCH_INDEX: u32 = 2;
    const MAX_EPOCH: u32 = u32::MAX >> Self::LOCATION_BITS;

    const fn new() -> Self {
        Self {
            epoch_and_flags: 0,
            batch_index: 0,
        }
    }

    fn reset(&mut self) {
        self.epoch_and_flags = 0;
        self.batch_index = 0;
    }

    fn epoch(&self) -> u32 {
        self.epoch_and_flags >> Self::LOCATION_BITS
    }

    fn set_epoch(&mut self, epoch: u32) {
        let epoch = epoch & Self::MAX_EPOCH;
        self.epoch_and_flags =
            (epoch << Self::LOCATION_BITS) | (self.epoch_and_flags & Self::LOCATION_MASK);
    }

    fn location_flag(&self) -> u32 {
        self.epoch_and_flags & Self::LOCATION_MASK
    }

    fn set_location_flag(&mut self, flag: u32) {
        self.epoch_and_flags =
            (self.epoch_and_flags & !Self::LOCATION_MASK) | (flag & Self::LOCATION_MASK);
    }

    fn location(&self) -> DenseLocation {
        match self.location_flag() {
            Self::FLAG_UNTOUCHED => DenseLocation::Untouched,
            Self::FLAG_EXISTING => DenseLocation::Existing,
            Self::FLAG_BATCH_INDEX => DenseLocation::BatchIndex(self.batch_index),
            _ => {
                debug_assert!(false, "invalid dense scratch slot location flag");
                DenseLocation::Untouched
            }
        }
    }

    fn set_location(&mut self, location: DenseLocation) {
        match location {
            DenseLocation::Untouched => {
                self.set_location_flag(Self::FLAG_UNTOUCHED);
            }
            DenseLocation::Existing => {
                self.set_location_flag(Self::FLAG_EXISTING);
            }
            DenseLocation::BatchIndex(idx) => {
                self.batch_index = idx;
                self.set_location_flag(Self::FLAG_BATCH_INDEX);
            }
        }
    }
}

impl Default for DenseScratchSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DenseScratchSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseScratchSlot")
            .field("epoch", &self.epoch())
            .field("location", &self.location())
            .finish()
    }
}

#[derive(Debug, Default, Clone)]
struct SimpleTrackedSlot {
    epoch: u64,
    group_index: usize,
    has_candidate: bool,
    is_collision: bool,
    candidate: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
struct OverflowCandidate {
    group_index: usize,
    candidate: Vec<u8>,
    has_candidate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadMode {
    DenseInline,
    SimpleTracked,
    SparseHashMap,
    Undecided,
}

#[derive(Debug, Default, Clone, Copy)]
struct BatchStats {
    unique_groups: usize,
    total_num_groups: usize,
    _rows_processed: usize,
    density_ppm: u32,
}

#[derive(Debug)]
struct AdaptiveState {
    density_history: [u32; LEARNING_WINDOW],
    unique_history: [usize; LEARNING_WINDOW],
    total_history: [usize; LEARNING_WINDOW],
    window_idx: usize,
    batches_seen: u32,
    sum_unique_groups: u64,
    sum_total_groups: u64,
    growth_events: u32,
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self {
            density_history: [0; LEARNING_WINDOW],
            unique_history: [0; LEARNING_WINDOW],
            total_history: [0; LEARNING_WINDOW],
            window_idx: 0,
            batches_seen: 0,
            sum_unique_groups: 0,
            sum_total_groups: 0,
            growth_events: 0,
        }
    }
}

#[derive(Debug)]
struct MinMaxBytesState {
    /// The minimum/maximum value for each group
    min_max: Vec<Option<Vec<u8>>>,
    /// The data type of the array
    data_type: DataType,
    /// The total bytes of the string data (for pre-allocating the final array,
    /// and tracking memory usage)
    total_data_bytes: usize,
    /// Dense mode scratch buffer reused across batches
    dense_scratch: Vec<DenseScratchSlot>,
    /// Monotonically increasing epoch used to lazily reset [`dense_scratch`]
    dense_epoch: u32,
    /// Groups touched in the current batch (used to apply updates)
    dense_touched_groups: SmallVec<[usize; DENSE_TOUCHED_INLINE_CAPACITY]>,
    /// Scratch buffer for SimpleTracked mode
    simple_scratch: Vec<SimpleTrackedSlot>,
    /// Epoch used to lazily reset [`simple_scratch`]
    simple_epoch: u64,
    /// Slots touched in current batch for SimpleTracked mode
    simple_touched_slots: SmallVec<[usize; SIMPLE_TOUCHED_INLINE_CAPACITY]>,
    /// Groups whose hashed slots collided and must be processed via overflow handling
    simple_overflow: HashSet<usize>,
    /// Candidates retained for overflow groups
    simple_overflow_candidates: Vec<OverflowCandidate>,
    /// Current adaptive mode decision
    mode: WorkloadMode,
    /// Statistics captured while learning workload density
    adaptive: AdaptiveState,
}

/// Implement the MinMaxBytesAccumulator with a comparison function
/// for comparing strings
impl MinMaxBytesState {
    /// Create a new MinMaxBytesAccumulator
    ///
    /// # Arguments:
    /// * `data_type`: The data type of the arrays that will be passed to this accumulator
    fn new(data_type: DataType) -> Self {
        Self {
            min_max: vec![],
            data_type,
            total_data_bytes: 0,
            dense_scratch: Vec::new(),
            dense_epoch: 0,
            dense_touched_groups: SmallVec::new(),
            simple_scratch: Vec::new(),
            simple_epoch: 0,
            simple_touched_slots: SmallVec::new(),
            simple_overflow: HashSet::new(),
            simple_overflow_candidates: Vec::new(),
            mode: WorkloadMode::Undecided,
            adaptive: AdaptiveState::default(),
        }
    }

    /// Set the specified group to the given value, updating memory usage appropriately
    fn set_value(&mut self, group_index: usize, new_val: &[u8]) {
        match self.min_max[group_index].as_mut() {
            None => {
                self.min_max[group_index] = Some(new_val.to_vec());
                self.total_data_bytes += new_val.len();
            }
            Some(existing_val) => {
                // Copy data over to avoid re-allocating
                self.total_data_bytes -= existing_val.len();
                self.total_data_bytes += new_val.len();
                existing_val.clear();
                existing_val.extend_from_slice(new_val);
            }
        }
    }

    /// Updates the min/max values for the given string values
    ///
    /// `cmp` is the  comparison function to use, called like `cmp(new_val, existing_val)`
    /// returns true if the `new_val` should replace `existing_val`
    fn update_batch<'a, F, I>(
        &mut self,
        iter: I,
        group_indices: &[usize],
        total_num_groups: usize,
        mut cmp: F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
        I: IntoIterator<Item = Option<&'a [u8]>>,
    {
        let batch_values: Vec<Option<&[u8]>> = iter.into_iter().collect();
        if batch_values.len() != group_indices.len() {
            return internal_err!(
                "MinMaxBytesState::update_batch received mismatched lengths: values={}, indices={}",
                batch_values.len(),
                group_indices.len()
            );
        }

        let is_monotonic = is_strictly_monotonic(group_indices);
        let stats = self.analyze_batch(group_indices, total_num_groups);
        self.record_batch_stats(&stats);
        self.evaluate_transition(&stats);

        if is_monotonic {
            return self.update_batch_monotonic(
                &batch_values,
                group_indices,
                total_num_groups,
                &mut cmp,
            );
        }

        let expected_dense_touched = self.estimate_dense_touched_capacity(&stats);

        match self.mode {
            WorkloadMode::DenseInline => self.update_batch_dense_inline(
                &batch_values,
                group_indices,
                total_num_groups,
                expected_dense_touched,
                &mut cmp,
            ),
            WorkloadMode::SimpleTracked => self.update_batch_simple_tracked(
                &batch_values,
                group_indices,
                total_num_groups,
                expected_dense_touched,
                &mut cmp,
            ),
            WorkloadMode::SparseHashMap => self.update_batch_sparse_hash_map(
                &batch_values,
                group_indices,
                total_num_groups,
                expected_dense_touched,
                &mut cmp,
            ),
            WorkloadMode::Undecided => self.update_batch_dense_inline(
                &batch_values,
                group_indices,
                total_num_groups,
                expected_dense_touched,
                &mut cmp,
            ),
        }
    }

    fn update_batch_monotonic<'a, F>(
        &mut self,
        batch_values: &[Option<&'a [u8]>],
        group_indices: &[usize],
        total_num_groups: usize,
        cmp: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    {
        if self.min_max.len() < total_num_groups {
            self.min_max.resize(total_num_groups, None);
        }

        let mut current_group: Option<usize> = None;
        let mut current_best: Option<&[u8]> = None;
        let mut batch_candidate: Option<&'a [u8]> = None;

        for (row_idx, &group_index) in group_indices.iter().enumerate() {
            let Some(new_val) = batch_values[row_idx] else {
                continue;
            };

            if current_group != Some(group_index) {
                if let Some(prev_group) = current_group {
                    if let Some(candidate) = batch_candidate {
                        self.set_value(prev_group, candidate);
                    }
                }

                if group_index >= self.min_max.len() {
                    return internal_err!(
                        "monotonic batch received group index {} beyond allocated {}",
                        group_index,
                        self.min_max.len()
                    );
                }

                current_group = Some(group_index);
                current_best = self.min_max[group_index]
                    .as_ref()
                    .map(|existing| existing.as_slice());
                batch_candidate = None;
            }

            let should_replace = match current_best {
                Some(existing) => cmp(new_val, existing),
                None => true,
            };

            if should_replace {
                current_best = Some(new_val);
                batch_candidate = Some(new_val);
            }
        }

        if let Some(group_index) = current_group {
            if let Some(candidate) = batch_candidate {
                self.set_value(group_index, candidate);
            }
        }

        Ok(())
    }

    fn update_batch_dense_inline<'a, F>(
        &mut self,
        batch_values: &[Option<&'a [u8]>],
        group_indices: &[usize],
        total_num_groups: usize,
        expected_touched: usize,
        cmp: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    {
        if self.min_max.len() < total_num_groups {
            self.min_max.resize(total_num_groups, None);
        }

        if let Some(&max_group_in_batch) = group_indices.iter().max() {
            let Some(required_len) = max_group_in_batch.checked_add(1) else {
                return internal_err!("group index overflow in dense accumulator");
            };

            if self.dense_scratch.len() < required_len {
                let Some(new_size) = required_len.checked_next_power_of_two() else {
                    return internal_err!("group index overflow in dense accumulator");
                };

                self.dense_scratch
                    .resize(new_size, DenseScratchSlot::default());
            }
        }

        self.reserve_dense_touched_capacity(expected_touched);
        self.dense_epoch =
            (self.dense_epoch.wrapping_add(1)) & DenseScratchSlot::MAX_EPOCH;
        if self.dense_epoch == 0 {
            for slot in &mut self.dense_scratch {
                slot.reset();
            }
            self.dense_epoch = 1;
        }
        self.dense_touched_groups.clear();

        let mut last_group_index = usize::MAX;
        let mut last_slot: Option<NonNull<DenseScratchSlot>> = None;
        let mut last_location = DenseLocation::Untouched;

        for (row_idx, &group_index) in group_indices.iter().enumerate() {
            let Some(new_val) = batch_values[row_idx] else {
                continue;
            };

            let row_idx_u32 = u32::try_from(row_idx).map_err(|_| {
                internal_datafusion_err!("dense accumulator batch index overflow")
            })?;

            if group_index == last_group_index {
                if let Some(mut slot_ptr) = last_slot {
                    // SAFETY: `dense_scratch` is not resized within the loop, so the
                    // previously stored pointer remains valid for the duration of the
                    // batch processing.
                    let slot = unsafe { slot_ptr.as_mut() };
                    match last_location {
                        DenseLocation::BatchIndex(prev_idx) => {
                            let prev_val = batch_values[prev_idx as usize].expect(
                                "dense slot batch index should reference a value",
                            );
                            if cmp(new_val, prev_val) {
                                last_location = DenseLocation::BatchIndex(row_idx_u32);
                                slot.set_location(last_location);
                            }
                        }
                        DenseLocation::Existing => {
                            let existing =
                                self.min_max[group_index].as_ref().map(|v| v.as_ref());
                            if existing
                                .map(|existing_val| cmp(new_val, existing_val))
                                .unwrap_or_else(|| {
                                    debug_assert!(
                                        false,
                                        "dense slot in Existing state without value"
                                    );
                                    true
                                })
                            {
                                last_location = DenseLocation::BatchIndex(row_idx_u32);
                                slot.set_location(last_location);
                            }
                        }
                        DenseLocation::Untouched => {
                            debug_assert!(
                                false,
                                "dense slot cache hit before initialization"
                            );
                        }
                    }
                } else {
                    debug_assert!(
                        false,
                        "dense slot cache missing pointer for consecutive group"
                    );
                }
                continue;
            }

            let slot = &mut self.dense_scratch[group_index];

            if slot.epoch() != self.dense_epoch {
                slot.set_epoch(self.dense_epoch);
                self.dense_touched_groups.push(group_index);

                match self.min_max[group_index].as_ref() {
                    None => {
                        slot.set_location(DenseLocation::BatchIndex(row_idx_u32));
                    }
                    Some(existing) => {
                        if cmp(new_val, existing.as_ref()) {
                            slot.set_location(DenseLocation::BatchIndex(row_idx_u32));
                        } else {
                            slot.set_location(DenseLocation::Existing);
                        }
                    }
                }
            } else {
                match slot.location() {
                    DenseLocation::Untouched => {
                        debug_assert!(false, "dense slot touched without epoch reset");
                        slot.set_location(DenseLocation::BatchIndex(row_idx_u32));
                    }
                    DenseLocation::Existing => {
                        let existing = self.min_max[group_index]
                            .as_ref()
                            .expect("existing value missing for dense slot");
                        if cmp(new_val, existing.as_ref()) {
                            slot.set_location(DenseLocation::BatchIndex(row_idx_u32));
                        }
                    }
                    DenseLocation::BatchIndex(prev_idx) => {
                        let prev_val = batch_values[prev_idx as usize]
                            .expect("previous batch value missing for dense slot");
                        if cmp(new_val, prev_val) {
                            slot.set_location(DenseLocation::BatchIndex(row_idx_u32));
                        }
                    }
                }
            }

            last_group_index = group_index;
            last_location = slot.location();
            last_slot = Some(NonNull::from(slot));
        }

        let touched_len = self.dense_touched_groups.len();
        for idx in 0..touched_len {
            let group_index = self.dense_touched_groups[idx];
            let location = self.dense_scratch[group_index].location();
            if let DenseLocation::BatchIndex(idx) = location {
                let value = batch_values[idx as usize]
                    .expect("batch value missing when applying dense update");
                self.set_value(group_index, value);
            }
        }
        self.dense_touched_groups.clear();

        Ok(())
    }

    fn update_batch_simple_tracked<'a, F>(
        &mut self,
        batch_values: &[Option<&'a [u8]>],
        group_indices: &[usize],
        total_num_groups: usize,
        _expected_dense_touched: usize,
        cmp: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    {
        if self.min_max.len() < total_num_groups {
            self.min_max.resize(total_num_groups, None);
        }

        if self.simple_scratch.len() != SIMPLE_SCRATCH_SIZE {
            self.simple_scratch = vec![SimpleTrackedSlot::default(); SIMPLE_SCRATCH_SIZE];
            self.simple_epoch = 1;
        } else {
            self.simple_epoch = self.simple_epoch.wrapping_add(1);
            if self.simple_epoch == 0 {
                self.simple_epoch = 1;
                for slot in &mut self.simple_scratch {
                    slot.epoch = 0;
                }
            }
        }

        self.simple_touched_slots.clear();
        self.simple_overflow.clear();

        for (row_idx, &group_index) in group_indices.iter().enumerate() {
            let Some(new_val) = batch_values[row_idx] else {
                continue;
            };

            if group_index >= self.min_max.len() {
                return internal_err!(
                    "simple tracked received group index {} beyond allocated {}",
                    group_index,
                    self.min_max.len()
                );
            }

            let slot_index = group_index % SIMPLE_SCRATCH_SIZE;
            let slot = &mut self.simple_scratch[slot_index];

            if slot.epoch != self.simple_epoch {
                slot.epoch = self.simple_epoch;
                slot.group_index = group_index;
                slot.has_candidate = false;
                slot.is_collision = false;
                slot.candidate.clear();
                self.simple_touched_slots.push(slot_index);

                let should_replace = match self.min_max[group_index].as_ref() {
                    None => true,
                    Some(existing) => cmp(new_val, existing.as_ref()),
                };

                if should_replace {
                    slot.candidate.extend_from_slice(new_val);
                    slot.has_candidate = true;
                }

                continue;
            }

            if slot.is_collision {
                self.simple_overflow.insert(group_index);
                continue;
            }

            if slot.group_index == group_index {
                if slot.has_candidate {
                    if cmp(new_val, &slot.candidate) {
                        slot.candidate.clear();
                        slot.candidate.extend_from_slice(new_val);
                    }
                } else {
                    let should_replace = match self.min_max[group_index].as_ref() {
                        None => true,
                        Some(existing) => cmp(new_val, existing.as_ref()),
                    };

                    if should_replace {
                        slot.candidate.clear();
                        slot.candidate.extend_from_slice(new_val);
                        slot.has_candidate = true;
                    }
                }
            } else {
                slot.is_collision = true;
                self.simple_overflow.insert(slot.group_index);
                self.simple_overflow.insert(group_index);
                slot.candidate.clear();
                slot.has_candidate = false;
            }
        }

        let touched_len = self.simple_touched_slots.len();
        for idx in 0..touched_len {
            let slot_index = self.simple_touched_slots[idx];
            if {
                let slot = &self.simple_scratch[slot_index];
                slot.epoch != self.simple_epoch
                    || slot.is_collision
                    || !slot.has_candidate
            } {
                continue;
            }

            let (group_index, candidate) = {
                let slot = &mut self.simple_scratch[slot_index];
                slot.has_candidate = false;
                (slot.group_index, take(&mut slot.candidate))
            };

            self.set_value(group_index, &candidate);
            self.simple_scratch[slot_index].candidate = candidate;
        }

        let overflow_len = self.simple_overflow.len();
        if overflow_len > 0 {
            if self.simple_overflow_candidates.len() < overflow_len {
                let additional = overflow_len - self.simple_overflow_candidates.len();
                self.simple_overflow_candidates.reserve(additional);
                for _ in 0..additional {
                    self.simple_overflow_candidates
                        .push(OverflowCandidate::default());
                }
            }

            let mut idx = 0;
            for &group_index in &self.simple_overflow {
                let entry = &mut self.simple_overflow_candidates[idx];
                entry.group_index = group_index;
                entry.has_candidate = false;
                entry.candidate.clear();
                idx += 1;
            }

            for (row_idx, &group_index) in group_indices.iter().enumerate() {
                if !self.simple_overflow.contains(&group_index) {
                    continue;
                }

                let Some(new_val) = batch_values[row_idx] else {
                    continue;
                };

                let entry = self.simple_overflow_candidates[..overflow_len]
                    .iter_mut()
                    .find(|entry| entry.group_index == group_index)
                    .expect("overflow entry missing");

                if entry.has_candidate {
                    if cmp(new_val, &entry.candidate) {
                        entry.candidate.clear();
                        entry.candidate.extend_from_slice(new_val);
                    }
                } else {
                    let should_replace = match self.min_max[group_index].as_ref() {
                        None => true,
                        Some(existing) => cmp(new_val, existing.as_ref()),
                    };

                    if should_replace {
                        entry.candidate.clear();
                        entry.candidate.extend_from_slice(new_val);
                        entry.has_candidate = true;
                    }
                }
            }

            for idx in 0..overflow_len {
                if !self.simple_overflow_candidates[idx].has_candidate {
                    continue;
                }

                let (group_index, candidate) = {
                    let entry = &mut self.simple_overflow_candidates[idx];
                    entry.has_candidate = false;
                    (entry.group_index, take(&mut entry.candidate))
                };

                self.set_value(group_index, &candidate);
                self.simple_overflow_candidates[idx].candidate = candidate;
            }

            self.simple_overflow.clear();
        }

        Ok(())
    }

    fn update_batch_sparse_hash_map<'a, F>(
        &mut self,
        batch_values: &[Option<&'a [u8]>],
        group_indices: &[usize],
        total_num_groups: usize,
        expected_dense_touched: usize,
        cmp: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    {
        // SparseHashMap mode is not yet implemented; fall back to dense logic.
        self.update_batch_dense_inline(
            batch_values,
            group_indices,
            total_num_groups,
            expected_dense_touched,
            cmp,
        )
    }

    /// Emits the specified min_max values
    ///
    /// Returns (data_capacity, min_maxes), updating the current value of total_data_bytes
    ///
    /// - `data_capacity`: the total length of all strings and their contents,
    /// - `min_maxes`: the actual min/max values for each group
    fn emit_to(&mut self, emit_to: EmitTo) -> (usize, Vec<Option<Vec<u8>>>) {
        match emit_to {
            EmitTo::All => {
                (
                    take(&mut self.total_data_bytes), // reset total bytes and min_max
                    take(&mut self.min_max),
                )
            }
            EmitTo::First(n) => {
                let first_min_maxes: Vec<_> = self.min_max.drain(..n).collect();
                let first_data_capacity: usize = first_min_maxes
                    .iter()
                    .map(|opt| opt.as_ref().map(|s| s.len()).unwrap_or(0))
                    .sum();
                self.total_data_bytes -= first_data_capacity;
                (first_data_capacity, first_min_maxes)
            }
        }
    }

    fn size(&self) -> usize {
        self.total_data_bytes + self.min_max.len() * size_of::<Option<Vec<u8>>>()
    }

    fn analyze_batch(
        &mut self,
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> BatchStats {
        let unique_groups = if is_strictly_monotonic(group_indices) {
            count_runs(group_indices)
        } else {
            let mut set = HashSet::with_capacity(group_indices.len());
            for &idx in group_indices {
                set.insert(idx);
            }
            set.len()
        };

        let density_ppm = if total_num_groups == 0 {
            0
        } else {
            ((unique_groups * 1_000_000) / total_num_groups).min(1_000_000) as u32
        };

        BatchStats {
            unique_groups,
            total_num_groups,
            _rows_processed: group_indices.len(),
            density_ppm,
        }
    }

    fn record_batch_stats(&mut self, stats: &BatchStats) {
        let slot = self.adaptive.window_idx % LEARNING_WINDOW;

        if self.adaptive.batches_seen as usize >= LEARNING_WINDOW {
            let prev_unique = self.adaptive.unique_history[slot] as u64;
            let prev_total = self.adaptive.total_history[slot] as u64;
            self.adaptive.sum_unique_groups =
                self.adaptive.sum_unique_groups.saturating_sub(prev_unique);
            self.adaptive.sum_total_groups =
                self.adaptive.sum_total_groups.saturating_sub(prev_total);
        }

        self.adaptive.density_history[slot] = stats.density_ppm;
        self.adaptive.unique_history[slot] = stats.unique_groups;
        self.adaptive.total_history[slot] = stats.total_num_groups;
        self.adaptive.sum_unique_groups += stats.unique_groups as u64;
        self.adaptive.sum_total_groups += stats.total_num_groups as u64;

        if self.adaptive.batches_seen < LEARNING_WINDOW as u32 {
            self.adaptive.batches_seen += 1;
        }

        self.adaptive.window_idx = self.adaptive.window_idx.wrapping_add(1);
    }

    fn estimate_dense_touched_capacity(&self, stats: &BatchStats) -> usize {
        if self.adaptive.batches_seen == 0 {
            return stats.unique_groups;
        }

        let average_unique = (self.adaptive.sum_unique_groups
            / self.adaptive.batches_seen as u64) as usize;

        stats.unique_groups.max(average_unique)
    }

    fn reserve_dense_touched_capacity(&mut self, expected: usize) {
        let capacity = self.dense_touched_groups.capacity();
        if capacity < expected {
            self.dense_touched_groups
                .reserve(expected.saturating_sub(capacity));
        }
    }

    fn should_commit_mode(&self) -> bool {
        self.adaptive.batches_seen as usize >= LEARNING_WINDOW
    }

    fn average_density_ppm(&self, stats: &BatchStats) -> u32 {
        if self.adaptive.sum_total_groups == 0 {
            stats.density_ppm
        } else {
            ((self.adaptive.sum_unique_groups * 1_000_000)
                / self.adaptive.sum_total_groups.max(1)) as u32
        }
    }

    fn select_committed_mode(&self, stats: &BatchStats) -> WorkloadMode {
        let avg_density = self.average_density_ppm(stats);
        let total = stats.total_num_groups;

        if total <= 4_096 && avg_density >= DENSE_THRESHOLD_PPM {
            WorkloadMode::DenseInline
        } else if avg_density >= SIMPLE_THRESHOLD_PPM {
            WorkloadMode::SimpleTracked
        } else {
            WorkloadMode::SparseHashMap
        }
    }

    fn evaluate_transition(&mut self, stats: &BatchStats) {
        if stats.total_num_groups > (self.min_max.len().max(1) * 3) / 2 {
            self.adaptive.growth_events = self.adaptive.growth_events.saturating_add(1);
        }

        if self.adaptive.growth_events >= MAX_GROWTH_EVENTS {
            self.reset_learning_phase();
        }

        match self.mode {
            WorkloadMode::DenseInline => {
                if stats.density_ppm < 150_000 && stats.total_num_groups > 16_384 {
                    self.mode = WorkloadMode::SimpleTracked;
                    self.reset_simple_epoch();
                    self.adaptive.growth_events = 0;
                }
            }
            WorkloadMode::SimpleTracked => {
                if stats.density_ppm < 60_000 && stats.total_num_groups > 65_536 {
                    self.mode = WorkloadMode::SparseHashMap;
                    self.reset_sparse_state();
                    self.adaptive.growth_events = 0;
                } else if stats.density_ppm > 700_000 && stats.total_num_groups < 8_192 {
                    self.mode = WorkloadMode::DenseInline;
                    self.adaptive.growth_events = 0;
                }
            }
            WorkloadMode::SparseHashMap => {
                if stats.density_ppm > 300_000 && stats.total_num_groups < 32_768 {
                    self.mode = WorkloadMode::SimpleTracked;
                    self.reset_simple_epoch();
                    self.adaptive.growth_events = 0;
                }
            }
            WorkloadMode::Undecided => {
                if self.should_commit_mode() {
                    self.mode = self.select_committed_mode(stats);
                    self.adaptive.growth_events = 0;
                }
            }
        }
    }

    fn reset_learning_phase(&mut self) {
        self.mode = WorkloadMode::Undecided;
        self.adaptive = AdaptiveState::default();
    }

    fn reset_simple_epoch(&mut self) {
        self.simple_epoch = 0;
        self.simple_touched_slots.clear();
        self.simple_overflow.clear();

        for slot in &mut self.simple_scratch {
            slot.epoch = 0;
            slot.is_collision = false;
            slot.has_candidate = false;
            slot.candidate.clear();
        }

        for entry in &mut self.simple_overflow_candidates {
            entry.has_candidate = false;
            entry.candidate.clear();
        }
    }

    fn reset_sparse_state(&mut self) {
        // Placeholder for future sparse mode implementation.
    }

    #[allow(dead_code)]
    fn mode_name(&self) -> &'static str {
        match self.mode {
            WorkloadMode::DenseInline => "dense_inline",
            WorkloadMode::SimpleTracked => "simple_tracked",
            WorkloadMode::SparseHashMap => "sparse_hash_map",
            WorkloadMode::Undecided => "undecided",
        }
    }
}

fn is_strictly_monotonic(values: &[usize]) -> bool {
    values.windows(2).all(|window| window[0] <= window[1])
}

fn count_runs(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }

    let mut unique = 1;
    for window in values.windows(2) {
        if window[0] != window[1] {
            unique += 1;
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    #[test]
    fn min_bytes_updates_only_touched_groups() -> Result<()> {
        let mut state = MinMaxBytesState::new(DataType::Utf8);

        let batch1 = StringArray::from(vec![Some("c"), Some("b"), None, Some("d")]);
        let groups1 = vec![0_usize, 1, 2, 3];
        state.update_batch(
            batch1.iter().map(|opt| opt.map(|v| v.as_bytes())),
            &groups1,
            6,
            |a, b| a < b,
        )?;

        assert_eq!(state.min_max[0].as_deref(), Some(b"c".as_ref()));
        assert_eq!(state.min_max[1].as_deref(), Some(b"b".as_ref()));
        assert!(state.min_max[4].is_none());

        // Provide a batch that touches only groups 1 and 4
        let batch2 = StringArray::from(vec![Some("a"), Some("z")]);
        let groups2 = vec![1_usize, 4];
        state.update_batch(
            batch2.iter().map(|opt| opt.map(|v| v.as_bytes())),
            &groups2,
            6,
            |a, b| a < b,
        )?;

        // Group 1 should be updated, untouched groups should remain unchanged
        assert_eq!(state.min_max[0].as_deref(), Some(b"c".as_ref()));
        assert_eq!(state.min_max[1].as_deref(), Some(b"a".as_ref()));
        assert_eq!(state.min_max[4].as_deref(), Some(b"z".as_ref()));

        // Larger total_num_groups should not disturb earlier entries
        let batch3 = StringArray::from(vec![Some("b")]);
        state.update_batch(
            batch3.iter().map(|opt| opt.map(|v| v.as_bytes())),
            &[10],
            12,
            |a, b| a < b,
        )?;

        assert_eq!(state.min_max[0].as_deref(), Some(b"c".as_ref()));
        assert_eq!(state.min_max[10].as_deref(), Some(b"b".as_ref()));
        Ok(())
    }

    #[test]
    fn simple_tracked_updates_without_collisions() -> Result<()> {
        let mut state = MinMaxBytesState::new(DataType::Utf8);
        let batch = StringArray::from(vec![
            Some("delta"),
            Some("alpha"),
            Some("bravo"),
            None,
            Some("charlie"),
        ]);
        let groups = vec![0_usize, 1, 1, 2, 1];
        let batch_values: Vec<Option<&[u8]>> =
            batch.iter().map(|opt| opt.map(|s| s.as_bytes())).collect();
        let mut cmp = |a: &[u8], b: &[u8]| a < b;

        state.update_batch_simple_tracked(&batch_values, &groups, 4, 0, &mut cmp)?;

        assert_eq!(state.min_max[0].as_deref(), Some(b"delta".as_ref()));
        assert_eq!(state.min_max[1].as_deref(), Some(b"alpha".as_ref()));
        assert!(state.min_max[2].is_none());
        assert!(state.min_max[3].is_none());

        Ok(())
    }

    #[test]
    fn simple_tracked_handles_collisions() -> Result<()> {
        let mut state = MinMaxBytesState::new(DataType::Utf8);
        let groups = vec![
            5_usize,
            SIMPLE_SCRATCH_SIZE + 5,
            5_usize,
            SIMPLE_SCRATCH_SIZE + 5,
        ];
        let batch = StringArray::from(vec![Some("z"), Some("m"), Some("a"), Some("k")]);
        let batch_values: Vec<Option<&[u8]>> =
            batch.iter().map(|opt| opt.map(|s| s.as_bytes())).collect();
        let mut cmp = |a: &[u8], b: &[u8]| a < b;

        state.update_batch_simple_tracked(
            &batch_values,
            &groups,
            SIMPLE_SCRATCH_SIZE * 2 + 10,
            0,
            &mut cmp,
        )?;

        assert_eq!(state.min_max[5].as_deref(), Some(b"a".as_ref()));
        assert_eq!(
            state.min_max[SIMPLE_SCRATCH_SIZE + 5].as_deref(),
            Some(b"k".as_ref())
        );

        Ok(())
    }

    #[test]
    fn monotonic_fast_path_updates_groups() -> Result<()> {
        let mut state = MinMaxBytesState::new(DataType::Utf8);

        let initial = StringArray::from(vec![Some("m"), Some("k"), Some("z")]);
        let initial_groups = vec![0_usize, 1, 2];
        state.update_batch(
            initial.iter().map(|opt| opt.map(|s| s.as_bytes())),
            &initial_groups,
            3,
            |a, b| a < b,
        )?;

        let batch = StringArray::from(vec![
            Some("h"),
            Some("g"),
            None,
            Some("j"),
            Some("x"),
            Some("y"),
        ]);
        let groups = vec![0_usize, 0, 1, 1, 2, 2];

        state.update_batch(
            batch.iter().map(|opt| opt.map(|s| s.as_bytes())),
            &groups,
            3,
            |a, b| a < b,
        )?;

        assert_eq!(state.min_max[0].as_deref(), Some(b"g".as_ref()));
        assert_eq!(state.min_max[1].as_deref(), Some(b"j".as_ref()));
        assert_eq!(state.min_max[2].as_deref(), Some(b"x".as_ref()));

        Ok(())
    }
}
