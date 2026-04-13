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

use arrow::array::ArrayRef;
use arrow::datatypes::{Int32Type, StringViewType};
use arrow::util::bench_util::{
    create_primitive_array, create_string_view_array_with_len,
    create_string_view_array_with_max_len,
};
use arrow::util::test_util::seedable_rng;
use arrow_schema::DataType;
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use datafusion_physical_plan::aggregates::group_values::multi_group_by::GroupColumn;
use datafusion_physical_plan::aggregates::group_values::multi_group_by::bytes_view::ByteViewGroupValueBuilder;
use datafusion_physical_plan::aggregates::group_values::multi_group_by::primitive::PrimitiveGroupValueBuilder;
use rand::distr::{Bernoulli, Distribution};
use std::hint::black_box;
use std::sync::Arc;

const SIZES: [usize; 3] = [1_000, 10_000, 100_000];
const NULL_DENSITIES: [f32; 3] = [0.0, 0.1, 0.5];
// Bound the additional row-selection matrix so CI keeps running the broad
// contiguous baseline without tripling the ByteView benchmark duration.
const ROW_SELECTION_BENCH_SIZE: usize = 10_000;
const ROW_SELECTION_BENCH_NULL_DENSITY: f32 = 0.1;
const EQUAL_TO_PROBABILITY_CASES: [(f64, &str); 3] =
    [(0.75, "0.75 true"), (0.5, "0.5 true"), (0.25, "0.25 true")];

fn bench_vectorized_append(c: &mut Criterion) {
    byte_view_vectorized_append(c);
    primitive_vectorized_append(c);
}

fn byte_view_vectorized_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("ByteViewGroupValueBuilder_vectorized_append");

    for &size in &SIZES {
        let rows: Vec<usize> = (0..size).collect();

        for &null_density in &NULL_DENSITIES {
            let row_selections = if should_bench_row_selection(size, null_density) {
                byte_view_row_selections(size)
            } else {
                Vec::new()
            };

            bench_byte_view_input(
                &mut group,
                "inline",
                size,
                &rows,
                null_density,
                Arc::new(create_string_view_array_with_len(
                    size,
                    null_density,
                    8,
                    false,
                )),
                &row_selections,
            );
            bench_byte_view_input(
                &mut group,
                "scenario",
                size,
                &rows,
                null_density,
                Arc::new(create_string_view_array_with_len(
                    size,
                    null_density,
                    64,
                    true,
                )),
                &row_selections,
            );
            bench_byte_view_input(
                &mut group,
                "random",
                size,
                &rows,
                null_density,
                Arc::new(create_string_view_array_with_max_len(
                    size,
                    null_density,
                    400,
                )),
                &row_selections,
            );
        }
    }

    group.finish();
}

fn should_bench_row_selection(size: usize, null_density: f32) -> bool {
    size == ROW_SELECTION_BENCH_SIZE && null_density == ROW_SELECTION_BENCH_NULL_DENSITY
}

fn byte_view_row_selections(size: usize) -> Vec<(&'static str, Vec<usize>)> {
    vec![
        (
            "non_contiguous",
            (0..size).step_by(2).chain((1..size).step_by(2)).collect(),
        ),
        ("duplicated", (0..size).map(|i| i / 2).collect()),
        ("unsorted", (0..size).rev().collect()),
    ]
}

fn bench_byte_view_input(
    group: &mut BenchmarkGroup<WallTime>,
    bench_prefix: &str,
    size: usize,
    rows: &[usize],
    null_density: f32,
    input: ArrayRef,
    row_selections: &[(&str, Vec<usize>)],
) {
    bytes_contiguous_bench(group, bench_prefix, size, rows, null_density, &input);

    for &(row_selection, ref rows) in row_selections {
        bytes_append_row_selection_bench(
            group,
            bench_prefix,
            row_selection,
            size,
            null_density,
            rows,
            &input,
        );
    }
}

fn bytes_append_row_selection_bench(
    group: &mut BenchmarkGroup<WallTime>,
    bench_prefix: &str,
    row_selection: &str,
    size: usize,
    null_density: f32,
    rows: &[usize],
    input: &ArrayRef,
) {
    let function_name =
        format!("{bench_prefix}_row_{row_selection}_null_{null_density:.1}_size_{size}");
    bytes_append_bench(group, &function_name, rows, input);
}

fn bytes_contiguous_bench(
    group: &mut BenchmarkGroup<WallTime>,
    bench_prefix: &str,
    size: usize,
    rows: &[usize],
    null_density: f32,
    input: &ArrayRef,
) {
    let function_name =
        format!("{bench_prefix}_row_contiguous_null_{null_density:.1}_size_{size}");
    bytes_append_bench(group, &function_name, rows, input);

    // vectorized_equal_to
    vectorized_equal_to(
        group,
        ByteViewGroupValueBuilder::<StringViewType>::new(),
        &function_name,
        rows,
        input,
        "all_true",
        vec![true; size],
    );
    for (probability, description) in EQUAL_TO_PROBABILITY_CASES {
        vectorized_equal_to(
            group,
            ByteViewGroupValueBuilder::<StringViewType>::new(),
            &function_name,
            rows,
            input,
            description,
            sample_equal_to_results(size, probability),
        );
    }
    // Not adding 0 true case here as if we optimize for 0 true cases the caller should avoid calling this method at all
}

fn sample_equal_to_results(size: usize, probability: f64) -> Vec<bool> {
    let mut rng = seedable_rng();
    let d = Bernoulli::new(probability).unwrap();
    (0..size).map(|_| d.sample(&mut rng)).collect()
}

fn bytes_append_bench(
    group: &mut BenchmarkGroup<WallTime>,
    function_name: &str,
    rows: &[usize],
    input: &ArrayRef,
) {
    let id = BenchmarkId::new(function_name, "vectorized_append");
    group.bench_function(id, |b| {
        b.iter(|| {
            let mut builder = ByteViewGroupValueBuilder::<StringViewType>::new();
            builder.vectorized_append(input, rows).unwrap();
        });
    });

    // append_val
    let id = BenchmarkId::new(function_name, "append_val");
    group.bench_function(id, |b| {
        b.iter(|| {
            let mut builder = ByteViewGroupValueBuilder::<StringViewType>::new();
            for &i in rows {
                builder.append_val(input, i).unwrap();
            }
        });
    });
}

fn primitive_vectorized_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("PrimitiveGroupValueBuilder_vectorized_append");

    for &size in &SIZES {
        let rows: Vec<usize> = (0..size).collect();

        for &null_density in &NULL_DENSITIES {
            if null_density == 0.0 {
                bench_single_primitive::<false>(&mut group, size, &rows, null_density)
            }
            bench_single_primitive::<true>(&mut group, size, &rows, null_density);
        }
    }

    group.finish();
}

fn bench_single_primitive<const NULLABLE: bool>(
    group: &mut BenchmarkGroup<WallTime>,
    size: usize,
    rows: &[usize],
    null_density: f32,
) {
    if !NULLABLE {
        assert_eq!(
            null_density, 0.0,
            "non-nullable case must have null_density 0"
        );
    }

    let input = create_primitive_array::<Int32Type>(size, null_density);
    let input: ArrayRef = Arc::new(input);
    let function_name = format!("null_{null_density:.1}_nullable_{NULLABLE}_size_{size}");

    // vectorized_append
    let id = BenchmarkId::new(&function_name, "vectorized_append");
    group.bench_function(id, |b| {
        b.iter(|| {
            let mut builder =
                PrimitiveGroupValueBuilder::<Int32Type, NULLABLE>::new(DataType::Int32);
            builder.vectorized_append(&input, rows).unwrap();
        });
    });

    // append_val
    let id = BenchmarkId::new(&function_name, "append_val");
    group.bench_function(id, |b| {
        b.iter(|| {
            let mut builder =
                PrimitiveGroupValueBuilder::<Int32Type, NULLABLE>::new(DataType::Int32);
            for &i in rows {
                builder.append_val(&input, i).unwrap();
            }
        });
    });

    // vectorized_equal_to
    vectorized_equal_to(
        group,
        PrimitiveGroupValueBuilder::<Int32Type, NULLABLE>::new(DataType::Int32),
        &function_name,
        rows,
        &input,
        "all_true",
        vec![true; size],
    );
    for (probability, description) in EQUAL_TO_PROBABILITY_CASES {
        vectorized_equal_to(
            group,
            PrimitiveGroupValueBuilder::<Int32Type, NULLABLE>::new(DataType::Int32),
            &function_name,
            rows,
            &input,
            description,
            sample_equal_to_results(size, probability),
        );
    }
    // Not adding 0 true case here as if we optimize for 0 true cases the caller should avoid calling this method at all
}

/// Test `vectorized_equal_to` with different number of true in the initial results
#[expect(clippy::needless_pass_by_value)]
fn vectorized_equal_to<GroupColumnBuilder: GroupColumn>(
    group: &mut BenchmarkGroup<WallTime>,
    mut builder: GroupColumnBuilder,
    function_name: &str,
    rows: &[usize],
    input: &ArrayRef,
    equal_to_result_description: &str,
    equal_to_results: Vec<bool>,
) {
    let id = BenchmarkId::new(
        function_name,
        format!("vectorized_equal_to_{equal_to_result_description}"),
    );
    group.bench_function(id, |b| {
        builder.vectorized_append(input, rows).unwrap();

        b.iter(|| {
            // Cloning is a must as `vectorized_equal_to` will modify the input vec
            // and without cloning all benchmarks after the first one won't be meaningful
            let mut equal_to_results = equal_to_results.clone();
            builder.vectorized_equal_to(rows, input, rows, &mut equal_to_results);

            // Make sure that the compiler does not optimize away the call
            black_box(equal_to_results);
        });
    });
}

criterion_group!(benches, bench_vectorized_append);
criterion_main!(benches);
