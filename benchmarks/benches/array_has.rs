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

use arrow::array::{Int32Builder, ListBuilder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use tokio::runtime::Runtime;

const NUM_ROWS: usize = 4_096;
const HAYSTACK_LEN: usize = 256;
const NEEDLES_LEN: usize = 32;
const MATCH_VALUE: i32 = 8_675_309;
const MISSING_VALUE: i32 = -1;

fn build_input_data() -> (Arc<Schema>, Vec<Vec<RecordBatch>>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "haystack",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ),
        Field::new(
            "needles",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ),
    ]));

    let mut haystack_builder =
        ListBuilder::new(Int32Builder::with_capacity(NUM_ROWS * HAYSTACK_LEN));
    let mut needles_builder =
        ListBuilder::new(Int32Builder::with_capacity(NUM_ROWS * NEEDLES_LEN));

    for row in 0..NUM_ROWS {
        let haystack_values = build_haystack_row(row);
        {
            let values_builder = haystack_builder.values();
            for value in &haystack_values {
                values_builder.append_value(*value);
            }
        }
        haystack_builder.append(true);

        let needle_values = build_needles_row(row, &haystack_values);
        {
            let values_builder = needles_builder.values();
            for value in &needle_values {
                values_builder.append_value(*value);
            }
        }
        needles_builder.append(true);
    }

    let haystack_array = haystack_builder.finish();
    let needles_array = needles_builder.finish();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(haystack_array), Arc::new(needles_array)],
    )
    .unwrap();

    (schema, vec![vec![batch]])
}

fn build_haystack_row(row: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(HAYSTACK_LEN);
    let base = (row as i32 * 7_919).rem_euclid(65_536);

    for i in 0..HAYSTACK_LEN {
        let mut value = base + ((i as i32 * 37).rem_euclid(65_536));
        value = value.rem_euclid(65_536);

        if i == 0 && row % 5 == 0 {
            value = MATCH_VALUE;
        }

        values.push(value);
    }

    values
}

fn build_needles_row(row: usize, haystack_values: &[i32]) -> Vec<i32> {
    let mut values = Vec::with_capacity(NEEDLES_LEN);

    for i in 0..NEEDLES_LEN {
        values.push(haystack_values[i % haystack_values.len()]);
    }

    if row % 4 == 0 && !values.is_empty() {
        values[0] = MATCH_VALUE;
    }

    if row % 2 == 0 && !values.is_empty() {
        let last = values.len() - 1;
        values[last] = MISSING_VALUE;
    }

    values
}

async fn run_query(ctx: &SessionContext, sql: &str) {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(rows, NUM_ROWS);
}

fn context_with_table(table: Arc<MemTable>) -> Arc<SessionContext> {
    let ctx = SessionContext::new();
    ctx.register_table("arrays", table).unwrap();
    Arc::new(ctx)
}

fn benchmark_array_membership(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();

    let (schema, partitions) = build_input_data();
    let table = Arc::new(MemTable::try_new(schema, partitions).unwrap());

    let ctx_has = context_with_table(Arc::clone(&table));
    let ctx_all = context_with_table(Arc::clone(&table));
    let ctx_any = context_with_table(table);

    let array_has_sql: Arc<str> = Arc::from(format!(
        "SELECT array_has(haystack, {MATCH_VALUE}) FROM arrays"
    ));
    let array_has_all_sql: Arc<str> = Arc::from(String::from(
        "SELECT array_has_all(haystack, needles) FROM arrays",
    ));
    let array_has_any_sql: Arc<str> = Arc::from(String::from(
        "SELECT array_has_any(haystack, needles) FROM arrays",
    ));

    let mut group = c.benchmark_group("array_membership");
    group.throughput(Throughput::Elements((NUM_ROWS * HAYSTACK_LEN) as u64));

    group.bench_function("array_has", |b| {
        let ctx = Arc::clone(&ctx_has);
        let sql = Arc::clone(&array_has_sql);
        b.to_async(&runtime).iter(move || {
            let ctx = Arc::clone(&ctx);
            let sql = Arc::clone(&sql);
            async move {
                run_query(ctx.as_ref(), sql.as_ref()).await;
            }
        });
    });

    group.bench_function("array_has_all", |b| {
        let ctx = Arc::clone(&ctx_all);
        let sql = Arc::clone(&array_has_all_sql);
        b.to_async(&runtime).iter(move || {
            let ctx = Arc::clone(&ctx);
            let sql = Arc::clone(&sql);
            async move {
                run_query(ctx.as_ref(), sql.as_ref()).await;
            }
        });
    });

    group.bench_function("array_has_any", |b| {
        let ctx = Arc::clone(&ctx_any);
        let sql = Arc::clone(&array_has_any_sql);
        b.to_async(&runtime).iter(move || {
            let ctx = Arc::clone(&ctx);
            let sql = Arc::clone(&sql);
            async move {
                run_query(ctx.as_ref(), sql.as_ref()).await;
            }
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_array_membership);
criterion_main!(benches);
