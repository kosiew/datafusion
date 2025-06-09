// Licensed to the Apache Software Foundation (ASF) under one
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

use arrow::array::{ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::scalar::ScalarValue::UInt64;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let n: usize = 5;
    let num = Arc::new(Int32Array::from((0..n as _).collect::<Vec<i32>>())) as ArrayRef;

    let dict_values = StringArray::from(vec![None, Some("abc")]);
    let dict_indices = Int32Array::from(vec![0; n]);
    // all idx point to 0 - which means that all values in the dictionary are None
    let dict = DictionaryArray::new(dict_indices, Arc::new(dict_values) as ArrayRef);

    let schema = Arc::new(Schema::new(vec![
        Field::new("num1", DataType::Int32, false),
        Field::new("num2", DataType::Int32, false), // num2 to disable SingleDistinctToGroupBy optimisation
        Field::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![num.clone(), num.clone(), Arc::new(dict)],
    )?;

    let provider = MemTable::try_new(schema, vec![vec![batch]])?;
    let mut session_config = SessionConfig::default();

    session_config = session_config.set(
        "datafusion.execution.target_partitions",
        &UInt64(Some(1u64)), // won't work with more than 1 partition
    );

    let ctx = SessionContext::new_with_config(session_config);
    ctx.register_table("t", Arc::new(provider))?;

    let df = ctx
        .sql("select count(distinct dict), count(num2) from t group by num1")
        .await?;
    // count(distinct ...) doesn't count None values, so the result should be 0
    df.show().await?;

    Ok(())
}
