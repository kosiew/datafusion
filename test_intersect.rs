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

use datafusion::error::Result;
use datafusion::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();

    // Register a test table
    ctx.sql("CREATE TABLE test (col_int32 INT, col_utf8 VARCHAR)")
        .await?
        .show()
        .await?;

    // Try the intersect query
    let sql = "SELECT col_int32, col_utf8 FROM test \
               INTERSECT SELECT col_int32, col_utf8 FROM test";

    let df = ctx.sql(sql).await?;
    let plan = df.logical_plan();

    println!("Plan:\n{}", plan.display_indent());

    Ok(())
}
