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

//! DataFusion benchmark runner
#[cfg(feature = "bench-cancellation")]
pub mod cancellation;
#[cfg(feature = "bench-clickbench")]
pub mod clickbench;
#[cfg(feature = "bench-h2o")]
pub mod h2o;
#[cfg(feature = "bench-hj")]
pub mod hj;
#[cfg(feature = "bench-imdb")]
pub mod imdb;
#[cfg(feature = "bench-nlj")]
pub mod nlj;
#[cfg(feature = "bench-smj")]
pub mod smj;
#[cfg(feature = "bench-sort-tpch")]
pub mod sort_tpch;
#[cfg(feature = "bench-tpcds")]
pub mod tpcds;
#[cfg(feature = "bench-tpch")]
pub mod tpch;
#[cfg(feature = "bench-common")]
pub mod util;
