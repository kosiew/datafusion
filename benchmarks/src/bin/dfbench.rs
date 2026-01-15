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
use datafusion::error::{DataFusionError, Result};

use clap::{Parser, Subcommand};

#[cfg(all(feature = "snmalloc", feature = "mimalloc"))]
compile_error!(
    "feature \"snmalloc\" and feature \"mimalloc\" cannot be enabled at the same time"
);

#[cfg(feature = "snmalloc")]
#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "bench-cancellation")]
use datafusion_benchmarks::cancellation;
#[cfg(feature = "bench-clickbench")]
use datafusion_benchmarks::clickbench;
#[cfg(feature = "bench-h2o")]
use datafusion_benchmarks::h2o;
#[cfg(feature = "bench-hj")]
use datafusion_benchmarks::hj;
#[cfg(feature = "bench-imdb")]
use datafusion_benchmarks::imdb;
#[cfg(feature = "bench-nlj")]
use datafusion_benchmarks::nlj;
#[cfg(feature = "bench-smj")]
use datafusion_benchmarks::smj;
#[cfg(feature = "bench-sort-tpch")]
use datafusion_benchmarks::sort_tpch;
#[cfg(feature = "bench-tpcds")]
use datafusion_benchmarks::tpcds;
#[cfg(feature = "bench-tpch")]
use datafusion_benchmarks::tpch;

/// Placeholder type for disabled benchmark subcommands.
/// When a benchmark feature is disabled, clap still needs a type for the variant,
/// but it will never be instantiated since the disabled_benchmark() function
/// returns early with an error.
#[derive(Debug, clap::Args)]
struct DisabledCommand;

#[derive(Debug, Parser)]
#[command(about = "benchmark command")]
struct Cli {
    #[command(subcommand)]
    command: Options,
}

#[derive(Debug, Subcommand)]
enum Options {
    #[cfg(feature = "bench-cancellation")]
    Cancellation(cancellation::RunOpt),
    #[cfg(not(feature = "bench-cancellation"))]
    #[command(name = "cancellation")]
    Cancellation(DisabledCommand),
    #[cfg(feature = "bench-clickbench")]
    Clickbench(clickbench::RunOpt),
    #[cfg(not(feature = "bench-clickbench"))]
    #[command(name = "clickbench")]
    Clickbench(DisabledCommand),
    #[cfg(feature = "bench-h2o")]
    H2o(h2o::RunOpt),
    #[cfg(not(feature = "bench-h2o"))]
    #[command(name = "h2o")]
    H2o(DisabledCommand),
    #[cfg(feature = "bench-hj")]
    HJ(hj::RunOpt),
    #[cfg(not(feature = "bench-hj"))]
    #[command(name = "hj")]
    HJ(DisabledCommand),
    #[cfg(feature = "bench-imdb")]
    Imdb(imdb::RunOpt),
    #[cfg(not(feature = "bench-imdb"))]
    #[command(name = "imdb")]
    Imdb(DisabledCommand),
    #[cfg(feature = "bench-nlj")]
    Nlj(nlj::RunOpt),
    #[cfg(not(feature = "bench-nlj"))]
    #[command(name = "nlj")]
    Nlj(DisabledCommand),
    #[cfg(feature = "bench-smj")]
    Smj(smj::RunOpt),
    #[cfg(not(feature = "bench-smj"))]
    #[command(name = "smj")]
    Smj(DisabledCommand),
    #[cfg(feature = "bench-sort-tpch")]
    SortTpch(sort_tpch::RunOpt),
    #[cfg(not(feature = "bench-sort-tpch"))]
    #[command(name = "sort-tpch")]
    SortTpch(DisabledCommand),
    #[cfg(feature = "bench-tpch")]
    Tpch(tpch::RunOpt),
    #[cfg(not(feature = "bench-tpch"))]
    #[command(name = "tpch")]
    Tpch(DisabledCommand),
    #[cfg(feature = "bench-tpcds")]
    Tpcds(tpcds::RunOpt),
    #[cfg(not(feature = "bench-tpcds"))]
    #[command(name = "tpcds")]
    Tpcds(DisabledCommand),
}

/// Returns an error message for disabled benchmarks.
/// This function is only called when a benchmark feature is disabled via conditional compilation,
/// so it may appear unused when all features are enabled. The `#[allow(dead_code)]` attribute
/// suppresses the warning in those cases.
#[allow(dead_code)]
fn disabled_benchmark(benchmark: &str, feature: &str) -> Result<()> {
    Err(DataFusionError::Execution(format!(
        "{benchmark} benchmark is disabled. Rebuild with --features {feature}."
    )))
}

// Main benchmark runner entrypoint
#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "bench-cancellation")]
        Options::Cancellation(opt) => opt.run().await,
        #[cfg(not(feature = "bench-cancellation"))]
        Options::Cancellation(_) => {
            disabled_benchmark("cancellation", "bench-cancellation")
        }
        #[cfg(feature = "bench-clickbench")]
        Options::Clickbench(opt) => opt.run().await,
        #[cfg(not(feature = "bench-clickbench"))]
        Options::Clickbench(_) => disabled_benchmark("clickbench", "bench-clickbench"),
        #[cfg(feature = "bench-h2o")]
        Options::H2o(opt) => opt.run().await,
        #[cfg(not(feature = "bench-h2o"))]
        Options::H2o(_) => disabled_benchmark("h2o", "bench-h2o"),
        #[cfg(feature = "bench-hj")]
        Options::HJ(opt) => opt.run().await,
        #[cfg(not(feature = "bench-hj"))]
        Options::HJ(_) => disabled_benchmark("hj", "bench-hj"),
        // Box::pin required for IMDB, TPCH, and TPCDS due to large future sizes
        // from recursive query execution that would exceed stack size limits
        #[cfg(feature = "bench-imdb")]
        Options::Imdb(opt) => Box::pin(opt.run()).await,
        #[cfg(not(feature = "bench-imdb"))]
        Options::Imdb(_) => disabled_benchmark("imdb", "bench-imdb"),
        #[cfg(feature = "bench-nlj")]
        Options::Nlj(opt) => opt.run().await,
        #[cfg(not(feature = "bench-nlj"))]
        Options::Nlj(_) => disabled_benchmark("nlj", "bench-nlj"),
        #[cfg(feature = "bench-smj")]
        Options::Smj(opt) => opt.run().await,
        #[cfg(not(feature = "bench-smj"))]
        Options::Smj(_) => disabled_benchmark("smj", "bench-smj"),
        #[cfg(feature = "bench-sort-tpch")]
        Options::SortTpch(opt) => opt.run().await,
        #[cfg(not(feature = "bench-sort-tpch"))]
        Options::SortTpch(_) => disabled_benchmark("sort-tpch", "bench-sort-tpch"),
        #[cfg(feature = "bench-tpch")]
        Options::Tpch(opt) => Box::pin(opt.run()).await,
        #[cfg(not(feature = "bench-tpch"))]
        Options::Tpch(_) => disabled_benchmark("tpch", "bench-tpch"),
        #[cfg(feature = "bench-tpcds")]
        Options::Tpcds(opt) => Box::pin(opt.run()).await,
        #[cfg(not(feature = "bench-tpcds"))]
        Options::Tpcds(_) => disabled_benchmark("tpcds", "bench-tpcds"),
    }
}
