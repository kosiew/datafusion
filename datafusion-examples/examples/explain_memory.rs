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

//! Demonstrates how to use [`ExplainMemory`] to inspect memory
//! reservations and obtain reports of the largest memory consumers.
//!
//! The example performs the following steps:
//! 1. Create a [`GreedyMemoryPool`] wrapped in a [`TrackConsumersPool`]
//!    so that the pool can report the top memory consumers.
//! 2. Register two [`MemoryConsumer`]s and grow their reservations.
//! 3. Use [`ExplainMemory::explain_memory`] to display the memory usage
//!    of each reservation.
//! 4. Call [`report_top_consumers`] to print the largest consumers
//!    currently tracked by the pool.

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::memory_pool::{
    report_top_consumers, ExplainMemory, GreedyMemoryPool, MemoryConsumer, MemoryPool,
    TrackConsumersPool,
};

fn main() -> Result<()> {
    // Create a pool limited to 2 KiB and wrap it so we can track consumers
    let inner_pool = GreedyMemoryPool::new(2 * 1024);
    let tracked_pool = Arc::new(TrackConsumersPool::new(
        inner_pool,
        NonZeroUsize::new(5).unwrap(),
    ));

    // Use a trait object for registering consumers
    let pool: Arc<dyn MemoryPool> = tracked_pool.clone();

    // Register two consumers with the pool
    let mut reservation_a = MemoryConsumer::new("consumer_a").register(&pool);
    let mut reservation_b = MemoryConsumer::new("consumer_b").register(&pool);

    // Grow the reservations
    reservation_a.try_grow(1024)?;
    reservation_b.try_grow(512)?;

    // Report memory usage for each reservation
    println!("{}", reservation_a.explain_memory()?);
    println!("{}", reservation_b.explain_memory()?);

    // Show the top consumers recorded by the pool
    if let Some(report) = report_top_consumers(tracked_pool.as_ref(), 5) {
        println!("\nTop consumers:\n{report}");
    }

    Ok(())
}
