<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Memory Accounting Integration Plan

This document outlines a strategy for evolving the Arrow memory accounting API proposed in [arrow-rs PR #7303](https://github.com/apache/arrow-rs/pull/7303) and how DataFusion can integrate it.

## Summary of the Arrow API

- `MemoryPool` and `MemoryReservation` provide RAII style accounting. A reservation is created with `MemoryPool::reserve`, resized with `resize`, and automatically released on drop.
- `Bytes`, `Buffer`, and `MutableBuffer` gain a `claim` method that attaches them to a pool and maintains a `MemoryReservation` internally.
- `TrackingMemoryPool` is a simple implementation that just counts bytes.

This API operates at the buffer level and is largely independent from higher-level `Array` implementations.

## Considerations for Arrow

1. **Array Level Integration** – Many arrays are composed of multiple buffers. Exposing a method such as `Array::claim(pool)` would allow bulk registration of all underlying buffers. This could be implemented via trait default methods that traverse child arrays and buffers.
2. **Optional Pool Parameter** – New array/buffer builders might accept an optional `&dyn MemoryPool` when created. Buffers allocated internally would automatically reserve from this pool.
3. **Resizable Reservations** – Reserving based only on capacity can misrepresent actual usage when buffers share allocations. Providing a way to detach or resize reservations upon slicing/cloning would help represent true usage.

## DataFusion Usage Patterns

DataFusion already has a memory tracking system based on `MemoryPool`, `MemoryConsumer`, and `MemoryReservation`. Operators manually reserve memory as they grow internal state. Buffer allocations from arrow-rs are currently outside this system.

Desired capabilities for DataFusion include:

- **Automatic accounting of buffer allocations** used by operators, so tracked memory reflects actual Arrow buffer usage.
- **Visibility into shared buffers** – when arrays share buffers, account the memory only once but attribute it to the relevant consumer.
- **Interoperability** – the same pool should work for both manual reservations and Arrow buffers to avoid double counting.

## Proposed Implementation Steps

1. **Implement the Arrow `MemoryPool` Trait**
   - Provide an adapter struct in DataFusion (e.g. `DataFusionMemoryPool`) that implements `arrow_buffer::pool::MemoryPool`.
   - Internally it will create a `MemoryConsumer` and `MemoryReservation` from DataFusion’s existing pool. `reserve` will correspond to `try_grow` on that reservation.
   - `resize` will grow or shrink the reservation appropriately.

2. **Expose Pool Handles in Execution Context**
   - Extend `RuntimeEnv` or the task context to expose a `&dyn arrow_buffer::pool::MemoryPool` alongside the existing DataFusion pool.
   - When operators create buffers (e.g. via `MutableBuffer::new` or `VecAllocExt` proxies), pass this pool and immediately call `claim`.

3. **Claim Memory for Arrays**
   - Introduce helper methods to claim memory for entire arrays and record batches. This will walk buffers and call `claim` on each unique buffer.
   - Operators that hold on to `ArrayRef`s for an extended time (aggregates, joins, etc.) should call this helper when the array is stored, ensuring memory usage is tracked even if the array was produced elsewhere.

4. **Update Buffer Builders and Proxies**
   - Update DataFusion’s `VecAllocExt` and hash table proxies to also create arrow buffers with the provided pool where possible.
   - When arrays are built using Arrow builders, pass the pool so allocations are immediately accounted for.

5. **Testing and Metrics**
   - Verify that existing memory limits still apply and that buffer allocations appear in memory pool statistics.
   - Add integration tests that construct queries with shared buffers and ensure memory is only counted once.

## Long Term Considerations

- Assess whether Arrow should expose higher level hooks (e.g. at `RecordBatch` creation) to make registration easier.
- Explore unified reporting so DataFusion can inspect per-buffer reservations if needed for diagnostics.
- Consider a background task to periodically reconcile allocations from Arrow and DataFusion to detect leaks or mismatches.

This plan aims to combine Arrow’s buffer-level accounting with DataFusion’s operator-level tracking, enabling more accurate measurement and ultimately better resource control.
