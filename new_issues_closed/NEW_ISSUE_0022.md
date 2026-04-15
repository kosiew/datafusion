same as NEW_ISSUE_0021.md
source: pr-21833_a
# Issue: Implement cross-partition chunk-state coordination for memory-limited NestedLoopJoin LEFT-producing joins

## Summary

Follow-up to [NEW_ISSUE_01](NEW_ISSUE_01.md). This issue tracks the longer-term architectural fix to safely enable memory-limited fallback for FULL/LEFT/LEFT ANTI/LEFT MARK joins when the right side has multiple output partitions.

Current status (after NEW_ISSUE_01):
- Memory-limited fallback is disabled for LEFT-producing joins with multi-partition right side to preserve correctness.
- This work item re-enables fallback via proper cross-partition coordination.

## Why this matters

- **Usability:** Large FULL/LEFT joins with tight memory limits can proceed via spill instead of failing.
- **Performance:** Avoids OOM for queries that could otherwise succeed by spilling to disk.
- **Architectural consistency:** Memory-limited path achieves feature parity with single-pass path for all join types.

## Gap: single-pass vs. memory-limited coordination

### Single-pass path

Uses shared `JoinLeftData` with:
- `probe_threads_counter = AtomicUsize::new(right_partition_count)`
- Each partition decrements on probe completion
- Unmatched-left emission guarded by `probe_completed()` check (counter reaches 0)

Result: unmatched-left rows emitted only after ALL right partitions finish probing the full left side.

### Memory-limited path (current broken state)

Each partition builds independent `JoinLeftData` for each left chunk with:
- `probe_threads_counter = AtomicUsize::new(1)` (per stream only)
- No cross-partition coordination
- Unmatched-left emission happens per-partition-per-chunk

Result: left rows incorrectly marked unmatched before other partitions probe them.

## Proposed solution: shared chunk-state registry

### Design overview

For memory-limited mode, maintain a shared registry of active left chunks (one per partition, in parallel):

```
SpillStateActive {
    // ... existing fields ...
    
    // NEW: per-chunk shared state, indexed by chunk sequence number
    chunk_states: Arc<Mutex<BTreeMap<ChunkIndex, Arc<SharedChunkState>>>>,
}

struct SharedChunkState {
    // The JoinLeftData for this chunk, shared across all right partitions
    left_data: Arc<JoinLeftData>,
    
    // Right partition completion counter: starts at right_partition_count
    // Each partition decrements; unmatched emission waits for 0
    right_partition_completion: AtomicUsize,
    
    // Chunk index (for ordering/cleanup)
    chunk_index: usize,
}
```

### Key behaviors

1. **Chunk creation:** When BufferingLeft creates a new left chunk, register it in `chunk_states`.
2. **Probe completion:** In ProbeRight→EmitRightUnmatched, each partition:
   - Merges right batch matched bitmap (existing logic)
   - Decrements `right_partition_completion` for this chunk
   - If counter reaches 0, signal that EmitLeftUnmatched can proceed
3. **Unmatched-left gating:** In process_left_unmatched, check `right_partition_completion == 0` before emitting.
4. **Chunk cleanup:** After all right partitions emit any chunk-specific state, remove from registry.

### Synchronization strategy

**Option A (lock-per-chunk):** Use interior Arc<Mutex<SharedChunkState>> to serialize updates
- Pro: simpler lifetime management
- Con: potential contention if many partitions

**Option B (atomic counter only):** Use AtomicUsize for completion, rely on eventual consistency
- Pro: lock-free
- Con: requires careful ordering guarantees and adds complexity

**Recommendation:** Start with Option A for clarity, optimize later if profiling shows contention.

## Implementation steps

### Phase 1: introduce SharedChunkState (no behavior change yet)

1. Define `SharedChunkState` struct in nested_loop_join.rs
2. Modify `SpillStateActive` to include chunk registry
3. Allocate chunk state when entering handle_buffering_left_memory_limited
4. Keep single-partition JoinLeftData logic unchanged (probe_threads_counter = 1)

### Phase 2: implement cross-partition counter logic

1. Modify JoinLeftData::new to accept an optional `right_partition_completion` Arc<AtomicUsize>
2. In handle_buffering_left_memory_limited:
   - Create SharedChunkState with right_partition_completion = AtomicUsize::new(right_partition_count)
   - Create JoinLeftData with reference to the shared counter
3. In handle_emit_right_unmatched:
   - After merging bitmap, decrement the shared counter
4. In process_left_unmatched:
   - Read the shared counter; skip emission if > 0

### Phase 3: add regression tests

1. Unit test: multi-partition FULL JOIN with memory limit forces spill and emits correct unmatched rows
2. Edge case: left row matches only in a non-local partition
3. Verify spill_count metric and result row counts

### Phase 4: SQL logic tests

Extend datafusion/sqllogictest/test_files/nested_loop_join_spill.slt:
- Add FULL JOIN case with 4 target partitions
- Verify row counts match single-partition baseline
- Assert spill_count > 0

## Acceptance criteria

1. **Correctness:** FULL/LEFT joins with multi-partition right side under memory limit produce identical results to single-partition or unlimited-memory execution.

2. **No regressions:**
   - Existing single-partition memory-limited tests pass
   - Existing multi-partition single-pass tests pass
   - Right-side unmatched bitmap accumulation (for RIGHT/FULL right unmatched) unaffected

3. **Coverage:**
   - FULL JOIN multi-partition regression test (unit + optional SLT)
   - LEFT JOIN multi-partition regression test (unit)
   - LEFT ANTI / LEFT MARK multi-partition (optional)

4. **Metrics:** Spill count and memory accounting remain correct across chunks and partitions.

5. **Fallback re-enabled:** After merge, memory-limited fallback is available for LEFT-producing joins with multi-partition right side.

## Risks and mitigations

**Risk:** Synchronization overhead with chunk registry lock.
- Mitigation: measure contention; opt for lock-free if profiling shows bottleneck. Start with locks for safety.

**Risk:** Lifetime/cleanup bugs with chunk registry.
- Mitigation: explicit cleanup after unmatched-left emission; add assertions for empty registry at end of execution.

**Risk:** interaction with existing right-side global bitmap accumulation (EmitGlobalRightUnmatched).
- Mitigation: ensure chunk state cleanup doesn't interfere; test FULL JOIN scenarios thoroughly.

**Risk:** order-of-operations bugs across multiple partitions.
- Mitigation: atomic operations for counter, interior mutability for registry, and deterministic unit tests with repeatable data.

## Non-goals

- Changing optimizer join selection or reorder rules.
- Redesigning NestedLoopJoin algorithm outside memory-limited fallback.
- Eliminating per-partition probes (retaining existing parallelism).

## Definition of done

1. Implementation passes all new unit + optional SLT tests.
2. Existing test suite (including NEW_ISSUE_01 regression test) green.
3. Memory-limited fallback re-enabled for LEFT-producing joins with multi-partition right side.
4. Code comments document chunk-state synchronization invariants.
5. PR description explains cross-partition coordination and references this issue.

## Related issues / future work

- Investigate lock-free atomic implementation if contention observed
- Consider upstreaming shared chunk-state pattern to other multi-partition operators
