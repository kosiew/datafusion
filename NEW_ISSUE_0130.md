source: memcalc-03-23393a
# Refactor symmetric hash join stream orchestration to an async generator

## Problem
`SymmetricHashJoinStream` in `datafusion/physical-plan/src/joins/symmetric_hash_join.rs` implements orchestration as a manual `poll_next` state machine: `SHJStreamState`, `poll_next_impl`, fetch/end handlers, and explicit `Poll` propagation. The join algorithm is distributed across control-flow transitions, making lifecycle changes—especially transformer retention, reservation refreshes, final-result handling, and cancellation—hard to review safely.

## Why it matters
The current state machine is correct but has a high maintenance cost. A local change to when a batch is retained, emitted, or released must preserve several coupled stream boundaries. More direct asynchronous control flow could make those boundaries visible without changing join semantics.

## Invariant / desired behavior
The refactor preserves all observable symmetric-hash-join behavior:

- Every input batch is processed once; outputs, ordering properties, and errors are unchanged.
- Either-side exhaustion and final unmatched-row handling emit each required result exactly once, then terminate.
- `emit(...).await` preserves output backpressure and cancellation behavior.
- The reservation continues to cover all stream-retained state, including transformer-held batches, and refreshes when that ownership changes.
- Metrics, cleanup, memory-limit failures, and dropped-stream resource release remain unchanged.

## Proposed direction
First prove that `datafusion_execution::async_try_stream` can own the required input streams and state without weakening cancellation or reservation behavior. If viable, express the orchestration loop with awaited input batches and `emitter.emit(batch).await`, while retaining the existing join-processing, cleanup, transformer, and accounting helpers. Remove manual polling state only after equivalent behavior is covered.

## Scope
### In
- Refactor stream orchestration in `datafusion/physical-plan/src/joins/symmetric_hash_join.rs`.
- Compare the generator lifecycle against current state transitions: input fetch, exhaustion, final results, transformer batching, reservation updates, metrics, and cancellation.
- Add focused regressions where current tests do not pin those boundaries.

### Out
- Changes to join algorithms, join-type semantics, ordering, pruning, or output schemas.
- Changes to `BatchTransformer`, `OneSideHashJoiner`, or shared memory-counter formulas.
- Public API, proto, or configuration changes.
- Unrelated join-stream refactors.

## Acceptance criteria
- [ ] The async-generator approach is validated as capable of preserving stream ownership, cancellation, and backpressure; otherwise the issue records the blocking constraint and closes without a partial rewrite.
- [ ] Existing symmetric-hash-join result tests pass unchanged across inner, outer, semi, anti, and mark joins.
- [ ] Empty inputs, either-side exhaustion, and final unmatched results match pre-refactor output and termination behavior.
- [ ] Input errors, metrics, bounded-pool failures, and stream-drop cleanup remain behaviorally equivalent.
- [ ] Transformer splitting and retained-batch reservation lifecycle regressions pass.
- [ ] Obsolete manual polling state is removed rather than retained alongside a second orchestration path.

## Tests / verification
- Focused unit tests for input/error/exhaustion/final-result paths and transformer reservation retain/release.
- Existing symmetric hash join unit and bounded-memory regressions.
- `cargo test -p datafusion-physical-plan --lib symmetric_hash_join`
- `cargo test -p datafusion-physical-plan --lib`
- `cargo fmt --all -- --check`

## Notes / open questions
- Confirm whether the generator can hold the required mutable join state and both input streams while preserving the current cancellation/drop behavior before committing to the rewrite.
