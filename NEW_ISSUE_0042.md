source: pr-22358_a
# Issue: Centralize shared-allocation accounting for Arc DFHeapSize implementations

## Type
Refactor / maintainability

## Priority
Medium

## Source
- Review note in PR_REVIEW_01.md (High-impact refactor opportunities, out of scope)
- Origin: pre-existing before PR #22358

## Problem Statement
The code currently has three separate `DFHeapSize` implementations for Arc-backed values:
- `impl<T: DFHeapSize> DFHeapSize for Arc<T>`
- `impl DFHeapSize for Arc<str>`
- `impl DFHeapSize for Arc<dyn DFHeapSize>`

Each implementation repeats the same shared-allocation deduplication control flow:
1. Extract an allocation identity pointer.
2. Insert pointer into `DFHeapSizeCtx.seen`.
3. Return `0` if already seen.
4. Compute type-specific payload size.

This repetition duplicates the same core invariant in multiple places and increases risk that one implementation diverges from the others over time.

## Why This Matters
`DFHeapSize` depends on a critical invariant: shared heap allocations must be counted once per traversal context. If Arc implementations drift, memory accounting becomes inconsistent and hard to reason about.

Consolidating the one-time-accounting logic improves:
- Correctness durability (single source of truth for Arc dedup behavior)
- Maintainability (less repeated logic)
- Reviewability (future changes touch one helper)

## Current Code Locations
- datafusion/common/src/heap_size.rs
  - Arc<T> impl around line 284
  - Arc<str> impl around line 297
  - Arc<dyn DFHeapSize> impl around line 310

## Proposed Change
Introduce a small internal helper for Arc allocation deduplication, for example:
- Input: allocation identity (usize pointer), mutable `DFHeapSizeCtx`
- Output: whether this allocation should be counted now

Pseudo-shape:
- `fn count_allocation_once(ptr: usize, ctx: &mut DFHeapSizeCtx) -> bool`
  - returns `true` on first sighting
  - returns `false` if already accounted

Then update all Arc implementations to:
1. Compute allocation identity pointer in impl-specific way.
2. Call helper and early-return `0` when already seen.
3. Keep each impl's payload-size math local and explicit.

## Important Constraints
- Do not force one shared payload formula across Arc variants.
- Keep unsized and trait-object cases explicit (`Arc<str>`, `Arc<dyn DFHeapSize>`).
- Preserve behavior exactly; this is a refactor, not a semantic change.

## Non-Goals
- Changing ownership semantics or documented `DFHeapSize` contract.
- Reworking non-Arc implementations.
- Broad API redesign.

## Acceptance Criteria
1. Arc implementations no longer duplicate `ctx.seen.insert` control flow.
2. Existing Arc-related tests continue to pass without behavior changes.
3. Readability is improved: dedup invariant is obvious and centralized.
4. No regression in docs or test expectations for heap-size semantics.

## Validation Plan
Run targeted tests:
- `cargo test -p datafusion-common heap_size --lib`
- `cargo test -p datafusion-common --doc heap_size`

If practical, ensure clone/dedup Arc tests still cover:
- Arc<T>
- Arc<str>
- Arc<dyn DFHeapSize>

## Risks and Mitigations
- Risk: incorrect pointer identity abstraction for unsized/trait object Arc types.
  - Mitigation: keep pointer extraction per impl, centralize only seen-set insertion gate.
- Risk: accidental behavior change masked as refactor.
  - Mitigation: rely on existing Arc clone-accounting tests and doc tests.

## Suggested Follow-up
After this refactor lands, consider small cleanup to reduce repetitive pointer-casting style while preserving type clarity in each Arc impl.
