source: filtering-logic-01-22665a
# Refactor Issue 01: Centralize Grouped Aggregate FILTER Pass/Fail Checks Beyond `first_value` / `last_value`

## Summary

Grouped aggregate implementations should use one shared representation of aggregate `FILTER` pass/fail semantics: a row passes only when the predicate is `Some(true)`. `Some(false)` and `None` both reject the row.

The recent `first_value` / `last_value` fix moved one grouped path to `filter_to_validity`, but other grouped accumulators may still have bespoke checks against `BooleanArray` values or hand-written null handling. This issue is an audit and incremental migration to reduce duplicated logic and prevent future FILTER null-semantics drift.

## Core Invariant

Aggregate `FILTER` semantics must be:

| Predicate value | Predicate validity | Passes? |
| --- | --- | --- |
| `true` | valid | yes |
| `false` | valid | no |
| `true` | NULL | no |
| `false` | NULL | no |

The implementation-level invariant is:

```rust
filter_to_validity(filter).value(row_idx)
```

must be the source of truth for deciding whether a grouped aggregate FILTER row passes, unless the path can prove it already receives a precomputed filter validity bitmap with identical semantics.

## Problem

DataFusion currently has a shared helper:

```rust
// datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/nulls.rs
pub fn filter_to_validity(filter: &BooleanArray) -> BooleanBuffer {
    let Some(filter_nulls) = filter.nulls() else {
        return filter.values().clone();
    };
    filter.values() & filter_nulls.inner()
}
```

This encodes the desired `Some(true)` semantics. However, grouped accumulator code has historically had local checks such as:

```rust
filter.value(idx)
```

or equivalent bespoke branches. Those checks are easy to get wrong because `BooleanArray::value(idx)` reads the value bit and does not itself mean the predicate is non-null. A NULL predicate row can still have a true value bit in the underlying Arrow buffers.

The `first_value` / `last_value` grouped path was fixed by precomputing:

```rust
let filter_validity = opt_filter.map(filter_to_validity);
```

and then checking the validity bitmap in the per-row loop. Other grouped aggregate paths should be audited and migrated where applicable.

## Evidence

Recent fix location:

- `datafusion/functions-aggregate/src/first_last.rs`
  - `FirstLastGroupsAccumulator::get_filtered_extreme_of_each_group`
  - now routes FILTER checks through `filter_to_validity`.

Existing shared helper location:

- `datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/nulls.rs`
  - `filter_to_validity(filter)` returns `values & validity`.

Existing common-path usage:

- `datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/accumulate.rs`
  - grouped accumulation helpers already use `filter_to_validity` in at least some paths.

Known risk pattern:

```rust
let filter = BooleanArray::new(
    BooleanBuffer::from(vec![true]),
    Some(NullBuffer::from(vec![false])),
);
```

A direct `filter.value(0)` check returns the value bit, but aggregate FILTER semantics must reject this row because the predicate is NULL.

## Scope

In scope:

- Audit grouped aggregate FILTER pass/fail checks in:
  - `datafusion/functions-aggregate/src/`
  - `datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/`
- Replace obvious local `BooleanArray` row-pass checks with shared `filter_to_validity` semantics.
- Keep value-null handling separate from filter-null handling.
- Add targeted Rust tests for every migrated path.
- Document audited files in the PR description.

Out of scope:

- Broad aggregate algorithm rewrites.
- Non-grouped accumulator rewrites unless a small, obvious FILTER bug is found.
- Performance redesigns or iterator abstraction rewrites.
- SQLLogicTest-only coverage for Arrow value-bit-under-null behavior; SQL may not reliably expose those low-level buffers.
- Public API reshaping beyond using the existing public helper.

## Candidate Audit Commands

Start with focused searches:

```bash
grep -R "opt_filter" datafusion/functions-aggregate datafusion/functions-aggregate-common -n
grep -R "filter.value\|\.value(idx\|\.value(i\|is_none_or" datafusion/functions-aggregate datafusion/functions-aggregate-common -n
grep -R "filter_to_validity" datafusion/functions-aggregate datafusion/functions-aggregate-common -n
```

Then classify each match as:

1. already uses `filter_to_validity`
2. safe because the input is already a validity bitmap with `Some(true)` semantics
3. unsafe or duplicated FILTER pass/fail logic
4. unrelated to aggregate FILTER semantics

## Proposed Implementation Plan

### Step 1: Build an audit table

Create a short PR-local table, for example in the PR description:

| File/function | FILTER handling | Classification | Action |
| --- | --- | --- | --- |
| `first_last.rs::get_filtered_extreme_of_each_group` | `filter_to_validity` | migrated | no action |
| `...` | `filter.value(idx)` | unsafe | migrate |
| `...` | no filter handling | unrelated | no action |

Do not change code before classifying enough context to avoid confusing internal `is_set` bitmaps with aggregate FILTER predicates.

### Step 2: Migrate only simple, local cases

Preferred shape:

```rust
let filter_validity = opt_filter.map(filter_to_validity);

for row_idx in 0..len {
    let passed_filter = filter_validity
        .as_ref()
        .is_none_or(|validity| validity.value(row_idx));

    if !passed_filter {
        continue;
    }

    // existing value/null/state logic unchanged
}
```

Rules:

- Precompute once per batch, not inside the per-row loop.
- Do not replace internal state bitmaps such as `is_set_arr` unless they are actually aggregate FILTER predicates.
- Do not merge unrelated null-handling branches.
- Preserve existing ordering, group indexing, and state-update behavior.

### Step 3: Add regression tests per migrated path

For every changed accumulator path, add a direct Rust test with a nullable filter whose value bit is true under NULL validity:

```rust
let filter = BooleanArray::new(
    BooleanBuffer::from(vec![true, true, false]),
    Some(NullBuffer::from(vec![false, true, true])),
);
```

Expected pass rows: only row 1.

Test both when useful:

- no rows pass -> result remains unset / NULL
- one non-null `Some(true)` row passes -> result comes only from that row

### Step 4: Split if the audit finds too much

If more than 2-3 accumulator families need changes, split by family:

- one PR for common helper paths
- one PR per aggregate family with custom grouped logic
- one PR for test-only SQL coverage, if needed

## Validation

Run targeted tests for changed crates first:

```bash
cargo test -p datafusion-functions-aggregate-common --lib
cargo test -p datafusion-functions-aggregate --lib
```

If SQLLogicTest files are touched:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate.slt
```

Before marking ready:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

## Risks

Correctness risk: low if changes only replace raw filter checks with `filter_to_validity` and preserve all state/value logic.

Regression risk: medium because grouped accumulators are hot execution paths and some bitmaps may not represent aggregate FILTER predicates. Audit before changing.

Performance risk: low/medium. `filter_to_validity` can allocate/combine a bitmap when the filter has nulls, but it avoids repeated ad-hoc validity checks and should be computed once per batch.

Scope risk: high if this becomes a broad accumulator cleanup. Keep migration PR-sized and split findings.

## Acceptance Criteria

- Grouped aggregate FILTER pass/fail checks are audited in the aggregate crates.
- Obvious unsafe local checks are migrated to `filter_to_validity` semantics.
- Internal non-FILTER bitmaps are left unchanged.
- Each changed path has a Rust regression test with true value bits under NULL validity.
- PR description lists checked files and classifications.
- No broad public API changes or unrelated rewrites.
