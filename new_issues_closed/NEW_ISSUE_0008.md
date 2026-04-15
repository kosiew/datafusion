Partly valid, but weak / over-scoped as written.

# Return structured extracted-filter metadata from `try_transform_to_simple_table_scan_with_filters`

## Summary

`try_transform_to_simple_table_scan_with_filters` currently returns a bare `Vec<Expr>` alongside the transformed plan. That keeps the helper mechanically simple, but it pushes semantic decisions onto every caller: each caller has to decide how to combine those expressions, where to emit them in the generated SQL, and how they interact with any existing predicates.

The unparser now has enough clause-placement logic around extracted filters that a plain vector is starting to underspecify the contract. Returning a small structured type would make the API clearer and reduce ad hoc recombination at the call site.

## Is your feature request related to a problem or challenge?

Yes.

The current helper signature exposes extracted filters as raw expressions with no metadata about intended handling. That creates a few problems:

- callers must decide on their own whether the filters should be emitted into `WHERE`, `ON`, or some other clause,
- callers must rebuild combination logic such as `AND` folding,
- ownership and control flow become more awkward because clause-placement decisions happen after extraction,
- future semantics, such as dialect constraints or richer join-policy decisions, have no natural place to live.

This is already visible in the join unparser path, where extracted filters are collected as a `Vec<Expr>` and then reinterpreted later based on join type.

## Describe the solution you'd like

Change `try_transform_to_simple_table_scan_with_filters` to return a small structured result rather than a bare `Vec<Expr>`.

For example, the helper could return a dedicated type that contains:

- the transformed simple table-scan plan,
- the extracted predicates,
- enough structure or metadata to describe intended clause handling or future routing policy.

The structure does not need to be over-designed up front. Even a modest type that groups the extracted predicates and centralizes combination behavior would be an improvement over a raw vector.

The main goal is to move from:

- “here are some expressions, figure out what they mean later”

to:

- “here is the transformed scan plus extracted filter information with an explicit contract.”

## Acceptance criteria

- `try_transform_to_simple_table_scan_with_filters` no longer returns a bare `Vec<Expr>` for extracted predicates.
- Call sites become simpler because they do not need to reconstruct as much meaning from raw expressions.
- The join unparser path in `datafusion/sql/src/unparser/plan.rs` becomes easier to read, with less ad hoc predicate recombination.
- The new structure leaves room for future semantics such as clause placement or dialect constraints without another round of API churn.
- Existing unparser behavior and tests remain unchanged unless a follow-up explicitly expands functionality.

## Describe alternatives you've considered

- Keep returning `Vec<Expr>` and add helper functions at each call site. This avoids changing the helper API, but it leaves the core contract implicit.
- Return a single combined expression instead of a vector. That may simplify some callers, but it still loses semantic structure and can make later policy decisions harder.
- Solve this only in the join unparser. That reduces local duplication, but the extraction helper would still expose an underspecified interface to future callers.

## Additional context

- The current helper lives in `datafusion/sql/src/unparser/utils.rs`.
- One visible downstream consumer is the join path in `datafusion/sql/src/unparser/plan.rs`, which currently collects the raw expressions and then decides how to combine and place them.
- This issue is intended as an API/maintainability refactor, not as a behavior change by itself.
- Related PR review context: `PR_REVIEW_01.md`.
