source: dynamic-filter-01-22772a
# Consolidate shared-bounds partitioned test fixtures without changing callers

## Problem
`shared_bounds.rs` now has more than one partitioned accumulator fixture:
- existing `pub(super) make_partitioned_accumulator_for_test`, used for lifecycle-oriented tests and built with empty `on_right`;
- private expression-test fixture setup that needs real `on_right` expressions and schema-compatible dynamic-filter children.

The duplication is small, but future tests may add more fixture variants. A naive cleanup that changes the exported helper defaults could break callers or blur the difference between lifecycle tests and expression-shape tests.

## Why it matters
Test fixtures encode important assumptions about accumulator state, dynamic-filter children, schema, and partition count. Changing a shared fixture can silently alter what tests are proving, especially when a fixture is visible outside the local test module.

## Invariant / desired behavior
Existing callers of `make_partitioned_accumulator_for_test` must keep the same behavior unless every call site is audited and intentionally updated. Expression-policy tests that require real `on_right` expressions should not change lifecycle-test fixture semantics.

## Proposed direction
Prefer a private low-level builder that accepts the variable parts explicitly:
- accumulated data/mode;
- `on_right` expressions;
- dynamic-filter children/schema if needed.

Then keep wrappers with clear names for each fixture role:
- lifecycle partitioned accumulator with empty `on_right`;
- expression-policy partitioned accumulator with real `on_right`.

Only alter the existing `pub(super)` helper if it can remain backward-compatible or all call sites are proven local and updated intentionally.

## Scope
### In
- Audit uses of `make_partitioned_accumulator_for_test`.
- Consolidate duplicated fixture construction where safe.
- Preserve fixture defaults for existing callers.
- Keep helper visibility as narrow as possible.

### Out
- No production code changes.
- No change to accumulator behavior, dynamic-filter semantics, or expression policy.
- No broad rewrite of hash join test infrastructure.
- No public API changes.

## Acceptance criteria
- [ ] Existing callers of `make_partitioned_accumulator_for_test` compile and pass unchanged, or all changes are mechanical and justified by a call-site audit.
- [ ] Fixture names make the lifecycle-vs-expression-test distinction clear.
- [ ] Shared construction removes duplication without hiding important defaults (`on_right`, schema, seed, initial `lit(true)` filter).
- [ ] Helper visibility is not broadened.

## Tests / verification
- Search call sites: `rg "make_partitioned_accumulator_for_test"`.
- `cargo test -p datafusion-physical-plan shared_bounds`.
- If call sites outside `shared_bounds.rs` are affected, run their targeted tests too.

## Notes / open questions
- If the only duplication is inside local tests, this may not be worth a standalone PR. Prefer doing it only when adding more fixture variants or touching nearby tests.
