closed
# Issue Proposal: Add first-class sqllogictest support for asserting output column names

## Summary

DataFusion's `sqllogictest` harness can assert row values and plan text, but it does not currently provide a direct way to assert output column names. That limitation makes schema-regression tests much harder to write and read, especially when the bug is about aliases or runtime-emitted field names rather than row contents.

PR #21770 adds a regression in `cte.slt` for a recursive CTE schema leak. Because SLT cannot assert output headers directly, the test has to:

- write the query result to CSV with headers enabled
- read the file back as headerless CSV
- surface the header row as data
- compare that data row against the expected column names

The workaround is effective, but it is indirect and heavier than the behavior under test. This issue proposes first-class sqllogictest support for asserting output column names directly.

## Motivation

Column names are part of observable query behavior. They affect:

- CLI display and dataframe presentation
- CSV / JSON / Arrow serialization
- downstream SQL built on subqueries, views, and CTEs
- user code that inspects schemas on returned batches

Bugs in this area are easy to miss because:

- values can still be fully correct
- the logical plan schema can look correct while emitted batches drift at runtime
- existing SLT coverage naturally emphasizes row data over output schema details

This means contributors often need to choose between:

- a Rust integration test that inspects `RecordBatch` schemas directly
- a workaround-based `.slt` test using file round-trips
- no end-to-end SQL regression at all

That is a poor ergonomics tradeoff for an engine where aliases and output naming are user-visible semantics.

## Problem Statement

There is no straightforward way in current DataFusion sqllogictest files to say:

- "the output columns must be `id, parent_id, ts, val`"
- "this alias must be preserved"
- "runtime-emitted batches must not leak child-branch field names"

Without a first-class assertion mechanism:

- schema-sensitive tests become indirect
- tests are noisier and less local to the behavior they validate
- output-schema regressions are less likely to get concise `.slt` coverage

## Proposed Direction

Extend the DataFusion sqllogictest harness so tests can assert output column names directly.

The initial version should stay intentionally small:

- assert names only
- assert them in output order
- operate on runtime query results, not only logical plan text

Possible syntax directions:

### Option A: A dedicated directive

```text
query_columns TTTT
SELECT id, parent_id, ts, val FROM descendants;
----
id parent_id ts val
```

This is likely the clearest for test authors if the parser impact is manageable.

### Option B: Extend existing query directives

```text
query TTTT headers
SELECT id, parent_id, ts, val FROM descendants;
----
columns: id parent_id ts val
...
```

This may reduce parser surface area, though it is somewhat less elegant.

### Option C: Separate schema assertion block

```text
statement schema
SELECT id, parent_id, ts, val FROM descendants;
----
id parent_id ts val
```

The exact shape can be decided during implementation, but readability in `.slt` files should be the primary criterion.

## Scope

In scope:

- add direct output-column-name assertions to the DataFusion sqllogictest harness
- validate names from actual runtime query results
- add clear mismatch reporting
- migrate at least one motivating regression away from the CSV round-trip workaround
- document the feature for contributors

Out of scope:

- a full schema assertion language covering nullability, metadata, and Arrow extension semantics
- broad migration of every existing schema-adjacent test
- unrelated sqllogictest grammar redesign

## Design Considerations

### 1. Validate runtime-visible behavior

The feature should inspect the schema that execution actually returns. The recursive CTE bug motivating this issue was specifically about runtime-emitted batch schemas diverging from the plan-level schema.

### 2. Handle multi-batch output intentionally

The harness should define whether it:

- checks only the first batch's schema
- checks all batches for consistency

Checking all batches is more valuable for catching runtime drift bugs like the one in PR #21770, where different batches can expose different names.

### 3. Keep authoring lightweight

The feature should remove the need for side effects like `COPY` and scratch-file rereads when the test only cares about output headers.

### 4. Produce review-friendly diffs

Failures should present expected and actual column-name lists clearly and compactly.

## Migration Opportunity

Once available, this feature would let the new recursive CTE regression in `cte.slt` be rewritten in a much simpler form. More broadly, it would make it easier to cover:

- alias propagation bugs
- output naming across recursive / union-style operators
- projection rename regressions
- serializer-facing header correctness where output names matter

## Testing Plan

Add harness-level tests for:

- exact match on a simple query
- mismatch reporting when a single column name differs
- alias preservation
- multi-batch results with stable schemas
- multi-batch results with drifted schemas, if the chosen implementation checks all batches

Add at least one end-to-end `.slt` regression migrated from a current workaround-based approach.

## Expected Benefits

- simpler and clearer `.slt` regressions for schema bugs
- stronger end-to-end SQL coverage for output-schema behavior
- fewer file-based testing workarounds
- lower friction for adding regression coverage in alias- and schema-sensitive areas

## Acceptance Criteria

- sqllogictest supports direct assertion of output column names
- mismatch output clearly shows expected vs actual names
- the recursive CTE regression from PR #21770 can be expressed without the CSV header round-trip
- contributor documentation covers the new assertion style

## Recommendation

Close this for now.

The motivating gap is real, and the workaround in `cte.slt` demonstrates that direct output-column-name assertions would be nicer to author and review. However, the implementation is likely larger and less local than the issue description suggests.

Supporting reasons:

- The current DataFusion sqllogictest runner only returns column types and row values. It does not carry output field names through query execution.
- DataFusion depends on the upstream `sqllogictest` crate, and that crate's current parser and `DBOutput` model also only represent query output as column types plus rows. There is no existing first-class place to represent asserted or actual output column names.
- Because of that boundary, a clean implementation likely requires upstream `sqllogictest` parser and runner changes, or else a DataFusion-specific fork / compatibility layer. That is more of a cross-project testing-infrastructure change than a localized DataFusion refactor.
- The impact is mainly contributor ergonomics and test readability, not engine correctness or end-user functionality. The feature would be useful, but it is better described as medium-impact infrastructure work than a high-impact product change.
- Critical regressions in this area are already coverable today via targeted Rust tests that inspect `RecordBatch` schemas directly, and when necessary via the existing SLT workaround used in PR #21770.

In short: this is a valid idea, but it does not look like a good near-term DataFusion-only refactor to pursue unless there is appetite to make corresponding upstream `sqllogictest` changes.

## Origin

This refactor opportunity predates PR #21770. The PR simply made the tooling gap obvious because it needed end-to-end SQL coverage for a runtime header/schema regression.
