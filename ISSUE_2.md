# [Discussion] Case-insensitive field matching in struct casting

## Problem

This is a **design discussion** issue regarding whether DataFusion should adopt **case-insensitive field matching** when casting between structs.

### Current Behavior
DataFusion uses **case-sensitive** field name matching. For example:
- Field `x` and field `X` are treated as different fields
- A cast from `struct<x int, y int>` to `struct<X int, Y int>` would fail (no name overlap)

### DuckDB Behavior
DuckDB uses **case-insensitive** field name matching:
- Field `x` and field `X` are treated as the same field
- The same cast would succeed, with fields matched case-insensitively

## Motivation for Case-Insensitive Matching

**Pros:**
- ✅ **Aligns with DuckDB** — improves compatibility with a major SQL database
- ✅ **More forgiving** — handles common casing variations (e.g., JSON sources with inconsistent field names)
- ✅ **Follows SQL conventions** — SQL generally treats identifiers as case-insensitive
- ✅ **User-friendly** — reduces friction when working with data from different sources

## Arguments for Keeping Case-Sensitive Matching

**Pros:**
- ✅ **Arrow foundation** — DataFusion is built on Apache Arrow, which is case-sensitive:
  - [`Fields::find()`](https://github.com/apache/arrow-rs/blob/main/arrow-schema/src/fields.rs#L83) uses exact string equality: `b.name() == name`
  - [`StructArray::column_by_name()`](https://github.com/apache/arrow-rs/blob/main/arrow-array/src/array/struct_array.rs#L324) compares with `==`: `c == &column_name`
  - [`Field` equality](https://github.com/apache/arrow-rs/blob/main/arrow-schema/src/field.rs#L109) uses direct comparison: `self.name == other.name`
- ✅ **Language consistency** — matches Rust and JSON conventions (case-sensitive)
- ✅ **Prevents ambiguity** — avoids edge cases where source has both `x` and `X` (rare but possible)
- ✅ **Predictable behavior** — case-sensitive matching is more explicit and easier to reason about in programmatic contexts
- ✅ **Conservative approach** — can always relax to case-insensitive later; harder to go the other direction

## Design Question

**Should DataFusion follow SQL's case-insensitivity or remain aligned with Arrow's case-sensitive semantics?**

This decision has broader architectural implications:

1. **Type system consistency** — Should struct field matching be case-insensitive throughout DataFusion?
2. **Identifier resolution** — How would this interact with column name resolution in queries?
3. **Schema merging** — Would case-insensitive matching apply to other schema operations?
4. **Performance** — What's the cost of case-insensitive string comparisons at scale?

## Recommendation

**For now, maintain case-sensitive matching** because:

1. Arrow (DataFusion's foundation) is case-sensitive
2. It's more conservative and prevents ambiguous matches
3. Users can explicitly handle case differences with casts or field renames
4. This decision can be revisited later if community consensus emerges for case-insensitivity
5. It requires community discussion as it affects the broader type system

A change to case-insensitive matching should be a **deliberate architectural decision** with:
- Clear community consensus
- Consistent application across the entire type system
- Performance impact analysis
- Migration path for existing users

## Next Steps

This issue is intended to **surface the design question** and gather community feedback. 

### Before Implementation
- [ ] Community discussion on case-sensitivity philosophy for DataFusion
- [ ] Consensus on whether this aligns with DataFusion's design principles
- [ ] If consensus emerges for case-insensitivity, document the broader implications
- [ ] Identify all subsystems that would be affected by this change

### If Decision is Made to Implement Case-Insensitive Matching
- [ ] Create a separate follow-up issue with acceptance criteria
- [ ] Update field matching logic in `fields_have_name_overlap()` and related functions
- [ ] Ensure consistency with identifier resolution throughout the system
- [ ] Add comprehensive tests covering mixed-case scenarios
- [ ] Update documentation to clarify case-insensitive semantics

## Related Issues

- **ISSUE_1**: Eliminate positional fallback in struct casting (should be implemented first, independently)
- **ISSUE_3**: Validate non-nullable missing fields in struct casting (separate correctness fix)

## Context

This issue stems from a comprehensive review of struct casting semantics to align with DuckDB (see `PR_RESPONSE.md`, section "2. Consider case-insensitive matching - DuckDB treats x and X as matching").

The broader struct casting effort includes:
1. ✅ ISSUE_1: Eliminate positional fallback (agreed, high priority)
2. ⏳ ISSUE_2: Case-insensitive matching (this issue — needs discussion)
3. ✅ ISSUE_3: Validate non-nullable missing fields (agreed, high priority)

Recommendation: Address ISSUE_1 and ISSUE_3 first (clear improvements), then revisit ISSUE_2 after community input on broader type system design.
