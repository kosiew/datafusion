created #22163
source: pr-22077_a
# Refactor: Centralize numeric `%c` formatting dispatch in format_string.rs

## Overview
The `ConversionSpecifier::format` method in `datafusion/spark/src/function/string/format_string.rs` contains highly repetitive branches for handling `%c` (character) conversion across all signed and unsigned integer scalar types (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64). Each branch duplicates the same validation and conversion logic, making the code difficult to maintain and prone to inconsistency when invariants change.

## Problem Statement
Currently, the code has near-identical match arms for each numeric type:
- `Int8`, `Int16`, `Int32`, `Int64`: All call `signed_to_char(*value as i64)`
- `UInt8`, `UInt16`, `UInt32`, `UInt64`: All call `unsigned_to_char(*value as u64)`

**Maintenance burden**: When a new invariant or fix must be applied to `%c` handling (as happened in PR #22077), every branch must be updated individually, increasing the risk of missed updates and regressions.

**Example of duplication**:
```rust
(ConversionType::CharLower | ConversionType::CharUpper, Some(value)) => {
    self.format_char(string, signed_to_char(*value as i64)?)
}
```
This pattern appears ~8 times with only the type changed.

## Solution
Create a helper dispatch mechanism that unifies the conversion logic once, eliminating duplication:

### Option A: Helper trait + blanket impl
Define a trait for types that can be converted to `char` for `%c` formatting:
```rust
trait FormatCharable {
    fn to_char_for_format(&self) -> Result<char>;
}

impl FormatCharable for i64 { /* signed logic */ }
impl FormatCharable for u64 { /* unsigned logic */ }
```

Then in the match arms, call a single method that dispatches to the appropriate helper.

### Option B: Match-arm extraction helper
Create a focused helper that centralizes the signed/unsigned decision:
```rust
fn format_codepoint(
    &self,
    string: &mut String,
    value: Option<i64>,
    is_unsigned: bool,
) -> Result<()> {
    match (value, is_unsigned) {
        (Some(v), false) => self.format_char(string, signed_to_char(v)?),
        (Some(v), true) => self.format_char(string, unsigned_to_char(v as u64)?),
        (None, _) => self.format_string(string, "null"),
    }
}
```

### Option C: Inline macro or function for match arm pattern
Use a macro to reduce boilerplate in the match itself (less clean but simpler).

## Benefits
1. **Maintainability**: Single source of truth for `%c` validation and conversion logic
2. **Safety**: Reduced risk of missed updates when fixing bugs or adding invariants
3. **Clarity**: Easier to understand the unified behavior across all numeric types
4. **Testability**: Easier to add comprehensive tests for the dispatch logic in isolation
5. **Performance**: No runtime overhead; can be inlined or const-evaluated

## Scope
- **File**: `datafusion/spark/src/function/string/format_string.rs`
- **Affected code**: `ConversionSpecifier::format` match arms for all numeric types
- **Tests**: Existing unit tests in the same file; no new test files needed
- **Breaking changes**: None (internal refactor)

## Effort Estimate
**Medium** (1–2 days):
- Identify all affected match arms
- Design and implement the dispatch helper
- Verify all numeric types still pass existing tests
- Optional: Add additional edge-case coverage

## Related Issues
- PR #22077 (format_string.rs recent changes)
- Context: `signed_to_char()` and `unsigned_to_char()` functions provide the core validation logic

## Definition of Done
- [ ] Dispatch helper extracted and tested with all numeric scalar types
- [ ] All existing unit tests pass without modification
- [ ] Code review confirms no logic changes (pure refactor)
- [ ] Optional: Add SLT case for edge case like `1114112` (code point above max valid)
