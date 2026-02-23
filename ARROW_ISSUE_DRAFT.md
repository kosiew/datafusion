# Arrow Issue: Reduce `'static` lifetime constraints in `FormatOptions` and `CastOptions`

## Summary

Arrow's `FormatOptions<'static>` and `CastOptions<'static>` require all string fields to be `&'static str` references, which prevents runtime-created format strings and makes it difficult for downstream projects to work with dynamic formatting options. DataFusion and other consumers must create owned wrapper types to work around this limitation.

## Problem

### Current API Design

```rust
// arrow/compute/cast.rs (simplified)
pub struct FormatOptions<'a> {
    pub null: &'a str,
    pub date_format: Option<&'a str>,
    pub datetime_format: Option<&'a str>,
    pub timestamp_format: Option<&'a str>,
    // ... other string fields as &'a str
}

pub struct CastOptions<'a> {
    pub safe: bool,
    pub format_options: FormatOptions<'a>,
}
```

### Pain Points

1. **Requires statically-known strings:** Users cannot dynamically construct format strings (e.g., from TOML config, environment variables, or user input).

2. **Downstream duplication:** Projects like DataFusion must create parallel owned types to support runtime options:
   ```rust
   // In DataFusion (workaround)
   pub struct OwnedFormatOptions {
       pub null: String,
       pub date_format: Option<String>,
       pub datetime_format: Option<String>,
       // ...
   }
   ```

3. **Conversion overhead:** Every use requires manual borrowing or temporary lifetime management:
   ```rust
   let owned = OwnedFormatOptions::new();
   let borrowed = owned.as_arrow_options();  // Creates temporary references
   arrow::compute::cast(&array, &type, Some(&borrowed))?;
   ```

4. **Maintenance burden:** Owned wrappers must be kept in sync with Arrow's API, adding maintenance overhead.

## Proposed Solutions

### Option 1: Owned version in Arrow (preferred)

Add an `OwnedFormatOptions` and `OwnedCastOptions` directly to the Arrow crate:

```rust
pub struct OwnedFormatOptions {
    pub null: String,
    pub date_format: Option<String>,
    pub datetime_format: Option<String>,
    pub timestamp_format: Option<String>,
    pub timestamp_tz_format: Option<String>,
    pub time_format: Option<String>,
    pub duration_format: DurationFormat,
    pub types_info: bool,
}

impl OwnedFormatOptions {
    pub fn as_format_options(&'_ self) -> FormatOptions<'_> {
        FormatOptions::new()
            .with_null(self.null.as_str())
            .with_date_format(self.date_format.as_deref())
            // ... other fields
    }
}

pub struct OwnedCastOptions {
    pub safe: bool,
    pub format_options: OwnedFormatOptions,
}
```

**Advantages:**
- Single source of truth for cast/format options
- Eliminates downstream duplication
- Cleaner API for runtime-created options

**Disadvantages:**
- Adds to Arrow's API surface

### Option 2: Relax `'static` to arbitrary lifetime

Change the API to use generic lifetimes instead of `'static`:

```rust
pub struct FormatOptions<'a> {
    pub null: Cow<'a, str>,
    pub date_format: Option<Cow<'a, str>>,
    // ...
}
```

**Advantages:**
- Supports both static and dynamic strings
- Smaller API footprint

**Disadvantages:**
- `Cow<'a, str>` adds indirection and complexity
- May break existing code using `'static` assumptions

### Option 3: Builder pattern with `String` values (least breaking)

Keep the existing API but add a builder that works with owned strings:

```rust
pub struct FormatOptionsBuilder {
    null: String,
    date_format: Option<String>,
    // ...
}

impl FormatOptionsBuilder {
    pub fn build<'a>(&'a self) -> FormatOptions<'a> {
        FormatOptions::new()
            .with_null(self.null.as_str())
            // ...
    }
}
```

## Impact on DataFusion

DataFusion is implementing `CastColumnExpr`, which requires schema-aware casting with runtime-created format options (e.g., for cases where a user specifies a format string via SQL or config). Currently, DataFusion works around this by maintaining a parallel `OwnedFormatOptions` type that mirrors Arrow's structure.

With upstream support for owned options, DataFusion could:
- Remove the parallel `OwnedFormatOptions` wrapper
- Simplify serialization/deserialization logic in proto handling
- Reduce maintenance burden of tracking Arrow API changes

## References

- DataFusion PR #20202: [CastColumnExpr with schema-aware validation](https://github.com/apache/datafusion/pull/20202)
- DataFusion `OwnedFormatOptions`: [datafusion/common/src/format.rs](https://github.com/apache/datafusion/blob/main/datafusion/common/src/format.rs)

## Questions for Arrow maintainers

1. Is this a known limitation?
2. Which approach (Option 1, 2, or 3) is most aligned with Arrow's long-term direction?
3. Would you welcome a contribution adding owned option types?

---

**Suggested assignees:** @tustvold, @alamb (Arrow compute maintainers)  
**Labels:** `enhancement`, `compute`, `casting`
