# Arrow Issue: Support runtime-owned format and cast options

## Summary

Arrow's `FormatOptions<'a>` and `CastOptions<'a>` use generic lifetimes, but the API design and default constants are built around `'static` strings. This makes it impractical for downstream projects to work with dynamically-created format options (e.g., from user config or SQL). DataFusion and other consumers must create owned wrapper types to work around this limitation.

## Problem

### Why Generic Lifetimes Aren't Enough

While Arrow's `FormatOptions<'a>` and `CastOptions<'a>` *can* accept generic lifetimes, the practical API design makes them hard to use with runtime data:

```rust
// arrow/compute/cast.rs
pub const DEFAULT_FORMAT_OPTIONS: FormatOptions<'static> = FormatOptions {
    null: "NULL",
    date_format: None,
    // ... all fields are 'static
};

pub const DEFAULT_CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: DEFAULT_FORMAT_OPTIONS,
};

// Most public APIs accept CastOptions<'static>
pub fn cast(
    array: &dyn Array,
    to_type: &DataType,
    options: Option<&CastOptions<'static>>,  // <-- 'static constraint
) -> Result<Arc<dyn Array>>
```

### Real-world Examples from DataFusion

Arrow's actual `cast_with_options` function signature is:

```rust
pub fn cast_with_options(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastOptions,  // Accepts CastOptions by reference
) -> Result<ArrayRef, ArrowError>
```

In practice, DataFusion code works with this API like:

```rust
// datafusion/functions/src/datetime/to_date.rs
use arrow::compute::cast_with_options;
use datafusion::common::format::DEFAULT_CAST_OPTIONS;

cast_with_options(
    &cast_with_options(&array, &Int32, &DEFAULT_CAST_OPTIONS)?,
    &Date32,
    &DEFAULT_CAST_OPTIONS,  // ← Always using the pre-defined constant
)
```

And in nested struct casting:

```rust
// datafusion/common/src/nested_struct.rs
pub fn cast_column(
    source_col: &ArrayRef,
    target_field: &Field,
    cast_options: &CastOptions,  // Accepts generic CastOptions
) -> Result<ArrayRef> {
    match target_field.data_type() {
        Struct(target_fields) => {
            cast_struct_column(source_col, target_fields, cast_options)
        }
        _ => Ok(cast_with_options(
            source_col,
            target_field.data_type(),
            cast_options,  // ← Passes through to Arrow
        )?),
    }
}
```

### Why This Limitation Causes Problems

When users (or DataFusion) want to create custom format options at runtime—e.g., a user-specified date format from SQL or config—they face this constraint:

```rust
// This fails to compile:
let user_format = "yyyy-MM-dd";  // From SQL/config, not a compile-time literal
let cast_opts = CastOptions {
    safe: false,
    format_options: FormatOptions {
        null: "NULL",
        date_format: Some(&user_format),  // ❌ Error: user_format lives 
                                           //    only in this scope, not 'static
        // ...
    },
};
```

The generic lifetime in the struct definition *allows* any `'a`, but the constants and common usage patterns enforce `'static`.

### Root Cause

The issue stems from:
1. **Default constants require `'static`** (hardcoded string literals in `const` declarations)
2. **Public APIs often specify `CastOptions<'static>`** to ensure compatibility with the constants
3. **No way to create dynamic options** that satisfy the `'static` lifetime bound

### Current API Design (simplified)

```rust
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
