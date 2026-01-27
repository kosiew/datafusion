# Timeline: Evolution of String Interning Issue in DataFusion

This document traces the historical evolution of how the string interning issue was created through a series of feature additions and library upgrades.

## Phase 1: Physical Expression Adapter Introduction (June 2025)

**Date:** June 27, 2025  
**Commit:** `3dd130e01` - "feat: implement predicate adaptation for nested structs"

### What Happened
The `datafusion/physical-expr-adapter/` crate was introduced to handle schema adaptation and expression rewriting for nested structs. This established the foundation for later changes:

- Created 1000+ lines of new code for schema rewriting logic
- Introduced `PhysicalExprAdapter` for adapting expressions between different schemas
- Added ability to rewrite column references and cast expressions

### Why It Matters
This crate became the central place where physical expressions are serialized to/from protobuf in Flight SQL and Substrait scenarios. All subsequent format handling changes would flow through this layer.

---

## Phase 2: CastColumnExpr Introduction (September 2025)

**Date:** September 23, 2025  
**Commit:** `4e803e821` - "feat: introduce CastColumnExpr for enhanced casting functionality"

### What Happened
A new expression type `CastColumnExpr` was introduced specifically for casting columns during schema adaptation:

- Added dedicated expression type in `datafusion/physical-expr/src/expressions/cast.rs`
- Created specialized handling in `PhysicalExprAdapter` for column-level casting
- Supports casting individual columns when adapting between schemas

### Why It Matters
This set the stage for needing format options at the expression level. `CastColumnExpr` would eventually need access to formatting options for safe casting (date formats, null handling, etc.).

---

## Phase 3: Arrow 57.0.0 Upgrade (October 2025)

**Date:** October 27, 2025  
**Commit:** `c6ad17cf2` - "Upgrade DataFusion to arrow/parquet 57.0.0"

### What Happened
DataFusion upgraded to Arrow 57.0.0, which introduced **lifetime parameters to `FormatOptions`**:

**Arrow 56.x (before):**
```rust
pub struct FormatOptions {
    null: String,  // Owned strings
    date_format: Option<String>,
    // ...
}
```

**Arrow 57.0.0 (after):**
```rust
pub struct FormatOptions<'a> {
    null: &'a str,           // Borrowed strings
    date_format: Option<&'a str>,
    // ... all string fields changed to &'a str
}
```

### Why Arrow Made This Change
This was likely an optimization in Arrow to reduce allocations. By allowing borrowed strings, callers could pass references to existing data without cloning.

### The Problem This Created
This change introduced the fundamental lifetime mismatch that would plague proto deserialization. Protobuf deserialization produces owned strings that don't live for `'static`, but Arrow now requires strings that outlive the deserialized message.

---

## Phase 4: Cast Column Format Options in Protobuf (January 23, 2026)

**Date:** January 23, 2026  
**Commit:** `6d97babae` - "Add cast column format option fields to protobuf"

### What Happened
Format options were added to the protobuf definition for `CastColumnExpr`:

**Changes to `datafusion/proto/proto/datafusion.proto`:**
```protobuf
message PhysicalCastColumnNode {
  PhysicalExprNode expr = 1;
  datafusion_common.Field input_field = 2;
  datafusion_common.Field target_field = 3;
  bool safe = 4;
  FormatOptions format_options = 5;  // ← NEW
}

message FormatOptions {
  bool safe = 1;
  string null = 2;
  optional string date_format = 3;
  optional string datetime_format = 4;
  optional string timestamp_format = 5;
  optional string timestamp_tz_format = 6;
  optional string time_format = 7;
  optional string duration_format = 8;
  bool types_info = 9;
}
```

### Why This Was Needed
When serializing `CastColumnExpr` expressions to protobuf (for Flight SQL/Substrait), the format options needed to be serialized too. This allows remote systems to understand how to format values during casting.

### The Dilemma
Now deserialization had to convert protobuf-generated strings (owned by the message) into `ArrowFormatOptions<'a>` which expected borrowed strings tied to some lifetime.

---

## Phase 5: String Interning Cache Implementation (January 24, 2026)

**Date:** January 24, 2026  
**Commit:** `1eb8796ed` - "Optimize format string handling with caching"

### What Happened
The string interning solution was implemented in `datafusion/proto/src/physical_plan/from_proto.rs`:

**Old approach (before):**
```rust
fn leak_str(value: &str) -> &'static str {
    Box::leak(value.to_string().into_boxed_str())  // Leak without reuse
}

fn format_options_from_proto(
    options: &protobuf::FormatOptions,
) -> Result<ArrowFormatOptions<'static>> {
    let null = leak_string(options.null.clone());  // Leak each string
    let date_format = options.date_format.as_deref().map(leak_str);  // Leak each
    // ...
    Ok(ArrowFormatOptions::new()
        .with_null(null)
        .with_date_format(date_format)
        // ...
    )
}
```

**New approach (with caching):**
```rust
static FORMAT_STRING_CACHE: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

fn format_string_cache() -> &'static Mutex<HashSet<&'static str>> {
    FORMAT_STRING_CACHE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn intern_format_str(value: &str) -> &'static str {
    let mut cache = format_string_cache()
        .lock()
        .expect("format string cache lock poisoned");
    
    // Check if string already exists
    if let Some(existing) = cache.get(value).copied() {
        return existing;  // Reuse
    }
    
    // Only leak if new
    let leaked = Box::leak(value.to_owned().into_boxed_str());
    cache.insert(leaked);
    leaked
}

fn format_options_from_proto(
    options: &protobuf::FormatOptions,
) -> Result<ArrowFormatOptions<'static>> {
    let null = intern_format_str(&options.null);  // Deduplicated
    let date_format = options.date_format.as_deref().map(intern_format_str);  // Deduplicated
    // ...
}
```

### Key Innovation
**Deduplication through caching**: Instead of leaking every format string deserialized, the same string is only leaked once and then reused.

Example: If 100 queries deserialize `date_format: "%Y-%m-%d"`, only one `&'static str` is created and reused 100 times.

### Trade-offs Made
1. **Memory vs. API compatibility**: Accepted minor memory leaking to avoid changing Arrow's public API
2. **Simplicity vs. Efficiency**: Used `HashSet` cache with mutex for simplicity; could be more sophisticated
3. **Safety vs. Performance**: Added bounded cache limit to prevent pathological growth while keeping mutex contention low

---

## Phase 6: Cache Optimization & Size Limits (January 24, 2026)

**Date:** January 24, 2026  
**Commit:** `a3cb753f9` - "Refactor format string caching and testing"

### What Happened
The cache implementation was refined with size limits:

**Added bounded eviction:**
```rust
#[cfg(test)]
const FORMAT_STRING_CACHE_LIMIT: usize = 8;      // Tight test limit

#[cfg(not(test))]
const FORMAT_STRING_CACHE_LIMIT: usize = 1024;   // Production limit

fn intern_format_str(value: &str) -> Result<&'static str> {
    let mut cache = format_string_cache()
        .lock()
        .expect("format string cache lock poisoned");
    
    if cache.len() >= FORMAT_STRING_CACHE_LIMIT {
        return Err(DataFusionError::Internal(
            format!("Format string cache limit ({}) reached", FORMAT_STRING_CACHE_LIMIT)
        ));
    }
    
    // ... rest of logic
}
```

### Rationale

| Environment | Limit | Purpose |
|-------------|-------|---------|
| **Tests** | 8 | Catch accidental unbounded growth during development |
| **Production** | 1024 | Cover realistic distinct format patterns without leaking excessively |

The test limit is intentionally tight to fail-fast if a test accidentally creates too many distinct format strings (a sign of a problem).

---

## Phase 7: Further Refinements (January 24, 2026)

**Commits:**
- `2c4beeb94` - "Fix format-string cache leak and add size cap"
- `6be67cbdf` - "Improve format string cache error handling and tests"
- `84175a884` - "Enhance format string cache tests: add unique string generation and cache limit checks"

### Changes Made
1. Fixed potential cache leak scenarios
2. Improved error messages when cache limit is exceeded
3. Added comprehensive tests for cache behavior and uniqueness

---

## Summary: The Causal Chain

```
June 2025: PhysicalExprAdapter introduced
    ↓
Sept 2025: CastColumnExpr added (expression type for schema adaptation)
    ↓
Oct 2025: Arrow upgraded to 57.0.0 (adds lifetimes to FormatOptions)
    ↓
Jan 23, 2026: Format options serialized in protobuf for CastColumnExpr
    ↓ [COLLISION POINT]
           Protobuf deserialization produces owned strings
           Arrow expects &'static str (requires lifetime outliving deserialize)
           
Jan 24, 2026: String interning with bounded cache implemented to solve mismatch
```

---

## Why String Interning Was Necessary

Before January 24, 2026, there was **no working solution** to deserialize `FormatOptions` because:

1. **On-demand conversion** not possible: Arrow's type signature is fixed, can't change it
2. **Owned strings** don't work: Arrow specifically requires `&'a str`, not `String`
3. **Regular borrowing** doesn't work: Deserialized data doesn't live long enough
4. **Smart pointers** don't work: Arrow signature is hardcoded for `&str`

**String interning** is the only practical solution that:
- Works with Arrow's existing API (no external dependencies can be changed)
- Avoids unbounded memory growth (via caching and size limits)
- Provides reasonable performance (deduplication)
- Integrates seamlessly with protobuf deserialization

---

## Remaining Questions for Implementation

### Open Issues:
1. **When can this be removed?** Only if Arrow changes its `FormatOptions<'a>` signature
2. **Can the cache be made more efficient?** Currently uses `Mutex<HashSet>`, could use `parking_lot::Mutex` or a lock-free structure
3. **Is 1024 the right limit?** Telemetry data from production would help inform this
4. **Can errors be more user-friendly?** Currently returns a generic "cache limit reached" error

### Future Opportunities:
- Track cache hit/miss rates for observability
- Add configuration option for cache size limit
- Consider alternative data structures (e.g., `DashMap` for lock-free caching)
- Investigate if Arrow library would accept lifetime-parameterized alternatives
