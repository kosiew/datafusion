# Answers to Historical Investigation Questions

## Question 1: When was `PhysicalExprAdapter` first introduced?

**Answer:** June 27, 2025

**Commit:** `3dd130e01` - "feat: implement predicate adaptation for nested structs"

**Details:**
- Created entirely new crate: `datafusion/physical-expr-adapter/`
- ~1000 lines of code added (Cargo.toml, README, src/lib.rs, schema_rewriter.rs)
- Purpose: Implement schema adaptation logic for physical expressions, enabling expressions to work correctly when encountering different schemas (e.g., in nested struct scenarios)
- This crate became the central hub for physical expression serialization/deserialization in Flight SQL and Substrait workflows

**What it enabled:** The ability to rewrite and adapt physical expressions when schemas differ between systems, laying groundwork for the later CastColumnExpr feature.

---

## Question 2: When was `FormatOptions`/`CastOptions` handling added to `PhysicalExprAdapter`?

**Answer:** Two separate phases:

### Phase A: CastColumnExpr Introduction
**Date:** September 23, 2025  
**Commit:** `4e803e821` - "feat: introduce CastColumnExpr for enhanced casting functionality"

- Added new expression type `CastColumnExpr` in `datafusion/physical-expr/`
- Integrated into schema rewriter for column-level casting
- This was the first time format options were conceptually tied to physical expressions
- However, format options themselves were not yet serialized to protobuf

### Phase B: FormatOptions Serialization
**Date:** January 23, 2026  
**Commit:** `6d97babae` - "Add cast column format option fields to protobuf"

- Added `FormatOptions` message to protobuf: `datafusion/proto/proto/datafusion.proto`
- Added `format_options` field to `PhysicalCastColumnNode` protobuf message
- Regenerated protobuf bindings with new serialization support
- Updated deserialization logic in `from_proto.rs` to handle format options

**Timeline gap explanation:** Between Sept 2025 and Jan 2026, `CastColumnExpr` existed but wasn't serializable to protobuf. In January, serialization support was added, which exposed the lifetime mismatch problem.

---

## Question 3: What changed in Arrow or DataFusion that introduced the lifetime mismatch constraint?

**Answer:** Arrow library upgrade from 56.x to 57.0.0

**Date:** October 27, 2025  
**Commit:** `c6ad17cf2` - "Upgrade DataFusion to arrow/parquet 57.0.0"

### The Specific Change in Arrow

**Arrow 56.x signature:**
```rust
pub struct FormatOptions {
    pub null: String,
    pub date_format: Option<String>,
    pub datetime_format: Option<String>,
    pub timestamp_format: Option<String>,
    pub timestamp_tz_format: Option<String>,
    pub time_format: Option<String>,
}
```

**Arrow 57.0.0 signature:**
```rust
pub struct FormatOptions<'a> {
    pub null: &'a str,
    pub date_format: Option<&'a str>,
    pub datetime_format: Option<&'a str>,
    pub timestamp_format: Option<&'a str>,
    pub timestamp_tz_format: Option<&'a str>,
    pub time_format: Option<&'a str>,
}
```

### Why Arrow Made This Change
This was a **performance optimization** to reduce allocations:
- With owned `String`, every construction allocated and cloned strings
- With borrowed `&'a str`, callers could pass references to existing data
- Arrow consumer code could reuse existing strings in many scenarios

### Why This Created the Problem
When combined with protobuf deserialization:
- Protobuf messages own their string data
- That data has a limited lifetime (tied to the message)
- Arrow's `FormatOptions<'a>` with a specific `'a` lifetime can borrow strings for `'a`
- But we need `&'static str` for use in persisted cast options
- No way to bridge this gap without string interning/leaking

**The collision happened in January 2026** when format options were serialized to protobuf:
- Before Jan 23: No protobuf serialization of format options
- Jan 23: Format options added to protobuf
- Jan 23-24: Deserialization broke because protobuf strings are owned, Arrow expects borrowed
- Jan 24: String interning solution implemented

---

## Question 4: Was there a time when `FormatOptions` deserialization from protobuf worked without string interning, and if so, what changed?

**Answer:** NO - String interning was implemented the same day format options were added to protobuf

**Timeline:**
- **Jan 23, 2026:** Format options added to protobuf definition
- **Jan 23, 2026:** Deserialization logic added (but broken—no string interning solution yet)
- **Jan 24, 2026:** String interning cache implemented to fix the deserialization

**Evidence:**
- Commit `6d97babae` (Jan 23) adds format options to protobuf but doesn't include interning
- Commit `1eb8796ed` (Jan 24, 15:02) implements the actual interning solution
- These are separate commits, indicating the problem was discovered immediately when format options were added, and the solution was implemented the next day

**What was in the original implementation (Jan 23):**
From git history, the original approach in `6d97babae` was naive string leaking without a cache:
```rust
fn leak_str(value: &str) -> &'static str {
    Box::leak(value.to_string().into_boxed_str())  // Leak every string
}

// Problem: Every deserialization event leaks all strings
fn format_options_from_proto(
    options: &protobuf::FormatOptions,
) -> Result<ArrowFormatOptions<'static>> {
    let null = leak_string(options.null.clone());  // Leak
    let date_format = options.date_format.as_deref().map(leak_str);  // Leak
    // ... unbounded leaking!
}
```

**What changed (Jan 24):**
```rust
fn intern_format_str(value: &str) -> Result<&'static str> {
    let mut cache = format_string_cache().lock()?;
    
    if let Some(existing) = cache.get(value).copied() {
        return Ok(existing);  // Reuse!
    }
    
    // Only leak if new and within limit
    if cache.len() >= FORMAT_STRING_CACHE_LIMIT {
        return Err("cache limit reached".into());
    }
    
    let leaked = Box::leak(value.to_owned().into_boxed_str());
    cache.insert(leaked);
    Ok(leaked)  // Reused on future calls
}
```

**The key insight:** Someone (likely during code review or testing) realized that naive leaking would cause unbounded memory growth, and the cache solution was designed and implemented within a day.

---

## Summary: The Causal Chain

```
┌─────────────────────────────────────────────────────────────────┐
│ June 27, 2025: PhysicalExprAdapter created                      │
│ → Foundation for expression serialization                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Sept 23, 2025: CastColumnExpr introduced                        │
│ → Enables column-level casting in schema adaptation             │
│ → Not yet serializable to protobuf                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ [3-month gap]
┌────────────────────────▼────────────────────────────────────────┐
│ Oct 27, 2025: Arrow upgraded to 57.0.0                          │
│ → FormatOptions<'a> now requires borrowed strings               │
│ → But we still use owned String in DataFusion                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ [2+ months gap]
┌────────────────────────▼────────────────────────────────────────┐
│ Jan 23, 2026: FormatOptions added to protobuf (12:00 AM)        │
│ → Exposes lifetime mismatch for the first time                  │
│ ⚠ → Naive implementation leaks unbounded memory                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ [24 hours]
┌────────────────────────▼────────────────────────────────────────┐
│ Jan 24, 2026: String interning cache implemented (3:02 PM)      │
│ ✓ → Solves lifetime mismatch with bounded memory leaks          │
│ ✓ → Caches format strings to enable deduplication               │
│ ✓ → Prevents unbounded growth with limit (8 test, 1024 prod)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

1. **This was not a design choice**: String interning with memory leaking is a *necessary workaround*, not a preference. It's the only practical solution given:
   - Arrow's requirement for `&'static str` in `FormatOptions`
   - Protobuf producing owned strings
   - Cannot modify Arrow library (external dependency)

2. **The problem was inevitable**: Once format options needed to be serialized to protobuf AND Arrow upgraded to require borrowed strings, the lifetime mismatch became unavoidable.

3. **The solution was well-thought-out**: The cache wasn't an afterthought—someone recognized unbounded leaking as dangerous and implemented bounded caching within a day.

4. **This can only be removed if**: Arrow changes its API or DataFusion stops using Arrow for formatting. Until then, string interning is the status quo solution.

5. **Tight test limit is intentional**: The 8-string limit in tests is designed to fail-fast if code accidentally generates unbounded format strings, preventing production issues.
