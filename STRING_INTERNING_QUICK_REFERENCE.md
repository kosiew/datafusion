# String Interning Issue: Quick Reference Timeline

## Key Milestones

| Date | Commit | Component | Change | Impact |
|------|--------|-----------|--------|--------|
| Jun 27, 2025 | `3dd130e01` | `physical-expr-adapter/` | Introduced PhysicalExprAdapter crate | Foundation for expression serialization |
| Sep 23, 2025 | `4e803e821` | `CastColumnExpr` | New expression type for column casting | Schema adaptation capabilities added |
| Oct 27, 2025 | `c6ad17cf2` | Arrow upgrade | Arrow 57.0.0 with lifetime parameters | **INTRODUCES LIFETIME MISMATCH** |
| Jan 23, 2026 | `6d97babae` | Protobuf | Format options added to CastColumnExpr | Proto deserialize→Arrow mismatch exposed |
| Jan 24, 2026 | `1eb8796ed` | String interning | Interning cache implemented | **SOLVES MISMATCH** |
| Jan 24, 2026 | `a3cb753f9` | Cache bounds | Size limits added (8 test, 1024 prod) | Prevent unbounded memory leaks |

---

## The Root Cause: Arrow 57.0.0 Update

### Before (Arrow 56.x)
```rust
pub struct FormatOptions {
    pub null: String,                    // Owned
    pub date_format: Option<String>,     // Owned
}
```

### After (Arrow 57.0+)
```rust
pub struct FormatOptions<'a> {
    pub null: &'a str,                   // Borrowed!
    pub date_format: Option<&'a str>,    // Borrowed!
}
```

**Why Arrow changed this:** Performance optimization to reduce allocations

**Why this broke protobuf deserialization:** Protobuf messages produce owned strings that don't outlive the message lifetime—can't be cast to `&'static str`

---

## The Solution: String Interning with Bounded Cache

### Problem Solved
- Deserialize protobuf strings to `&'static str` (required by Arrow)
- Avoid unbounded memory leaks through deduplication
- Maintain simple API without breaking changes

### Implementation
```rust
// Global cache: deduplicate leaked strings
static FORMAT_STRING_CACHE: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

// When deserializing FormatOptions:
fn intern_format_str(value: &str) -> Result<&'static str> {
    let mut cache = format_string_cache().lock()?;
    
    // Return cached string if exists
    if let Some(existing) = cache.get(value).copied() {
        return Ok(existing);
    }
    
    // Leak only if new and within cache limit
    if cache.len() >= FORMAT_STRING_CACHE_LIMIT {
        return Err("Format string cache limit reached".into());
    }
    
    let leaked = Box::leak(value.to_owned().into_boxed_str());
    cache.insert(leaked);
    Ok(leaked)
}
```

### Cache Limits
- **Tests:** 8 strings (tight to catch problems early)
- **Production:** 1024 strings (covers realistic distinct patterns)

---

## Historical Context

This is **not** a design choice—it's a **necessary workaround**:

1. ✗ Can't use owned `String`: Arrow requires `&'a str`
2. ✗ Can't use regular borrows: Protobuf data doesn't live long enough
3. ✗ Can't change Arrow API: External dependency, not under our control
4. ✓ **Can use string interning:** Only practical solution that works

---

## When This Can Be Removed

Only if **either**:
1. Arrow removes the `'a` lifetime parameter from `FormatOptions` (upstream change)
2. Arrow adds a variant that works with owned `String` (unlikely)
3. DataFusion stops using Arrow for formatting (major refactor)

Until then, string interning is the status quo solution.
