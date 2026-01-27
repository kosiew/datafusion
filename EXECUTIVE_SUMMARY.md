# Executive Summary: String Interning Issue Investigation

## Overview

The string interning system in DataFusion's protobuf deserialization is not a design choice but a **necessary workaround** to resolve a lifetime mismatch between Arrow's API requirements and protobuf's data ownership model.

---

## The Problem in One Picture

```rust
// What Arrow 57.0+ requires:
pub struct FormatOptions<'a> {
    pub null: &'a str,  // Borrowed string
}

// What protobuf deserialization provides:
let proto: protobuf::FormatOptions = /* ... */;
let null: &str = &proto.null;  // This lifetime is bounded by proto

// The conflict:
// Arrow wants: &'static str (string lives forever)
// Protobuf gives: &'limited str (string dies when proto dies)
// Compiler says: "Impossible! Can't make this work!"
```

---

## Timeline at a Glance

| When | What | Why It Matters |
|------|------|-----------------|
| Jun 2025 | `PhysicalExprAdapter` created | Foundation for serialization |
| Sep 2025 | `CastColumnExpr` introduced | Enables column-level casting |
| **Oct 2025** | **Arrow upgraded to 57.0.0** | **Introduced lifetime requirement** |
| Jan 23, 2026 | Format options serialized to protobuf | **Exposed the lifetime mismatch** |
| Jan 24, 2026 | String interning cache implemented | **Solved the mismatch** |

---

## The Solution

### How It Works
```
1. Protobuf string (owned, limited lifetime)
   ↓
2. Intern function looks up in cache
   ├─ Cache hit? → Return cached &'static str
   └─ Cache miss? → Leak string, cache it, return &'static str
   ↓
3. Use in ArrowFormatOptions<'static> ✓
```

### Key Features
- **Deduplication**: Same string cached; only leaked once
- **Bounded cache**: Prevents unbounded memory growth
  - Tests: 8 entries (tight, catches problems)
  - Production: 1024 entries (covers realistic patterns)
- **Thread-safe**: Mutex-protected global cache
- **Transparent**: Deserialization automatically interns strings

---

## Why Other Solutions Don't Work

| Approach | Why It Doesn't Work |
|----------|-------------------|
| **Use owned `String`** | Arrow explicitly requires `&'a str`, not `String` |
| **Regular borrowing** | Borrowed data dies with proto message |
| **Change Arrow API** | External dependency, not under our control |
| **Avoid serializing FormatOptions** | Breaks Flight SQL/Substrait compatibility |

**String interning** is the *only* practical solution.

---

## Impact Assessment

### Memory Behavior
- **Best case**: 1000 deserializations, same format string → 1 leak (~10 bytes)
- **Typical case**: 10,000 deserializations, 50 distinct formats → 50 leaks (~500 bytes)
- **Worst case**: Fails with error when cache limit exceeded (prevented)

### Performance
- Cache hit: O(1) HashMap lookup
- Cache miss: O(1) HashMap insert + string allocation
- Mutex contention: Low in typical scenarios, could be optimized if needed

### Thread Safety
- ✓ Multiple threads can deserialize concurrently
- ✓ No data races (mutex protects HashMap)
- ⚠ Potential mutex contention bottleneck (if thousands of concurrent deserializations)

---

## When This Can Be Fixed

### Option 1: Arrow Changes Its API
If Arrow removes the `'a` lifetime parameter (unlikely—it's a useful optimization):
```rust
// Future Arrow version (hypothetical)
pub struct FormatOptions {  // No lifetime parameter
    pub null: Arc<str>,     // Ownership instead of borrowing
}
```

### Option 2: DataFusion Stops Using Arrow for Formatting
Major refactor, but would eliminate the issue.

### Option 3: Accept Status Quo
String interning works, is well-tested, and has acceptable performance for typical workloads.

**Current recommendation: Status quo** (working solution, low change cost/risk)

---

## Code Location & Configuration

**Implementation**: [datafusion/proto/src/physical_plan/from_proto.rs](datafusion/proto/src/physical_plan/from_proto.rs#L820-L980)

**Cache size configuration**:
```rust
#[cfg(test)]
const FORMAT_STRING_CACHE_LIMIT: usize = 8;      // Tight in tests

#[cfg(not(test))]
const FORMAT_STRING_CACHE_LIMIT: usize = 1024;   // Reasonable in production
```

**Error handling**: When cache limit exceeded, deserialization fails with clear error message.

---

## For Developers

### Things to Know
1. Format strings are interned globally per process
2. Same string deserialized multiple times reuses the same `&'static str` (pointer equality)
3. Tight test limit (8 strings) is intentional—tests should not accidentally generate unbounded distinct format strings
4. Cache is transparent—just works during deserialization

### If You Need to Debug
- Strings in cache are meant to be permanently leaked (not a memory leak, by design)
- If cache limit exceeded: check if your code is generating too many distinct format string patterns
- For production issues: can adjust `FORMAT_STRING_CACHE_LIMIT` if needed (with strong justification)

### If You Want to Improve
Consider:
- Use `parking_lot::Mutex` for faster synchronization
- Use `DashMap` for lock-free concurrent access
- Add metrics for cache hit/miss rates
- Pre-populate with common format patterns
- Implement LRU eviction if needed

---

## References

- **INVESTIGATION_ANSWERS.md** - Detailed answers to all historical questions
- **TIMELINE_PROTO_STRING_INTERNING.md** - Complete timeline with commit details
- **STRING_INTERNING_QUICK_REFERENCE.md** - Quick lookup table
- **STRING_INTERNING_IMPLEMENTATION.md** - Technical implementation details
- **Original Issue Document** - [PROTO_STRING_INTERNING_ISSUE.md](PROTO_STRING_INTERNING_ISSUE.md)

---

## Summary Statement

The string interning system in DataFusion is a **proven, well-designed solution** to an unavoidable problem created by the intersection of:
1. Arrow's optimization to use borrowed strings in `FormatOptions<'a>`
2. The need to serialize format options to protobuf for Flight SQL
3. Protobuf producing owned strings with bounded lifetimes

The solution deduplicates format strings to minimize memory leaks, uses bounded caching to prevent pathological growth, and works transparently with existing code. It represents a **pragmatic balance between correctness, performance, and maintainability** given the constraints.
