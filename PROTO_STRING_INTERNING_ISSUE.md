# Protocol Buffer String Interning Issue

## Problem

When deserializing `FormatOptions` from protobuf into `ArrowFormatOptions` (used by the Arrow formatting library), there is a fundamental **lifetime mismatch**:

- **Protobuf deserialization** produces: Regular `String` or `&str` (tied to message lifetime)
- **ArrowFormatOptions requires**: `&'static str` (strings that live for the entire program)

### The Lifetime Constraint

```rust
// What Arrow expects
pub struct ArrowFormatOptions {
    null: &'static str,
    date_format: Option<&'static str>,
    datetime_format: Option<&'static str>,
    // ... etc
}

// What deserialization gives us
let format_options: protobuf::FormatOptions = /* ... */;
let date_fmt: &str = &format_options.date_format; // Borrows from proto message
// Cannot cast &str to &'static str - compiler rejects this!
```

## Current Solution: String Interning with Bounded Cache

The implementation in `datafusion/proto/src/physical_plan/from_proto.rs` uses **controlled string leaking** with a bounded cache:

### How It Works

1. **String Leaking** — Convert `String` to `&'static str` by permanently allocating:
   ```rust
   let leaked = Box::leak(value.to_owned().into_boxed_str()); // → &'static str
   ```

2. **Deduplication** — Store leaked strings in a cache to reuse them:
   ```rust
   let mut cache = format_string_cache().lock().unwrap();
   if let Some(existing) = cache.get(value) {
       return Ok(existing); // Reuse already-leaked string
   }
   let leaked = Box::leak(...);
   cache.insert(value, leaked);
   ```

3. **Bounded Growth** — Prevent unbounded memory leaks by limiting cache size:
   ```rust
   #[cfg(test)]
   const FORMAT_STRING_CACHE_LIMIT: usize = 8;      // Tight in tests
   #[cfg(not(test))]
   const FORMAT_STRING_CACHE_LIMIT: usize = 1024;   // Reasonable in production
   ```

### Why This Works

- **Deduplication**: If the same format string (e.g., `"%Y-%m-%d"`) is deserialized 100 times, it's leaked only once and reused
- **Early detection**: Test limit (8 strings) is intentionally tight to catch accidental unbounded growth during development
- **Production safety**: 1024 limit covers realistic use cases (distinct format patterns) while preventing pathological leaking

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Current (interning + cache)** | No API changes; works with Arrow API | Leaks memory intentionally; requires synchronization (mutex) |
| **On-demand conversion** | No memory leaks; simpler | Requires format conversion on every operation; may hurt performance |
| **Use `Arc<str>` or `Cow`** | No leaking; cleaner | Requires downstream API changes (Arrow library) |

## Alternative Approaches (Not Implemented)

### 1. On-Demand Conversion

Keep `FormatOptions` as proto struct, convert only at execution time:

```rust
pub struct CastOptions {
    safe: bool,
    format_options: protobuf::FormatOptions,  // Not ArrowFormatOptions
}

// At execution time
fn apply_cast(value: &str, opts: &CastOptions) -> Result<String> {
    let arrow_opts = convert_to_arrow(&opts.format_options)?; // Short-lived
    cast_value(value, &arrow_opts)
}
```

**Pros**: No memory leaks, no caching needed  
**Cons**: Repeated conversions if options used multiple times; may impact performance

### 2. Use Arc-based Format Options

Define a wrapper type that doesn't require `'static`:

```rust
#[derive(Clone)]
pub struct FormatOptions {
    null: Arc<str>,
    date_format: Option<Arc<str>>,
    // ...
}
```

**Pros**: No leaking, reference-counted semantics  
**Cons**: Requires changing Arrow library API or creating wrapper; performance overhead of Arc

### 3. Lazy Initialization

Store format options as proto, convert once, cache result per execution context:

```rust
pub struct CastColumnExpr {
    // ...
    format_options_proto: protobuf::FormatOptions,
    // Arrow options computed once on first use
    #[serde(skip)]
    cached_arrow_options: OnceLock<ArrowFormatOptions<'static>>,
}
```

**Pros**: Conversion happens once per expression per execution  
**Cons**: Still requires interning for `'static` lifetime; adds complexity with OnceLock

## Recommendations

## Is the Current Approach Correct for the Lifetime Mismatch?

**Yes, the current string interning approach described above is the correct and pragmatic solution for addressing the lifetime mismatch when deserializing `FormatOptions` from protobuf into `ArrowFormatOptions`.**

### Why This Approach Works

- **ArrowFormatOptions requires `&'static str`** for its fields, but protobuf deserialization only yields owned `String` or borrowed `&str` with a limited lifetime.
- **Rust does not allow casting a short-lived `&str` to `&'static str`** without leaking or otherwise extending the lifetime, so a conversion is necessary.
- **Interning with a bounded cache** (as implemented) allows converting these short-lived strings into `&'static str` by leaking them in a controlled, deduplicated, and size-limited way. This ensures:
    - Only a bounded number of strings are ever leaked (no unbounded memory growth)
    - The same format string is only leaked once and reused
    - The API contract with Arrow is satisfied without requiring upstream changes

### Alternatives and Trade-offs

As discussed, alternatives (on-demand conversion, using `Arc<str>`, or changing Arrow APIs) either introduce performance penalties, require non-trivial refactoring, or are not feasible without broader ecosystem changes. The current approach is a practical compromise that is safe for the intended use case (bounded, mostly static format strings in query plans).

## Historical Context: Why This Wasn't an Issue Before

### Timeline

1. **June 27, 2025** — `PhysicalExprAdapter` crate first created as foundation for expression serialization
2. **September 23, 2025** — `CastColumnExpr` introduced for column-level type casting operations
3. **October 27, 2025** — Arrow library upgraded to v57.0.0, which **changed `ArrowFormatOptions` fields from owned `String` to borrowed `&'static str`** as a performance optimization
4. **January 23, 2026** — `FormatOptions` serialized to protobuf and deserialization added to `PhysicalExprAdapter`
5. **January 24, 2026** — String interning cache implemented to resolve the lifetime mismatch

### Why It Became an Issue

The lifetime mismatch **was not an issue before** because:

- **Initially (June–September 2025)**: The `PhysicalExprAdapter` did not handle `FormatOptions` at all. It focused on other expression types.
- **After October 2025 (Arrow upgrade)**: Arrow changed its API to require `&'static str` instead of owned `String`, but DataFusion's `FormatOptions` was not yet being serialized to protobuf, so the incompatibility was not exposed.
- **January 23, 2026**: The first attempt to serialize `FormatOptions` to protobuf and deserialize it in `PhysicalExprAdapter` **immediately exposed the lifetime mismatch** because:
  - Protobuf deserialization produces owned `String` or borrowed `&str` (short-lived)
  - Arrow now requires `&'static str`
  - These two are fundamentally incompatible without string interning

### Why String Interning Was Unavoidable

At this point, there were no viable alternatives:

- **Cannot change Arrow** — External library; would require a major version change upstream
- **Cannot use owned `String`** — Arrow's API explicitly requires `&'static str`
- **Cannot borrow from protobuf** — The deserialized message has a limited lifetime that cannot be extended
- **Cannot avoid deserialization** — Format options must be extracted from protobuf to construct execution plans

**Therefore, string interning with a bounded cache is not a design choice but the only feasible solution** given these constraints.

### Conclusion

**For the current DataFusion and Arrow integration, the string interning approach with a bounded cache is the correct and recommended solution to the lifetime mismatch problem when deserializing `FormatOptions` from protobuf. The issue did not exist before because `FormatOptions` serialization was not yet implemented when Arrow's API changed.**

Key functions:

- `intern_format_str(value: &str) -> Result<&'static str>` — Lease and cache a single string
- `format_string_cache() -> &'static Mutex<FormatStringCache>` — Global cache singleton
- `format_options_from_proto(options: &protobuf::FormatOptions) -> Result<ArrowFormatOptions>` — Full conversion pipeline
- `cast_options_from_proto(...)` — Handles backward compatibility for legacy fields

### Testing

The test limit (`FORMAT_STRING_CACHE_LIMIT = 8` in test mode) ensures:
- Developers notice if serialization/deserialization pathways start leaking new format strings
- Tests fail fast with clear error message if cache limit exceeded
- Production limit (1024) is explicitly separate to avoid breaking real workloads

## See Also

- `datafusion/proto/tests/cases/roundtrip_physical_plan.rs` — Roundtrip tests for cast expressions
- `datafusion-common/src/cast_options.rs` — Definition of `CastOptions<'a>` and `FormatOptions`
- Arrow `arrow::compute::cast` — Uses `ArrowFormatOptions` with `&'static str` constraints
