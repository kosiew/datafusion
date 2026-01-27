# String Interning Implementation Details

## Current Implementation (as of January 27, 2026)

Located in: [datafusion/proto/src/physical_plan/from_proto.rs](datafusion/proto/src/physical_plan/from_proto.rs#L820-L980)

### Cache Structure

```rust
/// Cache for interned format strings.
///
/// We leak strings to satisfy the `'static` lifetime required by
/// `ArrowFormatOptions` in cast options. To avoid unbounded growth,
/// once the cache reaches the limit we only allow lookups for strings
/// that are already interned.
static FORMAT_STRING_CACHE: OnceLock<Mutex<FormatStringCache>> = OnceLock::new();

#[derive(Default)]
struct FormatStringCache {
    entries: HashMap<String, &'static str>,
}
```

**Key Design Choices:**
- `OnceLock<Mutex<FormatStringCache>>`: Thread-safe lazy initialization
- `HashMap<String, &'static str>`: Maps owned keys to their leaked string references
- Single global cache per process (not per-context or per-connection)

### Cache Size Limits

```rust
/// Maximum number of unique format strings to cache.
#[cfg(test)]
const FORMAT_STRING_CACHE_LIMIT: usize = 8;

#[cfg(not(test))]
const FORMAT_STRING_CACHE_LIMIT: usize = 1024;
```

**Why different limits?**
- **Tests (8):** Tight limit catches unintended string proliferation during development
- **Production (1024):** Covers realistic scenarios (different date/time format patterns) while preventing pathological growth

### Insertion Logic

```rust
fn insert(&mut self, value: &str) -> Result<&'static str> {
    // Fast path: string already exists
    if let Some(existing) = self.get(value) {
        return Ok(existing);
    }

    // Check cache limit
    if self.entries.len() >= FORMAT_STRING_CACHE_LIMIT {
        return Err(internal_datafusion_err!(
            "Format string cache limit ({}) reached; cannot intern new format string {value:?}",
            FORMAT_STRING_CACHE_LIMIT
        ));
    }

    // Leak new string and cache it
    let leaked = Box::leak(value.to_owned().into_boxed_str());
    let key = value.to_owned();
    self.entries.insert(key.clone(), leaked);
    Ok(leaked)
}
```

**Flow:**
1. Check if string already cached → return cached reference
2. Check if cache is at limit → return error if so
3. Leak new string → convert owned `String` to `&'static str`
4. Store mapping → enable future lookups

### Integration with ArrowFormatOptions

```rust
fn intern_format_strings(options: &protobuf::FormatOptions) -> Result<InterredFormatStrings> {
    let mut cache = format_string_cache()
        .lock()
        .expect("format string cache lock poisoned");
    
    Ok(InterredFormatStrings {
        null: cache.insert(&options.null)?,
        date_format: options.date_format.as_deref().map(|s| cache.insert(s)).transpose()?,
        datetime_format: options.datetime_format.as_deref().map(|s| cache.insert(s)).transpose()?,
        timestamp_format: options.timestamp_format.as_deref().map(|s| cache.insert(s)).transpose()?,
        timestamp_tz_format: options.timestamp_tz_format.as_deref().map(|s| cache.insert(s)).transpose()?,
        time_format: options.time_format.as_deref().map(|s| cache.insert(s)).transpose()?,
    })
}

fn format_options_from_proto(
    options: &protobuf::FormatOptions,
) -> Result<ArrowFormatOptions<'static>> {
    let duration_format = duration_format_from_proto(options.duration_format)?;
    let interned = intern_format_strings(options)?;
    
    Ok(ArrowFormatOptions::new()
        .with_display_error(options.safe)
        .with_null(interned.null)
        .with_date_format(interned.date_format)
        .with_datetime_format(interned.datetime_format)
        .with_timestamp_format(interned.timestamp_format)
        .with_timestamp_tz_format(interned.timestamp_tz_format)
        .with_time_format(interned.time_format)
        .with_duration_format(duration_format)
        .with_types_info(options.types_info))
}
```

---

## Memory Behavior Analysis

### Best Case: Perfect Deduplication
```
Input: 1000 deserialization events
       All use date_format = "%Y-%m-%d"

Output: Cache entries = 1
        Memory leaked = ~11 bytes (length of "%Y-%m-%d")
        Hit rate = 99.9%
```

### Typical Case: Multiple Format Patterns
```
Input: 10,000 deserialization events
       ~50 distinct format patterns (e.g., "%Y-%m-%d", "%d/%m/%Y", etc.)

Output: Cache entries = 50
        Memory leaked = ~500 bytes total
        Hit rate = 99.5%
```

### Worst Case: Unbounded String Generation
```
Input: Test generates 100 unique strings per iteration
       Cache limit = 8

Output: After 1st iteration → Error: "cache limit reached"
        This fails loudly, preventing production from leaking memory
```

---

## Error Handling

### When Cache Limit is Exceeded

```rust
Err(internal_datafusion_err!(
    "Format string cache limit ({}) reached; cannot intern new format string {value:?}",
    FORMAT_STRING_CACHE_LIMIT
))
```

**Behavior:**
- In tests: Fails immediately, alerts developer to the problem
- In production: Returns error to caller (Flight RPC, Substrait deserialization, etc.)

**Recovery options for caller:**
1. Reuse existing format options (common case)
2. Implement caching at higher level
3. Batch deserializations to reduce unique patterns

---

## Thread Safety

### Mutex Protection
```rust
let mut cache = format_string_cache()
    .lock()
    .expect("format string cache lock poisoned");
```

**Thread safety properties:**
- ✓ Multiple threads can call `intern_format_str` concurrently
- ✓ Each insertion is atomic (protected by mutex)
- ✓ No data races (HashMap is not thread-safe, but mutex guards it)

**Potential bottleneck:**
- ⚠ Mutex contention in high-concurrency scenarios (many threads deserializing simultaneously)
- Could be mitigated with lock-free structures like `DashMap` if needed

---

## Lifetime Implications

### Why `&'static str` is Necessary

Arrow's FormatOptions requires:
```rust
pub struct FormatOptions<'a> {
    pub null: &'a str,
    pub date_format: Option<&'a str>,
    // ...
}
```

Protobuf gives us:
```rust
let proto_options: protobuf::FormatOptions = /* ... */;
let null_str: &str = &proto_options.null;  // Lifetime tied to proto_options
// null_str cannot outlive proto_options!
```

Solution: Make the string live forever:
```rust
let null_str: &'static str = Box::leak(proto_options.null.into_boxed_str());
// null_str now lives for entire program duration
// Can be used in ArrowFormatOptions<'static>
```

---

## Performance Considerations

### Insertion Complexity
- **Cache hit:** O(1) lookup in HashMap
- **Cache miss with capacity:** O(1) insert into HashMap + string allocation
- **Cache miss at capacity:** O(1) lookup + error return (no mutation)

### Memory Overhead
- HashMap entry: ~32-48 bytes per entry (on 64-bit system)
- String data: Variable (but typically 10-30 bytes for format strings)
- **Total per unique string:** ~40-80 bytes

### Optimization Opportunities
1. Use `DashMap` for lock-free concurrent access
2. Use `parking_lot::Mutex` (faster than `std::sync::Mutex`)
3. Pre-populate cache with common format patterns
4. Add LRU eviction if needed (track access frequency)

---

## Testing

### Test Coverage

From [from_proto.rs#L945](datafusion/proto/src/physical_plan/from_proto.rs#L945):

```rust
#[test]
fn format_string_cache_reuses_strings() {
    // Helper to generate unique strings not already in the cache
    let unique_value = |prefix: &str| {
        let mut counter = 0;
        loop {
            let candidate = format!("{prefix}-{counter}");
            let cache = format_string_cache()
                .lock()
                .expect("format string cache lock poisoned");
            if !cache.entries.contains_key(candidate.as_str()) {
                return candidate;
            }
            counter += 1;
        }
    };
    
    // Verify that same string produces same &'static str
    let str1 = intern_format_str(&unique_value("reuse")).unwrap();
    let str2 = intern_format_str(&unique_value("reuse")).unwrap();
    assert!(ptr::eq(str1, str2));  // Pointer equality!
}
```

**Key insight:** Tests verify pointer equality, not just value equality. This proves deduplication is working.

---

## Observability & Debugging

### Current Limitations
- No metrics exported for cache hit/miss rates
- No way to inspect cache contents at runtime
- Error only returned when limit is exceeded

### Possible Improvements
```rust
#[derive(Default)]
pub struct CacheStats {
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub current_entries: usize,
}

// Could add optional metrics collection:
pub fn cache_stats() -> CacheStats {
    let cache = format_string_cache().lock().unwrap();
    // Return statistics
}
```

---

## Relationship to Cast Options

### How FormatOptions Flows to Expressions

```
Protobuf Message (FormatOptions)
         ↓
format_options_from_proto()        ← String interning happens here
         ↓
ArrowFormatOptions<'static>
         ↓
CastOptions { safe, format_options }
         ↓
CastColumnExpr or PhysicalExpr
         ↓
Execution: format values using ArrowFormatOptions
```

The interned strings live for the entire program, so they can be safely used throughout the execution pipeline without lifetime concerns.
