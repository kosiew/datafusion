# New `Signature::with_parameters` Method

## Overview

The new `Signature::with_parameters` method provides an ergonomic way to add parameter names to **ALL** TypeSignature variants, not just `Exact` and `Coercible` signatures.

## Comparison with Existing Methods

### Before: Using `with_parameter_names`
```rust
// Only accepts String or types that implement Into<String>
let sig = Signature::exact(vec![DataType::Int32, DataType::Utf8], Volatility::Immutable)
    .with_parameter_names(vec!["count".to_string(), "name".to_string()])?;
```

### Before: Using `from_parameter_variants`
```rust
// Limited to Exact, Coercible, Nullary, and OneOf signatures only
// Cannot handle Variadic, Uniform, Numeric, String, etc.
let sig = Signature::from_parameter_variants(
    &[vec![("count", DataType::Int32), ("name", DataType::Utf8)]],
    Volatility::Immutable
)?;
```

### After: Using `with_parameters`
```rust
// Works with &str directly, more ergonomic
let sig = Signature::exact(vec![DataType::Int32, DataType::Utf8], Volatility::Immutable)
    .with_parameters(vec!["count", "name"])?;
```

## Supported TypeSignature Variants

The `with_parameters` method supports **ALL** TypeSignature variants:

### 1. Exact Signature
```rust
let sig = Signature::exact(vec![DataType::Int32, DataType::Utf8], Volatility::Immutable)
    .with_parameters(vec!["count", "name"])?;
```

### 2. Uniform Signature
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::uniform(3, vec![DataType::Float64], Volatility::Immutable)
    .with_parameters(vec!["a", "b", "c"])?;
```

### 3. Numeric Signature
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::numeric(2, Volatility::Immutable)
    .with_parameters(vec!["x", "y"])?;
```

### 4. String Signature
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::string(3, Volatility::Immutable)
    .with_parameters(vec!["str1", "str2", "str3"])?;
```

### 5. Comparable Signature
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::comparable(2, Volatility::Immutable)
    .with_parameters(vec!["a", "b"])?;
```

### 6. Any Signature
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::any(2, Volatility::Immutable)
    .with_parameters(vec!["arg1", "arg2"])?;
```

### 7. Array Signatures
```rust
// Previously NOT supported by from_parameter_variants
let sig = Signature::array(Volatility::Immutable)
    .with_parameters(vec!["arr"])?;

let sig = Signature::array_and_element(Volatility::Immutable)
    .with_parameters(vec!["array", "element"])?;
```

### 8. Coercible Signature
```rust
let string_coercion = Coercion::new_exact(TypeSignatureClass::Native(logical_string()));
let int_coercion = Coercion::new_exact(TypeSignatureClass::Native(logical_int64()));

let sig = Signature::coercible(
    vec![string_coercion, int_coercion],
    Volatility::Immutable,
)
.with_parameters(vec!["str", "num"])?;
```

### 9. OneOf Signature
```rust
let sig = Signature::one_of(
    vec![
        TypeSignature::Exact(vec![DataType::Int32]),
        TypeSignature::Exact(vec![DataType::Int64]),
    ],
    Volatility::Immutable,
)
.with_parameters(vec!["value"])?;
```

### 10. Nullary Signature
```rust
let sig = Signature::nullary(Volatility::Immutable)
    .with_parameters::<&str>(vec![])?;
```

### 11. UserDefined Signature
```rust
// UserDefined is the ONLY variable-arity signature that allows parameter names
let sig = Signature::user_defined(Volatility::Stable)
    .with_parameters(vec!["custom"])?;
```

## Variable-Arity Signatures

Note: Variable-arity signatures (except `UserDefined`) **cannot** have parameter names:

```rust
// These will return errors:
let result = Signature::variadic(vec![DataType::Utf8], Volatility::Immutable)
    .with_parameters(vec!["strings"]); // ❌ Error

let result = Signature::variadic_any(Volatility::Immutable)
    .with_parameters(vec!["args"]); // ❌ Error
```

## Key Advantages

1. **Universal Support**: Works with ALL TypeSignature variants, not just Exact/Coercible
2. **Ergonomic**: Accepts `&str` directly without requiring `.to_string()` calls
3. **Fluent API**: Chainable builder pattern for clean, readable code
4. **Type Safe**: Generic over `AsRef<str>` for maximum flexibility
5. **Consistent**: Same validation rules as existing methods

## When to Use Each Method

- **`with_parameters`**: Use when you already have a `Signature` and want to add parameter names. Works with ALL signature types.
- **`from_parameter_variants`**: Use when defining multiple signature variants with optional parameters (e.g., `substr(str, pos)` OR `substr(str, pos, len)`). Limited to Exact/Coercible/Nullary/OneOf.
- **`with_parameter_names`**: Legacy method, prefer `with_parameters` for its ergonomics.

## Migration Guide

### From `with_parameter_names`
```rust
// Before
.with_parameter_names(vec!["count".to_string(), "name".to_string()])?

// After
.with_parameters(vec!["count", "name"])?
```

### For New Signatures
```rust
// If you were previously unable to add parameter names to Uniform/Numeric/String/etc:
let sig = Signature::uniform(3, vec![DataType::Float64], Volatility::Immutable)
    .with_parameters(vec!["a", "b", "c"])?;
```
