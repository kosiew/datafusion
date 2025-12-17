# Test Coverage Analysis: SchemaAdapter Removal vs PhysicalExprAdapter Coverage

## Deleted Tests (752 lines, 4 test functions)

### 1. test_parquet_flipped_projection
**Scenarios covered:**
- Reading Parquet where table schema column order differs from file schema
- Projection pushdown with reordered columns
- Type matching between file and table schemas

**Current equivalent:** 
- Partially: Physical expr adapter tests cover column matching
- MISSING: End-to-end integration test for column reordering in Parquet scan

### 2. test_parquet_missing_column  
**Scenarios covered:**
- File missing columns present in table schema
- Missing columns filled with NULL
- Projection with missing columns

**Current equivalent:**
- ✓ Unit tests: `test_rewrite_missing_column()`, `test_rewrite_missing_column_nullable()`
- ✓ Integration test: `schema_evolution.slt` covers missing columns
- QUALITY: SLT test is comprehensive but doesn't test the PhysicalExprAdapter path directly

### 3. test_parquet_integration_with_physical_expr_adapter
**Scenarios covered:**
- Custom adapter (UppercasePhysicalExprAdapterFactory)
- Column name mapping (lowercase ↔ uppercase)  
- Filter pushdown with adapted expressions
- Projection with adapted expressions

**Current equivalent:**
- ✓ Unit tests: `test_replace_columns_with_literals()`, custom adapter example docs
- ✓ Example: `default_column_values.rs` shows custom implementation
- MISSING: Integration test exercising custom adapter in actual Parquet scan

### 4. test_multi_source_schema_adapter_reuse
**Scenarios covered:**
- Reusing adapter across multiple file sources
- Consistency of adaptation across formats

**Current equivalent:**
- MISSING: No direct test of adapter reuse across formats

## Existing Tests in physical-expr-adapter

### Unit Tests (14 tests in schema_rewriter.rs)
1. `test_rewrite_column_with_type_cast` - Type casting single column
2. `test_rewrite_multi_column_expr_with_type_cast` - Multi-column type casting
3. `test_rewrite_struct_column_incompatible` - Struct type incompatibility
4. `test_rewrite_struct_compatible_cast` - Struct compatible casting  
5. `test_rewrite_missing_column` - Missing column handling
6. `test_rewrite_missing_column_non_nullable_error` - Error on missing non-nullable
7. `test_rewrite_missing_column_nullable` - Missing nullable column as NULL
8. `test_replace_columns_with_literals` - Literal column replacement
9. `test_replace_columns_with_literals_no_match` - No-match case
10. `test_replace_columns_with_literals_nested_expr` - Nested expression rewriting
11. `test_rewrite_no_change_needed` - No-op rewriting
12. `test_non_nullable_missing_column_error` - Error handling
13. `test_adapt_batches` - RecordBatch adaptation  
14. `test_adapt_struct_batches` - Struct RecordBatch adaptation
15. `test_try_rewrite_struct_field_access` - Struct field access

**Assessment:** Comprehensive unit test coverage of PhysicalExprAdapter functionality

## SQL Logic Tests

### schema_evolution.slt (141 lines)
**Scenarios covered:**
- File 1: Subset of columns (a, b) vs table (a, b, c)
- File 2: Single column (b) 
- File 3: Different column names (z, a) vs table (a, b, c)
- File 4: Same columns in different order with type matching (b, a, c)
- Queries: SELECT *, WHERE a='foo', WHERE a != 'foo', WHERE a IS NULL, WHERE b > 5, WHERE b < 150, WHERE c > 11.0

**Assessment:** Good coverage of basic schema evolution, but doesn't directly test PhysicalExprAdapter implementation

## Coverage Gap Summary

### HIGH CONFIDENCE (Covered)
✓ Type casting (unit + integration)
✓ Missing column handling (unit + integration)
✓ Struct field handling (unit)
✓ Expression rewriting (unit)
✓ Literal replacement (unit)

### MEDIUM CONFIDENCE (Partially Covered)
? End-to-end integration with PhysicalExprAdapter in real Parquet scans
? Custom adapter implementations (only documented via example)
? Filter/projection pushdown with adapted expressions

### GAPS (Not Covered or Unclear)
✗ Integration test for column reordering (flipped projection) in Parquet
✗ Integration test for custom PhysicalExprAdapterFactory in actual file scan
✗ Adapter reuse across multiple file sources
✗ Type coercion at expression level vs batch level (performance/correctness)