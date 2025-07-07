# Bug Report: Inability to Filter on Unnested Nested Structs

## Description

When attempting to query data containing nested structs and then unnesting these structs, it is currently impossible to apply a filter (e.g., a `WHERE` clause) on the fields of the unnested data. This limitation arises from a combination of how DataFusion's query planner handles column resolution and a specific restriction on `UNNEST` operations on structs within subqueries.

## To Reproduce

**1. Data (`nested.ndjson`):**

```json
{"metadata": {"product": {"name": "Product Name"}}}
```

**2. Original Test Case (Rust):**

This test case attempts to unnest `metadata` and then `product` within subqueries, and then filter on the `name` field.

```rust
use datafusion::common::DataFusionError;
use datafusion::prelude::{NdJsonReadOptions, SessionContext};

#[tokio::test]
async fn test_bug() -> Result<(), DataFusionError> {
  let ctx = SessionContext::new();
  ctx.register_json(
  "mock",
  "tests/fixtures/nested.ndjson", // Adjust path as necessary for your setup
  NdJsonReadOptions::default().file_extension("ndjson")
  ).await?;

  let query = r#"
  WITH one AS (SELECT unnest(metadata) FROM mock),
  two AS (SELECT unnest("__unnest_placeholder(mock.metadata).product") FROM one)
  SELECT * FROM two WHERE "__unnest_placeholder(one.__unnest_placeholder(mock.metadata).product).name" == 'Product Name'
  "#;

  let frame = ctx.sql(query).await?;
  frame.show().await?;

  Ok(())
}
```

**Actual Error with Original Query:**

```
Error: SchemaError(FieldNotFound { field: Column { relation: None, name: "__unnest_placeholder(one.__unnest_placeholder(mock.metadata).product).name" }, valid_fields: [Column { relation: Some(Bare { table: "mock" }), name: "metadata" }] }, Some(""))
```

This error indicates that the internal placeholder name is exposed and not resolvable in the `WHERE` clause.

## Expected Behavior

The query should successfully unnest the nested struct fields and then filter the results based on the specified condition, returning only the matching rows. The internal `__unnest_placeholder` names should not be exposed to the user or cause query failures.

## Actual Behavior and Debugging Analysis

Through extensive debugging, the following behaviors and limitations were observed:

1.  **Simple `UNNEST` Works:**
    A basic `SELECT * FROM UNNEST([1, 2, 3])` query executes successfully, confirming that the `UNNEST` function itself is registered and functional.

2.  **`LATERAL UNNEST` Fails:**
    Attempts to use `LATERAL UNNEST` (e.g., `SELECT * FROM mock, LATERAL UNNEST(metadata)` or `SELECT * FROM mock CROSS JOIN LATERAL UNNEST(metadata)`) consistently fail with:
    ```
    Error: Plan("table function 'unnest' not found")
    ```
    This suggests a specific issue with how `LATERAL UNNEST` is handled by the query planner, despite `UNNEST` being a recognized table function in simpler contexts.

3.  **Direct `UNNEST` on Nested Fields Works (but exposes internal names):**
    A query like `SELECT unnest(metadata.product) FROM mock` successfully unnests the `product` struct and returns its fields. However, the output column names are internal placeholders (e.g., `__unnest_placeholder(mock.metadata[product]).name`).

    **Output:**
    ```
    +---------------------------------------------------+
    | __unnest_placeholder(mock.metadata[product]).name |
    +---------------------------------------------------+
    | Product Name                                      |
    +---------------------------------------------------+
    ```

4.  **Filtering on Direct `UNNEST` Output Fails:**
    Attempting to filter on these internal placeholder names in the same query level (e.g., `WHERE "__unnest_placeholder(mock.metadata[product]).name" = 'Product Name'`) results in the original `SchemaError`:
    ```
    Error: Diagnostic(Diagnostic { kind: Error, message: "column '__unnest_placeholder(mock.metadata[product]).name' not found", span: None, notes: [], helps: [] }, SchemaError(FieldNotFound { field: Column { relation: None, name: "__unnest_placeholder(mock.metadata[product]).name" }, valid_fields: [Column { relation: Some(Bare { table: "mock" }), name: "metadata" }] }, Some("")))
    ```
    This is expected, as `WHERE` clauses are typically evaluated before `SELECT` list expressions, meaning the unnested column is not yet available.

5.  **Subquery Workaround for Filtering Fails due to `UNNEST` Limitation:**
    The standard SQL workaround for the above `SchemaError` is to use a subquery or CTE to perform the `UNNEST` operation first, and then filter in the outer query. However, this approach hits another DataFusion limitation:
    ```sql
    SELECT * FROM (SELECT unnest(metadata.product) FROM mock) AS unnested_data WHERE unnested_data."__unnest_placeholder(mock.metadata[product]).name" = 'Product Name'
    ```
    This query (and variations using `WITH` clauses) results in:
    ```
    Error: Internal("unnest on struct can only be applied at the root level of select expression")
    ```
    This error indicates that `UNNEST` on structs cannot be chained or used within subqueries/CTEs in a way that allows subsequent filtering.

## Conclusion

The combination of these issues creates a scenario where it is currently impossible to filter on the results of unnesting a nested struct in DataFusion. This significantly limits the practical utility of `UNNEST` for complex semi-structured data. The `LATERAL UNNEST` functionality also appears to be broken.

## Environment

*   **DataFusion Version:** Current workspace version (as of July 7, 2025)
*   **Operating System:** darwin
