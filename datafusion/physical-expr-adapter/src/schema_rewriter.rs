// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical expression schema rewriting utilities

use std::sync::Arc;

use arrow::compute::can_cast_types;
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use datafusion_common::{
    exec_err,
    tree_node::{Transformed, TransformedResult, TreeNode},
    Result, ScalarValue,
};
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::{
    expressions::{self, CastColumnExpr, CastExpr, Column},
    ScalarFunctionExpr,
};
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

/// Fast, shallow compatibility check for struct fields to catch obviously incompatible
/// types early without the cost of a full recursive walk. This preserves the behavior
/// of tests that expect early errors while deferring full validation to evaluation time.
fn quick_struct_compatibility_check(
    physical_fields: &Fields,
    logical_fields: &Fields,
) -> Result<()> {
    // Create name-to-field mappings to handle field reordering
    let physical_field_map: std::collections::HashMap<&str, &Field> = physical_fields
        .iter()
        .map(|f| (f.name().as_str(), f.as_ref()))
        .collect();

    let logical_field_map: std::collections::HashMap<&str, &Field> = logical_fields
        .iter()
        .map(|f| (f.name().as_str(), f.as_ref()))
        .collect();

    // Check for truly incompatible type combinations, including nested structs
    for (field_name, logical_field) in &logical_field_map {
        if let Some(physical_field) = physical_field_map.get(field_name) {
            check_field_compatibility(physical_field, logical_field)?;
        }
        // Missing fields are handled by CastColumnExpr at evaluation time
    }

    Ok(())
}

/// Helper function to check if two fields are compatible, including recursive struct checking
fn check_field_compatibility(
    physical_field: &Field,
    logical_field: &Field,
) -> Result<()> {
    let physical_type = physical_field.data_type();
    let logical_type = logical_field.data_type();

    match (physical_type, logical_type) {
        (
            DataType::Struct(physical_struct_fields),
            DataType::Struct(logical_struct_fields),
        ) => {
            // Recursively check nested struct fields for obvious incompatibilities
            quick_struct_compatibility_check(
                physical_struct_fields,
                logical_struct_fields,
            )?;
        }
        _ => {
            // For non-struct types, use Arrow's casting rules to catch obvious incompatibilities
            if !can_cast_types(physical_type, logical_type) {
                return exec_err!(
                    "Cannot cast struct field '{}' from '{}' to '{}'",
                    physical_field.name(),
                    physical_type,
                    logical_type
                );
            }
        }
    }

    Ok(())
}

/// Trait for adapting physical expressions to match a target schema.
///
/// This is used in file scans to rewrite expressions so that they can be evaluated
/// against the physical schema of the file being scanned. It allows for handling
/// differences between logical and physical schemas, such as type mismatches or missing columns.
///
/// ## Overview
///
/// The `PhysicalExprAdapter` allows rewriting physical expressions to match different schemas, including:
///
/// - **Type casting**: When logical and physical schemas have different types, expressions are
///   automatically wrapped with cast operations. For example, `lit(ScalarValue::Int32(123)) = int64_column`
///   gets rewritten to `lit(ScalarValue::Int32(123)) = cast(int64_column, 'Int32')`.
///   Note that this does not attempt to simplify such expressions - that is done by shared simplifiers.
///
/// - **Missing columns**: When a column exists in the logical schema but not in the physical schema,
///   references to it are replaced with null literals.
///
/// - **Struct field access**: Expressions like `struct_column.field_that_is_missing_in_schema` are
///   rewritten to `null` when the field doesn't exist in the physical schema.
///
/// - **Partition columns**: Partition column references can be replaced with their literal values
///   when scanning specific partitions.
///
/// ## Custom Implementations
///
/// You can create a custom implementation of this trait to handle specific rewriting logic.
/// For example, to fill in missing columns with default values instead of nulls:
///
/// ```rust
/// use datafusion_physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory};
/// use arrow::datatypes::{Schema, Field, DataType, FieldRef, SchemaRef};
/// use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
/// use datafusion_common::{Result, ScalarValue, tree_node::{Transformed, TransformedResult, TreeNode}};
/// use datafusion_physical_expr::expressions::{self, Column};
/// use std::sync::Arc;
///
/// #[derive(Debug)]
/// pub struct CustomPhysicalExprAdapter {
///     logical_file_schema: SchemaRef,
///     physical_file_schema: SchemaRef,
/// }
///
/// impl PhysicalExprAdapter for CustomPhysicalExprAdapter {
///     fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
///         expr.transform(|expr| {
///             if let Some(column) = expr.as_any().downcast_ref::<Column>() {
///                 // Check if the column exists in the physical schema
///                 if self.physical_file_schema.index_of(column.name()).is_err() {
///                     // If the column is missing, fill it with a default value instead of null
///                     // The default value could be stored in the table schema's column metadata for example.
///                     let default_value = ScalarValue::Int32(Some(0));
///                     return Ok(Transformed::yes(expressions::lit(default_value)));
///                 }
///             }
///             // If the column exists, return it as is
///             Ok(Transformed::no(expr))
///         }).data()
///     }
///
///     fn with_partition_values(
///         &self,
///         partition_values: Vec<(FieldRef, ScalarValue)>,
///     ) -> Arc<dyn PhysicalExprAdapter> {
///         // For simplicity, this example ignores partition values
///         Arc::new(CustomPhysicalExprAdapter {
///             logical_file_schema: self.logical_file_schema.clone(),
///             physical_file_schema: self.physical_file_schema.clone(),
///         })
///     }
/// }
///
/// #[derive(Debug)]
/// pub struct CustomPhysicalExprAdapterFactory;
///
/// impl PhysicalExprAdapterFactory for CustomPhysicalExprAdapterFactory {
///     fn create(
///         &self,
///         logical_file_schema: SchemaRef,
///         physical_file_schema: SchemaRef,
///     ) -> Arc<dyn PhysicalExprAdapter> {
///         Arc::new(CustomPhysicalExprAdapter {
///             logical_file_schema,
///             physical_file_schema,
///         })
///     }
/// }
/// ```
pub trait PhysicalExprAdapter: Send + Sync + std::fmt::Debug {
    /// Rewrite a physical expression to match the target schema.
    ///
    /// This method should return a transformed expression that matches the target schema.
    ///
    /// Arguments:
    /// - `expr`: The physical expression to rewrite.
    /// - `logical_file_schema`: The logical schema of the table being queried, excluding any partition columns.
    /// - `physical_file_schema`: The physical schema of the file being scanned.
    /// - `partition_values`: Optional partition values to use for rewriting partition column references.
    ///   These are handled as if they were columns appended onto the logical file schema.
    ///
    /// Returns:
    /// - `Arc<dyn PhysicalExpr>`: The rewritten physical expression that can be evaluated against the physical schema.
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>;

    fn with_partition_values(
        &self,
        partition_values: Vec<(FieldRef, ScalarValue)>,
    ) -> Arc<dyn PhysicalExprAdapter>;
}

pub trait PhysicalExprAdapterFactory: Send + Sync + std::fmt::Debug {
    /// Create a new instance of the physical expression adapter.
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExprAdapter>;
}

#[derive(Debug, Clone)]
pub struct DefaultPhysicalExprAdapterFactory;

impl PhysicalExprAdapterFactory for DefaultPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExprAdapter> {
        Arc::new(DefaultPhysicalExprAdapter {
            logical_file_schema,
            physical_file_schema,
            partition_values: Vec::new(),
        })
    }
}

/// Default implementation for rewriting physical expressions to match different schemas.
///
/// # Example
///
/// ```rust
/// use datafusion_physical_expr_adapter::{DefaultPhysicalExprAdapterFactory, PhysicalExprAdapterFactory};
/// use arrow::datatypes::Schema;
/// use std::sync::Arc;
///
/// # fn example(
/// #     predicate: std::sync::Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
/// #     physical_file_schema: &Schema,
/// #     logical_file_schema: &Schema,
/// # ) -> datafusion_common::Result<()> {
/// let factory = DefaultPhysicalExprAdapterFactory;
/// let adapter = factory.create(Arc::new(logical_file_schema.clone()), Arc::new(physical_file_schema.clone()));
/// let adapted_predicate = adapter.rewrite(predicate)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DefaultPhysicalExprAdapter {
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
    partition_values: Vec<(FieldRef, ScalarValue)>,
}

impl DefaultPhysicalExprAdapter {
    /// Create a new instance of the default physical expression adapter.
    ///
    /// This adapter rewrites expressions to match the physical schema of the file being scanned,
    /// handling type mismatches and missing columns by filling them with default values.
    pub fn new(logical_file_schema: SchemaRef, physical_file_schema: SchemaRef) -> Self {
        Self {
            logical_file_schema,
            physical_file_schema,
            partition_values: Vec::new(),
        }
    }
}

impl PhysicalExprAdapter for DefaultPhysicalExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let rewriter = DefaultPhysicalExprAdapterRewriter {
            logical_file_schema: &self.logical_file_schema,
            physical_file_schema: &self.physical_file_schema,
            partition_fields: &self.partition_values,
        };
        expr.transform(|expr| rewriter.rewrite_expr(Arc::clone(&expr)))
            .data()
    }

    fn with_partition_values(
        &self,
        partition_values: Vec<(FieldRef, ScalarValue)>,
    ) -> Arc<dyn PhysicalExprAdapter> {
        Arc::new(DefaultPhysicalExprAdapter {
            partition_values,
            ..self.clone()
        })
    }
}

struct DefaultPhysicalExprAdapterRewriter<'a> {
    logical_file_schema: &'a Schema,
    physical_file_schema: &'a Schema,
    partition_fields: &'a [(FieldRef, ScalarValue)],
}

impl<'a> DefaultPhysicalExprAdapterRewriter<'a> {
    fn rewrite_expr(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        if let Some(transformed) = self.try_rewrite_struct_field_access(&expr)? {
            return Ok(Transformed::yes(transformed));
        }

        if let Some(column) = expr.as_any().downcast_ref::<Column>() {
            return self.rewrite_column(Arc::clone(&expr), column);
        }

        Ok(Transformed::no(expr))
    }

    /// Attempt to rewrite struct field access expressions to return null if the field does not exist in the physical schema.
    /// Note that this does *not* handle nested struct fields, only top-level struct field access.
    /// See <https://github.com/apache/datafusion/issues/17114> for more details.
    fn try_rewrite_struct_field_access(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Result<Option<Arc<dyn PhysicalExpr>>> {
        let get_field_expr =
            match ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref()) {
                Some(expr) => expr,
                None => return Ok(None),
            };

        let source_expr = match get_field_expr.args().first() {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let field_name_expr = match get_field_expr.args().get(1) {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let lit = match field_name_expr
            .as_any()
            .downcast_ref::<expressions::Literal>()
        {
            Some(lit) => lit,
            None => return Ok(None),
        };

        let field_name = match lit.value().try_as_str().flatten() {
            Some(name) => name,
            None => return Ok(None),
        };

        let column = match source_expr.as_any().downcast_ref::<Column>() {
            Some(column) => column,
            None => return Ok(None),
        };

        let physical_field =
            match self.physical_file_schema.field_with_name(column.name()) {
                Ok(field) => field,
                Err(_) => return Ok(None),
            };

        let physical_struct_fields = match physical_field.data_type() {
            DataType::Struct(fields) => fields,
            _ => return Ok(None),
        };

        if physical_struct_fields
            .iter()
            .any(|f| f.name() == field_name)
        {
            return Ok(None);
        }

        let logical_field = match self.logical_file_schema.field_with_name(column.name())
        {
            Ok(field) => field,
            Err(_) => return Ok(None),
        };

        let logical_struct_fields = match logical_field.data_type() {
            DataType::Struct(fields) => fields,
            _ => return Ok(None),
        };

        let logical_struct_field = match logical_struct_fields
            .iter()
            .find(|f| f.name() == field_name)
        {
            Some(field) => field,
            None => return Ok(None),
        };

        let null_value = ScalarValue::Null.cast_to(logical_struct_field.data_type())?;
        Ok(Some(expressions::lit(null_value)))
    }

    fn rewrite_column(
        &self,
        expr: Arc<dyn PhysicalExpr>,
        column: &Column,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        // Get the logical field for this column if it exists in the logical schema
        let logical_field: FieldRef =
            match self.logical_file_schema.field_with_name(column.name()) {
                Ok(field) => Arc::new(field.clone()),
                Err(e) => {
                    // If the column is a partition field, we can use the partition value
                    if let Some(partition_value) = self.get_partition_value(column.name())
                    {
                        return Ok(Transformed::yes(expressions::lit(partition_value)));
                    }
                    // This can be hit if a custom rewrite injected a reference to a column that doesn't exist in the logical schema.
                    // For example, a pre-computed column that is kept only in the physical schema.
                    // If the column exists in the physical schema, we can still use it.
                    if let Ok(physical_field) =
                        self.physical_file_schema.field_with_name(column.name())
                    {
                        // If the column exists in the physical schema, we can use it in place of the logical column.
                        // This is nice to users because if they do a rewrite that results in something like `physical_int32_col = 123u64`
                        // we'll at least handle the casts for them.
                        Arc::new(physical_field.clone())
                    } else {
                        // A completely unknown column that doesn't exist in either schema!
                        // This should probably never be hit unless something upstream broke, but nonetheless it's better
                        // for us to return a handleable error than to panic / do something unexpected.
                        return Err(e.into());
                    }
                }
            };

        // Check if the column exists in the physical schema
        let physical_column_index =
            match self.physical_file_schema.index_of(column.name()) {
                Ok(index) => index,
                Err(_) => {
                    if !logical_field.is_nullable() {
                        return exec_err!(
                        "Non-nullable column '{}' is missing from the physical schema",
                        column.name()
                    );
                    }
                    // If the column is missing from the physical schema fill it in with nulls as `SchemaAdapter` would do.
                    // TODO: do we need to sync this with what the `SchemaAdapter` actually does?
                    // While the default implementation fills in nulls in theory a custom `SchemaAdapter` could do something else!
                    // See https://github.com/apache/datafusion/issues/16527
                    let null_value =
                        ScalarValue::Null.cast_to(logical_field.data_type())?;
                    return Ok(Transformed::yes(expressions::lit(null_value)));
                }
            };
        let physical_field = self.physical_file_schema.field(physical_column_index);

        let column = match (
            column.index() == physical_column_index,
            logical_field.data_type() == physical_field.data_type(),
        ) {
            // If the column index matches and the data types match, we can use the column as is
            (true, true) => return Ok(Transformed::no(expr)),
            // If the indexes or data types do not match, we need to create a new column expression
            (true, _) => column.clone(),
            (false, _) => {
                Column::new_with_schema(logical_field.name(), self.physical_file_schema)?
            }
        };

        if logical_field.data_type() == physical_field.data_type() {
            // If the data types match, we can use the column as is
            return Ok(Transformed::yes(Arc::new(column)));
        }

        if let DataType::Struct(logical_struct_fields) = logical_field.data_type() {
            match physical_field.data_type() {
                DataType::Struct(physical_struct_fields) => {
                    // Perform a quick, non-recursive check to catch obviously incompatible struct field types
                    // early (preserving test expectations), but defer full recursive validation to
                    // CastColumnExpr which provides richer error context and avoids duplicate deep walks.
                    quick_struct_compatibility_check(
                        physical_struct_fields,
                        logical_struct_fields,
                    )?;
                    let column_expr: Arc<dyn PhysicalExpr> = Arc::new(column.clone());
                    let cast_expr = Arc::new(CastColumnExpr::new(
                        column_expr,
                        logical_field.clone(),
                        None,
                    ));
                    return Ok(Transformed::yes(cast_expr));
                }
                _ => {
                    return exec_err!(
                        "Cannot cast column '{}' from '{}' (physical data type) to '{}' (logical data type)",
                        column.name(),
                        physical_field.data_type(),
                        logical_field.data_type()
                    );
                }
            }
        }

        // We need to cast the column to the logical data type
        // TODO: add optimization to move the cast from the column to literal expressions in the case of `col = 123`
        // since that's much cheaper to evalaute.
        // See https://github.com/apache/datafusion/issues/15780#issuecomment-2824716928
        let is_compatible =
            can_cast_types(physical_field.data_type(), logical_field.data_type());
        if !is_compatible {
            return exec_err!(
                "Cannot cast column '{}' from '{}' (physical data type) to '{}' (logical data type)",
                column.name(),
                physical_field.data_type(),
                logical_field.data_type()
            );
        }

        let cast_expr = Arc::new(CastExpr::new(
            Arc::new(column),
            logical_field.data_type().clone(),
            None,
        ));

        Ok(Transformed::yes(cast_expr))
    }

    fn get_partition_value(&self, column_name: &str) -> Option<ScalarValue> {
        self.partition_fields
            .iter()
            .find(|(field, _)| field.name() == column_name)
            .map(|(_, value)| value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, ArrayRef, BooleanArray, Int16Array, Int32Array, Int64Array, RecordBatch,
        RecordBatchOptions, StringArray, StructArray,
    };
    use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
    use datafusion_common::{
        assert_contains, config::ConfigOptions, record_batch, Result, ScalarValue,
    };
    use datafusion_expr::{ColumnarValue, Operator};
    use datafusion_physical_expr::expressions::{
        col, lit, CastColumnExpr, CastExpr, Column, Literal,
    };
    use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
    use itertools::Itertools;
    use std::sync::Arc;

    fn create_test_schema() -> (Schema, Schema) {
        let physical_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]);

        let logical_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false), // Different type
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Float64, true), // Missing from physical
        ]);

        (physical_schema, logical_schema)
    }

    #[test]
    fn test_rewrite_column_with_type_cast() {
        let (physical_schema, logical_schema) = create_test_schema();

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("a", 0));

        let result = adapter.rewrite(column_expr).unwrap();

        // Should be wrapped in a cast expression
        assert!(result.as_any().downcast_ref::<CastExpr>().is_some());
    }

    #[test]
    fn test_rewrite_multi_column_expr_with_type_cast() {
        let (physical_schema, logical_schema) = create_test_schema();
        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));

        // Create a complex expression: (a + 5) OR (c > 0.0) that tests the recursive case of the rewriter
        let column_a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let column_c = Arc::new(Column::new("c", 2)) as Arc<dyn PhysicalExpr>;
        let expr = expressions::BinaryExpr::new(
            Arc::clone(&column_a),
            Operator::Plus,
            Arc::new(expressions::Literal::new(ScalarValue::Int64(Some(5)))),
        );
        let expr = expressions::BinaryExpr::new(
            Arc::new(expr),
            Operator::Or,
            Arc::new(expressions::BinaryExpr::new(
                Arc::clone(&column_c),
                Operator::Gt,
                Arc::new(expressions::Literal::new(ScalarValue::Float64(Some(0.0)))),
            )),
        );

        let result = adapter.rewrite(Arc::new(expr)).unwrap();
        println!("Rewritten expression: {result}");

        let expected = expressions::BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 0)),
                DataType::Int64,
                None,
            )),
            Operator::Plus,
            Arc::new(expressions::Literal::new(ScalarValue::Int64(Some(5)))),
        );
        let expected = Arc::new(expressions::BinaryExpr::new(
            Arc::new(expected),
            Operator::Or,
            Arc::new(expressions::BinaryExpr::new(
                lit(ScalarValue::Float64(None)), // c is missing, so it becomes null
                Operator::Gt,
                Arc::new(expressions::Literal::new(ScalarValue::Float64(Some(0.0)))),
            )),
        )) as Arc<dyn PhysicalExpr>;

        assert_eq!(
            result.to_string(),
            expected.to_string(),
            "The rewritten expression did not match the expected output"
        );
    }

    #[test]
    fn test_rewrite_struct_column_incompatible() {
        let physical_schema = Schema::new(vec![Field::new(
            "data",
            DataType::Struct(vec![Field::new("field1", DataType::Binary, true)].into()),
            true,
        )]);

        let logical_schema = Schema::new(vec![Field::new(
            "data",
            DataType::Struct(vec![Field::new("field1", DataType::Int32, true)].into()),
            true,
        )]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("data", 0));

        let error_msg = adapter.rewrite(column_expr).unwrap_err().to_string();
        assert_contains!(error_msg, "Cannot cast struct field 'field1'");
    }

    #[test]
    fn test_rewrite_struct_compatible_cast() {
        let physical_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new("id", DataType::Int32, false),
                    Field::new("name", DataType::Utf8, true),
                ]
                .into(),
            ),
            false,
        )]));

        let logical_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new("id", DataType::Int64, false),
                    Field::new("name", DataType::Utf8View, true),
                ]
                .into(),
            ),
            false,
        )]));

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));
        let column_expr = Arc::new(Column::new("data", 0));

        let result = adapter.rewrite(column_expr).unwrap();

        let expected = Arc::new(CastColumnExpr::new(
            Arc::new(Column::new("data", 0)),
            Arc::new(Field::new(
                "data",
                DataType::Struct(
                    vec![
                        Field::new("id", DataType::Int64, false),
                        Field::new("name", DataType::Utf8View, true),
                    ]
                    .into(),
                ),
                false,
            )),
            None,
        )) as Arc<dyn PhysicalExpr>;

        assert!(result.as_any().downcast_ref::<CastColumnExpr>().is_some());
        assert_eq!(result.to_string(), expected.to_string());
    }

    #[test]
    fn test_rewrite_struct_cast_evaluates_with_missing_fields() -> Result<()> {
        let physical_struct_fields: Fields = vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]
        .into();
        let physical_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Struct(physical_struct_fields.clone()),
            true,
        )]));

        let logical_struct_fields: Fields = vec![
            Field::new("id", DataType::Int64, true),
            Field::new("age", DataType::Int32, true),
        ]
        .into();
        let logical_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Struct(logical_struct_fields.clone()),
            false,
        )]));

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));
        let expr = adapter.rewrite(Arc::new(Column::new("data", 0)))?;

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let name_array: ArrayRef =
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob")]));
        let struct_array: ArrayRef = Arc::new(StructArray::new(
            physical_struct_fields,
            vec![id_array, name_array],
            None,
        ));
        let batch = RecordBatch::try_new(
            Arc::clone(&physical_schema),
            vec![Arc::clone(&struct_array)],
        )?;

        let ColumnarValue::Array(result_array) = expr.evaluate(&batch)? else {
            panic!("expected array result");
        };

        assert_eq!(
            result_array.data_type(),
            &DataType::Struct(logical_struct_fields.clone())
        );
        let result_struct = result_array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("struct result");

        let cast_id = result_struct
            .column_by_name("id")
            .expect("id field")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 id");
        assert_eq!(cast_id.len(), 2);
        assert_eq!(cast_id.value(0), 1);
        assert!(cast_id.is_null(1));

        let filled_age = result_struct
            .column_by_name("age")
            .expect("age field")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int32 age");
        assert_eq!(filled_age.len(), 2);
        assert!(filled_age.is_null(0));
        assert!(filled_age.is_null(1));

        let return_field = expr.return_field(physical_schema.as_ref())?;
        assert!(return_field.is_nullable());
        assert_eq!(
            return_field.data_type(),
            &DataType::Struct(logical_struct_fields)
        );

        Ok(())
    }

    #[test]
    fn test_rewrite_column_preserves_metadata() -> Result<()> {
        use std::collections::HashMap;

        let physical_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let mut metadata = HashMap::new();
        metadata.insert("hint".to_string(), "value".to_string());
        let logical_field =
            Field::new("a", DataType::Int64, false).with_metadata(metadata);
        let logical_schema = Arc::new(Schema::new(vec![logical_field.clone()]));

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));
        let column_expr = Arc::new(Column::new("a", 0));

        let result = adapter.rewrite(column_expr)?;
        let returned_field = result.return_field(logical_schema.as_ref())?;

        assert_eq!(returned_field.metadata(), logical_field.metadata());
        Ok(())
    }

    #[test]
    fn test_rewrite_missing_column() -> Result<()> {
        let (physical_schema, logical_schema) = create_test_schema();

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("c", 2));

        let result = adapter.rewrite(column_expr)?;

        // Should be replaced with a literal null
        if let Some(literal) = result.as_any().downcast_ref::<expressions::Literal>() {
            assert_eq!(*literal.value(), ScalarValue::Float64(None));
        } else {
            panic!("Expected literal expression");
        }

        Ok(())
    }

    #[test]
    fn test_rewrite_missing_column_non_nullable_error() {
        let physical_schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let logical_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false), // Missing and non-nullable
        ]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("b", 1));

        let error_msg = adapter.rewrite(column_expr).unwrap_err().to_string();
        assert_contains!(error_msg, "Non-nullable column 'b' is missing");
    }

    #[test]
    fn test_rewrite_missing_column_nullable() {
        let physical_schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let logical_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, true), // Missing but nullable
        ]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("b", 1));

        let result = adapter.rewrite(column_expr).unwrap();

        let expected =
            Arc::new(Literal::new(ScalarValue::Utf8(None))) as Arc<dyn PhysicalExpr>;

        assert_eq!(result.to_string(), expected.to_string());
    }

    #[test]
    fn test_rewrite_partition_column() -> Result<()> {
        let (physical_schema, logical_schema) = create_test_schema();

        let partition_field =
            Arc::new(Field::new("partition_col", DataType::Utf8, false));
        let partition_value = ScalarValue::Utf8(Some("test_value".to_string()));
        let partition_values = vec![(partition_field, partition_value)];

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let adapter = adapter.with_partition_values(partition_values);

        let column_expr = Arc::new(Column::new("partition_col", 0));
        let result = adapter.rewrite(column_expr)?;

        // Should be replaced with the partition value
        if let Some(literal) = result.as_any().downcast_ref::<expressions::Literal>() {
            assert_eq!(
                *literal.value(),
                ScalarValue::Utf8(Some("test_value".to_string()))
            );
        } else {
            panic!("Expected literal expression");
        }

        Ok(())
    }

    #[test]
    fn test_rewrite_no_change_needed() -> Result<()> {
        let (physical_schema, logical_schema) = create_test_schema();

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("b", 1)) as Arc<dyn PhysicalExpr>;

        let result = adapter.rewrite(Arc::clone(&column_expr))?;

        // Should be the same expression (no transformation needed)
        // We compare the underlying pointer through the trait object
        assert!(std::ptr::eq(
            column_expr.as_ref() as *const dyn PhysicalExpr,
            result.as_ref() as *const dyn PhysicalExpr
        ));

        Ok(())
    }

    #[test]
    fn test_non_nullable_missing_column_error() {
        let physical_schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let logical_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false), // Non-nullable missing column
        ]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter = factory.create(Arc::new(logical_schema), Arc::new(physical_schema));
        let column_expr = Arc::new(Column::new("b", 1));

        let result = adapter.rewrite(column_expr);
        assert!(result.is_err());
        assert_contains!(
            result.unwrap_err().to_string(),
            "Non-nullable column 'b' is missing from the physical schema"
        );
    }

    /// Helper function to project expressions onto a RecordBatch
    fn batch_project(
        expr: Vec<Arc<dyn PhysicalExpr>>,
        batch: &RecordBatch,
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        let arrays = expr
            .iter()
            .map(|expr| {
                expr.evaluate(batch)
                    .and_then(|v| v.into_array(batch.num_rows()))
            })
            .collect::<Result<Vec<_>>>()?;

        if arrays.is_empty() {
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            RecordBatch::try_new_with_options(Arc::clone(&schema), arrays, &options)
                .map_err(Into::into)
        } else {
            RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(Into::into)
        }
    }

    fn field_ref(name: &str, data_type: DataType) -> FieldRef {
        field_ref_with_nullability(name, data_type, true)
    }

    fn field_ref_with_nullability(
        name: &str,
        data_type: DataType,
        nullable: bool,
    ) -> FieldRef {
        Arc::new(Field::new(name, data_type, nullable))
    }

    fn struct_field_ref(name: &str, fields: Vec<FieldRef>) -> FieldRef {
        struct_field_ref_with_nullability(name, fields, true)
    }

    fn struct_field_ref_with_nullability(
        name: &str,
        fields: Vec<FieldRef>,
        nullable: bool,
    ) -> FieldRef {
        // `arrow::datatypes::Fields` is a type alias for `Vec<FieldRef>`, so converting here
        // simply transfers ownership of the provided field references into the Struct type.
        Arc::new(Field::new(name, DataType::Struct(fields.into()), nullable))
    }

    fn schema_from_field_refs(fields: Vec<FieldRef>) -> SchemaRef {
        schema_from_fields(
            fields
                .into_iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        )
    }

    fn schema_from_fields(fields: Vec<Field>) -> SchemaRef {
        Arc::new(Schema::new(fields))
    }

    fn struct_array_from(pairs: Vec<(&FieldRef, ArrayRef)>) -> StructArray {
        StructArray::from(
            pairs
                .into_iter()
                .map(|(field, array)| (Arc::clone(field), array))
                .collect::<Vec<_>>(),
        )
    }

    fn int16_array(values: &[Option<i16>]) -> ArrayRef {
        Arc::new(Int16Array::from(values.to_vec())) as ArrayRef
    }

    fn int32_array(values: &[Option<i32>]) -> ArrayRef {
        Arc::new(Int32Array::from(values.to_vec())) as ArrayRef
    }

    fn string_array(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec())) as ArrayRef
    }

    fn record_batch_from_struct(
        field: &FieldRef,
        struct_array: StructArray,
    ) -> Result<RecordBatch> {
        RecordBatch::try_new(
            schema_from_field_refs(vec![Arc::clone(field)]),
            vec![Arc::new(struct_array) as ArrayRef],
        )
        .map_err(Into::into)
    }

    fn get_field_expr(
        base: Arc<dyn PhysicalExpr>,
        field_name: &str,
        logical_schema: &SchemaRef,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        ScalarFunctionExpr::try_new(
            datafusion_functions::core::get_field(),
            vec![
                base,
                Arc::new(expressions::Literal::new(ScalarValue::Utf8(Some(
                    field_name.to_string(),
                )))) as Arc<dyn PhysicalExpr>,
            ],
            logical_schema.as_ref(),
            Arc::new(ConfigOptions::default()),
        )
        .map(|expr| Arc::new(expr) as Arc<dyn PhysicalExpr>)
    }

    fn boolean_values(
        expr: Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
    ) -> Result<Vec<Option<bool>>> {
        let values = expr.evaluate(batch)?.into_array(batch.num_rows())?;
        let bool_values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
        Ok(bool_values.iter().collect())
    }

    /// Example showing how we can use the `DefaultPhysicalExprAdapter` to adapt RecordBatches during a scan
    /// to apply projections, type conversions and handling of missing columns all at once.
    #[test]
    fn test_adapt_batches() {
        let physical_batch = record_batch!(
            ("a", Int32, vec![Some(1), None, Some(3)]),
            ("extra", Utf8, vec![Some("x"), Some("y"), None])
        )
        .unwrap();

        let physical_schema = physical_batch.schema();

        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true), // Different type
            Field::new("b", DataType::Utf8, true),  // Missing from physical
        ]));

        let projection = vec![
            col("b", &logical_schema).unwrap(),
            col("a", &logical_schema).unwrap(),
        ];

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));

        let adapted_projection = projection
            .into_iter()
            .map(|expr| adapter.rewrite(expr).unwrap())
            .collect_vec();

        let adapted_schema = Arc::new(Schema::new(
            adapted_projection
                .iter()
                .map(|expr| expr.return_field(&physical_schema).unwrap())
                .collect_vec(),
        ));

        let res = batch_project(
            adapted_projection,
            &physical_batch,
            Arc::clone(&adapted_schema),
        )
        .unwrap();

        assert_eq!(res.num_columns(), 2);
        assert_eq!(res.column(0).data_type(), &DataType::Utf8);
        assert_eq!(res.column(1).data_type(), &DataType::Int64);
        assert_eq!(
            res.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap()
                .iter()
                .collect_vec(),
            vec![None, None, None]
        );
        assert_eq!(
            res.column(1)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap()
                .iter()
                .collect_vec(),
            vec![Some(1), None, Some(3)]
        );
    }

    #[test]
    fn test_rewrite_struct_column_cast() -> Result<()> {
        let score_field = field_ref("score", DataType::Int16);
        let tag_field = field_ref("tag", DataType::Utf8);
        let info_field = struct_field_ref(
            "info",
            vec![Arc::clone(&score_field), Arc::clone(&tag_field)],
        );
        let id_field = field_ref("id", DataType::Int32);
        let extra_field = field_ref("extra", DataType::Utf8);
        let physical_struct_field = struct_field_ref(
            "struct_col",
            vec![
                Arc::clone(&id_field),
                Arc::clone(&info_field),
                Arc::clone(&extra_field),
            ],
        );

        let info_array = struct_array_from(vec![
            (&score_field, int16_array(&[Some(10), Some(20)])),
            (&tag_field, string_array(&[Some("foo"), None])),
        ]);
        let struct_array = struct_array_from(vec![
            (&id_field, int32_array(&[Some(1), Some(2)])),
            (&info_field, Arc::new(info_array) as ArrayRef),
            (&extra_field, string_array(&[Some("drop"), Some("me")])),
        ]);
        let physical_batch =
            record_batch_from_struct(&physical_struct_field, struct_array)?;
        let physical_schema = physical_batch.schema();

        let logical_score_field = field_ref("score", DataType::Int32);
        let logical_tag_field = field_ref("tag", DataType::Utf8);
        let logical_flag_field = field_ref("flag", DataType::Boolean);
        let logical_info_field = struct_field_ref(
            "info",
            vec![
                Arc::clone(&logical_score_field),
                Arc::clone(&logical_tag_field),
                Arc::clone(&logical_flag_field),
            ],
        );
        let logical_id_field = field_ref("id", DataType::Int64);
        let logical_missing_field = field_ref("missing", DataType::Utf8);
        let logical_struct_field = struct_field_ref(
            "struct_col",
            vec![
                Arc::clone(&logical_id_field),
                Arc::clone(&logical_info_field),
                Arc::clone(&logical_missing_field),
            ],
        );
        let logical_schema = schema_from_field_refs(vec![logical_struct_field]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));

        let struct_col_expr = col("struct_col", &logical_schema)?;
        let rewritten_struct = adapter.rewrite(Arc::clone(&struct_col_expr))?;
        assert!(rewritten_struct
            .as_any()
            .downcast_ref::<CastColumnExpr>()
            .is_some());

        let struct_values = rewritten_struct
            .evaluate(&physical_batch)?
            .into_array(physical_batch.num_rows())?;
        let struct_values = struct_values
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(struct_values.fields().len(), 3);

        let id_cast = struct_values
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id_cast.iter().collect_vec(), vec![Some(1), Some(2)]);

        let info_cast = struct_values
            .column_by_name("info")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let score_cast = info_cast
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(score_cast.iter().collect_vec(), vec![Some(10), Some(20)]);
        let flag_cast = info_cast
            .column_by_name("flag")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(flag_cast.iter().collect_vec(), vec![None, None]);

        let missing_cast = struct_values
            .column_by_name("missing")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(missing_cast.iter().collect_vec(), vec![None, None]);

        let get_id_expr =
            get_field_expr(Arc::clone(&struct_col_expr), "id", &logical_schema)?;
        let predicate = Arc::new(expressions::BinaryExpr::new(
            Arc::clone(&get_id_expr),
            Operator::Gt,
            Arc::new(expressions::Literal::new(ScalarValue::Int64(Some(1)))),
        )) as Arc<dyn PhysicalExpr>;

        let predicate_values =
            boolean_values(adapter.rewrite(predicate)?, &physical_batch)?;
        assert_eq!(predicate_values, vec![Some(false), Some(true)]);

        Ok(())
    }

    #[test]
    fn test_rewrite_nested_struct_field_reordering() -> Result<()> {
        let tag_field = field_ref("tag", DataType::Utf8);
        let score_field = field_ref("score", DataType::Int16);
        let info_field = struct_field_ref(
            "info",
            vec![Arc::clone(&tag_field), Arc::clone(&score_field)],
        );
        let id_field = field_ref("id", DataType::Int32);
        let physical_struct_field = struct_field_ref(
            "struct_col",
            vec![Arc::clone(&info_field), Arc::clone(&id_field)],
        );

        let info_array = struct_array_from(vec![
            (&tag_field, string_array(&[Some("a"), Some("b")])),
            (&score_field, int16_array(&[Some(5), Some(15)])),
        ]);
        let struct_array = struct_array_from(vec![
            (&info_field, Arc::new(info_array) as ArrayRef),
            (&id_field, int32_array(&[Some(10), Some(20)])),
        ]);
        let physical_batch =
            record_batch_from_struct(&physical_struct_field, struct_array)?;
        let physical_schema = physical_batch.schema();

        let logical_score_field = field_ref("score", DataType::Int32);
        let logical_flag_field = field_ref("flag", DataType::Boolean);
        let logical_tag_field = field_ref("tag", DataType::Utf8);
        let logical_info_field = struct_field_ref(
            "info",
            vec![
                Arc::clone(&logical_score_field),
                Arc::clone(&logical_flag_field),
                Arc::clone(&logical_tag_field),
            ],
        );
        let logical_id_field = field_ref("id", DataType::Int64);
        let logical_missing_field = field_ref("missing", DataType::Utf8);
        let logical_struct_field = struct_field_ref(
            "struct_col",
            vec![
                Arc::clone(&logical_id_field),
                Arc::clone(&logical_info_field),
                Arc::clone(&logical_missing_field),
            ],
        );
        let logical_schema = schema_from_field_refs(vec![logical_struct_field]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));

        let struct_col_expr = col("struct_col", &logical_schema)?;

        let get_info_expr =
            get_field_expr(Arc::clone(&struct_col_expr), "info", &logical_schema)?;
        let get_score_expr =
            get_field_expr(Arc::clone(&get_info_expr), "score", &logical_schema)?;

        let predicate = Arc::new(expressions::BinaryExpr::new(
            Arc::clone(&get_score_expr),
            Operator::Gt,
            Arc::new(expressions::Literal::new(ScalarValue::Int32(Some(10))))
                as Arc<dyn PhysicalExpr>,
        )) as Arc<dyn PhysicalExpr>;

        let predicate_values =
            boolean_values(adapter.rewrite(predicate)?, &physical_batch)?;
        assert_eq!(predicate_values, vec![Some(false), Some(true)]);

        let get_flag_expr =
            get_field_expr(Arc::clone(&get_info_expr), "flag", &logical_schema)?;
        let flag_values =
            boolean_values(adapter.rewrite(get_flag_expr)?, &physical_batch)?;
        assert_eq!(flag_values, vec![None, None]);

        Ok(())
    }

    #[test]
    fn test_rewrite_nested_struct_incompatible_types() {
        let physical_info_field = struct_field_ref(
            "info",
            vec![
                field_ref("tag", DataType::Utf8),
                field_ref("score", DataType::Binary),
            ],
        );
        let physical_struct_field =
            struct_field_ref("struct_col", vec![physical_info_field]);
        let logical_info_field = struct_field_ref(
            "info",
            vec![
                field_ref("score", DataType::Int32),
                field_ref("tag", DataType::Utf8),
            ],
        );
        let logical_struct_field =
            struct_field_ref("struct_col", vec![logical_info_field]);

        let physical_schema =
            schema_from_field_refs(vec![Arc::clone(&physical_struct_field)]);
        let logical_schema =
            schema_from_field_refs(vec![Arc::clone(&logical_struct_field)]);

        let factory = DefaultPhysicalExprAdapterFactory;
        let adapter =
            factory.create(Arc::clone(&logical_schema), Arc::clone(&physical_schema));

        let struct_col_expr = col("struct_col", &logical_schema).unwrap();
        let error_msg = adapter.rewrite(struct_col_expr).unwrap_err().to_string();
        assert_contains!(error_msg, "Cannot cast struct field 'score'");
    }

    #[test]
    fn test_try_rewrite_struct_field_access() {
        // Test the core logic of try_rewrite_struct_field_access
        let physical_schema = Schema::new(vec![Field::new(
            "struct_col",
            DataType::Struct(
                vec![Field::new("existing_field", DataType::Int32, true)].into(),
            ),
            true,
        )]);

        let logical_schema = Schema::new(vec![Field::new(
            "struct_col",
            DataType::Struct(
                vec![
                    Field::new("existing_field", DataType::Int32, true),
                    Field::new("missing_field", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        )]);

        let rewriter = DefaultPhysicalExprAdapterRewriter {
            logical_file_schema: &logical_schema,
            physical_file_schema: &physical_schema,
            partition_fields: &[],
        };

        // Test that when a field exists in physical schema, it returns None
        let column = Arc::new(Column::new("struct_col", 0)) as Arc<dyn PhysicalExpr>;
        let result = rewriter.try_rewrite_struct_field_access(&column).unwrap();
        assert!(result.is_none());

        // The actual test for the get_field expression would require creating a proper ScalarFunctionExpr
        // with ScalarUDF, which is complex to set up in a unit test. The integration tests in
        // datafusion/core/tests/parquet/schema_adapter.rs provide better coverage for this functionality.
    }
}
