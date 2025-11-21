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

use crate::logical_plan::producer::{
    to_substrait_literal, to_substrait_named_struct, SubstraitProducer,
};
use datafusion::common::{not_impl_err, substrait_datafusion_err, DFSchema, ToDFSchema};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{EmptyRelation, Expr, TableScan, Values};
use pbjson_types::Any as ProtoAny;
use std::sync::Arc;
use substrait::proto::expression::literal::Struct;
use substrait::proto::expression::mask_expression::{StructItem, StructSelect};
use substrait::proto::expression::MaskExpression;
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::read_rel::{NamedTable, ReadType, VirtualTable};
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, ReadRel, Rel};

pub fn from_table_scan(
    producer: &mut impl SubstraitProducer,
    scan: &TableScan,
) -> datafusion::common::Result<Box<Rel>> {
    let projection = scan.projection.as_ref().map(|p| {
        p.iter()
            .map(|i| StructItem {
                field: *i as i32,
                child: None,
            })
            .collect()
    });

    let projection = projection.map(|struct_items| MaskExpression {
        select: Some(StructSelect { struct_items }),
        maintain_singular_struct: false,
    });

    let table_schema = scan.source.schema().to_dfschema_ref()?;
    let base_schema = to_substrait_named_struct(producer, &table_schema)?;

    let filter_option = if scan.filters.is_empty() {
        None
    } else {
        let table_schema_qualified = Arc::new(
            DFSchema::try_from_qualified_schema(
                scan.table_name.clone(),
                &(scan.source.schema()),
            )
            .unwrap(),
        );

        let combined_expr = conjunction(scan.filters.clone()).unwrap();
        let filter_expr =
            producer.handle_expr(&combined_expr, &table_schema_qualified)?;
        Some(Box::new(filter_expr))
    };

    // Encode table function metadata if present
    let advanced_extension = if let Some((func_name, args)) = &scan.table_function_call {
        let mut arg_expressions = Vec::new();
        let empty_schema = Arc::new(DFSchema::empty());
        for arg in args {
            arg_expressions.push(producer.handle_expr(arg, &empty_schema)?);
        }

        // Create a protobuf Any message containing the table function metadata
        // Type URL follows the pattern: datafusion.io/TableFunctionCall
        let metadata = TableFunctionCallMetadata {
            function_name: func_name.clone(),
            arguments: arg_expressions,
        };

        Some(encode_table_function_metadata(&metadata)?)
    } else {
        None
    };

    Ok(Box::new(Rel {
        rel_type: Some(RelType::Read(Box::new(ReadRel {
            common: None,
            base_schema: Some(base_schema),
            filter: filter_option,
            best_effort_filter: None,
            projection,
            advanced_extension: None,
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: scan.table_name.to_vec(),
                advanced_extension: advanced_extension.map(|ext| AdvancedExtension {
                    optimization: vec![],
                    enhancement: Some(ext),
                }),
            })),
        }))),
    }))
}

/// Metadata for table function calls, serialized in NamedTable::advanced_extension
///
/// This structure is encoded into a Substrait AdvancedExtension to preserve table function
/// information across serialization boundaries. Since Substrait doesn't have native support
/// for DataFusion's table functions, we use a custom extension with type URL
/// "datafusion.io/TableFunctionCall".
struct TableFunctionCallMetadata {
    function_name: String,
    arguments: Vec<Expression>,
}

/// Encode table function metadata into a Protobuf Any message.
///
/// # Wire Format
///
/// The binary encoding uses a simple, robust format:
/// ```text
/// [function_name (UTF-8 bytes)][0x00 (null terminator)]
/// [arg_count (u32, little-endian)]
/// [arg_0_length (u32, LE)][arg_0_bytes (protobuf-encoded Expression)]
/// [arg_1_length (u32, LE)][arg_1_bytes (protobuf-encoded Expression)]
/// ...
/// ```
///
/// This format is:
/// - Simple to parse and validate
/// - Self-describing (includes lengths)
/// - Forward-compatible (can add version byte in future if needed)
/// - Uses standard protobuf encoding for complex argument expressions
///
/// # Arguments
///
/// * `metadata` - The table function name and argument expressions to encode
///
/// # Returns
///
/// A `pbjson_types::Any` with type URL "datafusion.io/TableFunctionCall"
fn encode_table_function_metadata(
    metadata: &TableFunctionCallMetadata,
) -> datafusion::common::Result<ProtoAny> {
    use prost::Message;

    // Validate function name doesn't contain null bytes
    if metadata.function_name.contains('\0') {
        return datafusion::common::plan_err!(
            "Table function name cannot contain null bytes: {}",
            metadata.function_name
        );
    }

    // Encode each argument as protobuf
    let mut encoded_args = Vec::new();
    for (idx, arg) in metadata.arguments.iter().enumerate() {
        let mut buf = Vec::new();
        arg.encode(&mut buf).map_err(|e| {
            datafusion::common::plan_datafusion_err!(
                "Failed to encode table function argument {idx}: {e}"
            )
        })?;
        encoded_args.push(buf);
    }

    // Build the wire format
    let mut combined = Vec::new();

    // Function name with null terminator
    combined.extend_from_slice(metadata.function_name.as_bytes());
    combined.push(0);

    // Argument count
    let arg_count = encoded_args.len();
    if arg_count > u32::MAX as usize {
        return datafusion::common::plan_err!(
            "Table function has too many arguments: {arg_count}"
        );
    }
    combined.extend_from_slice(&(arg_count as u32).to_le_bytes());

    // Each argument with length prefix
    for arg_bytes in encoded_args {
        let len = arg_bytes.len();
        if len > u32::MAX as usize {
            return datafusion::common::plan_err!(
                "Table function argument too large: {len} bytes"
            );
        }
        combined.extend_from_slice(&(len as u32).to_le_bytes());
        combined.extend_from_slice(&arg_bytes);
    }

    Ok(ProtoAny {
        type_url: "datafusion.io/TableFunctionCall".to_string(),
        value: combined.into(),
    })
}

pub fn from_empty_relation(
    producer: &mut impl SubstraitProducer,
    e: &EmptyRelation,
) -> datafusion::common::Result<Box<Rel>> {
    if e.produce_one_row {
        return not_impl_err!("Producing a row from empty relation is unsupported");
    }
    #[allow(deprecated)]
    Ok(Box::new(Rel {
        rel_type: Some(RelType::Read(Box::new(ReadRel {
            common: None,
            base_schema: Some(to_substrait_named_struct(producer, &e.schema)?),
            filter: None,
            best_effort_filter: None,
            projection: None,
            advanced_extension: None,
            read_type: Some(ReadType::VirtualTable(VirtualTable {
                values: vec![],
                expressions: vec![],
            })),
        }))),
    }))
}

pub fn from_values(
    producer: &mut impl SubstraitProducer,
    v: &Values,
) -> datafusion::common::Result<Box<Rel>> {
    let values = v
        .values
        .iter()
        .map(|row| {
            let fields = row
                .iter()
                .map(|v| match v {
                    Expr::Literal(sv, _) => to_substrait_literal(producer, sv),
                    Expr::Alias(alias) => match alias.expr.as_ref() {
                        // The schema gives us the names, so we can skip aliases
                        Expr::Literal(sv, _) => to_substrait_literal(producer, sv),
                        _ => Err(substrait_datafusion_err!(
                                    "Only literal types can be aliased in Virtual Tables, got: {}", alias.expr.variant_name()
                                )),
                    },
                    _ => Err(substrait_datafusion_err!(
                                "Only literal types and aliases are supported in Virtual Tables, got: {}", v.variant_name()
                            )),
                })
                .collect::<datafusion::common::Result<_>>()?;
            Ok(Struct { fields })
        })
        .collect::<datafusion::common::Result<_>>()?;
    #[allow(deprecated)]
    Ok(Box::new(Rel {
        rel_type: Some(RelType::Read(Box::new(ReadRel {
            common: None,
            base_schema: Some(to_substrait_named_struct(producer, &v.schema)?),
            filter: None,
            best_effort_filter: None,
            projection: None,
            advanced_extension: None,
            read_type: Some(ReadType::VirtualTable(VirtualTable {
                values,
                expressions: vec![],
            })),
        }))),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use substrait::proto::expression::literal::LiteralType;
    use substrait::proto::expression::Literal;

    #[test]
    fn test_encode_table_function_metadata_basic() {
        // Test encoding with simple arguments
        let metadata = TableFunctionCallMetadata {
            function_name: "generate_series".to_string(),
            arguments: vec![
                Expression {
                    rex_type: Some(substrait::proto::expression::RexType::Literal(
                        Literal {
                            nullable: false,
                            type_variation_reference: 0,
                            literal_type: Some(LiteralType::I64(1)),
                        },
                    )),
                },
                Expression {
                    rex_type: Some(substrait::proto::expression::RexType::Literal(
                        Literal {
                            nullable: false,
                            type_variation_reference: 0,
                            literal_type: Some(LiteralType::I64(10)),
                        },
                    )),
                },
            ],
        };

        let result = encode_table_function_metadata(&metadata).unwrap();

        assert_eq!(result.type_url, "datafusion.io/TableFunctionCall");
        assert!(!result.value.is_empty());

        // Verify we can decode what we encoded
        let data = &result.value;
        let null_pos = data.iter().position(|&b| b == 0).unwrap();
        let decoded_name = String::from_utf8(data[..null_pos].to_vec()).unwrap();
        assert_eq!(decoded_name, "generate_series");
    }

    #[test]
    fn test_encode_table_function_metadata_zero_args() {
        // Test encoding with no arguments
        let metadata = TableFunctionCallMetadata {
            function_name: "some_function".to_string(),
            arguments: vec![],
        };

        let result = encode_table_function_metadata(&metadata).unwrap();
        assert_eq!(result.type_url, "datafusion.io/TableFunctionCall");

        // Verify argument count is 0
        let data = &result.value;
        let null_pos = data.iter().position(|&b| b == 0).unwrap();
        let arg_count_pos = null_pos + 1;
        let arg_count = u32::from_le_bytes([
            data[arg_count_pos],
            data[arg_count_pos + 1],
            data[arg_count_pos + 2],
            data[arg_count_pos + 3],
        ]);
        assert_eq!(arg_count, 0);
    }

    #[test]
    fn test_encode_table_function_metadata_null_in_name() {
        // Test that function names with null bytes are rejected
        let metadata = TableFunctionCallMetadata {
            function_name: "bad\0name".to_string(),
            arguments: vec![],
        };

        let result = encode_table_function_metadata(&metadata);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cannot contain null bytes"));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Test that encoding and decoding preserves data
        let metadata = TableFunctionCallMetadata {
            function_name: "test_func".to_string(),
            arguments: vec![Expression {
                rex_type: Some(substrait::proto::expression::RexType::Literal(Literal {
                    nullable: false,
                    type_variation_reference: 0,
                    literal_type: Some(LiteralType::I64(42)),
                })),
            }],
        };

        let encoded = encode_table_function_metadata(&metadata).unwrap();

        // Verify the structure can be decoded
        let data = &encoded.value;

        // Check function name
        let null_pos = data.iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8(data[..null_pos].to_vec()).unwrap();
        assert_eq!(name, "test_func");

        // Check argument count
        let mut pos = null_pos + 1;
        let arg_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        assert_eq!(arg_count, 1);
        pos += 4;

        // Check first argument length and decode
        let arg_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
        pos += 4;
        let arg_data = &data[pos..pos + arg_len];
        let decoded_expr = Expression::decode(arg_data).unwrap();

        assert_eq!(decoded_expr.rex_type, metadata.arguments[0].rex_type);
    }
}
