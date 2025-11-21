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

use crate::logical_plan::consumer::from_substrait_literal;
use crate::logical_plan::consumer::from_substrait_named_struct;
use crate::logical_plan::consumer::utils::ensure_schema_compatibility;
use crate::logical_plan::consumer::SubstraitConsumer;
use datafusion::common::{
    not_impl_err, plan_err, substrait_datafusion_err, substrait_err, DFSchema,
    DFSchemaRef, TableReference,
};
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::utils::split_conjunction_owned;
use datafusion::logical_expr::{
    EmptyRelation, Expr, LogicalPlan, LogicalPlanBuilder, Values,
};
use pbjson_types::Any as ProtoAny;
use std::sync::Arc;
use substrait::proto::expression::MaskExpression;
use substrait::proto::read_rel::local_files::file_or_files::PathType::UriFile;
use substrait::proto::read_rel::ReadType;
use substrait::proto::{Expression, ReadRel};
use url::Url;

#[allow(deprecated)]
pub async fn from_read_rel(
    consumer: &impl SubstraitConsumer,
    read: &ReadRel,
) -> datafusion::common::Result<LogicalPlan> {
    async fn read_with_schema(
        consumer: &impl SubstraitConsumer,
        table_ref: TableReference,
        schema: DFSchema,
        projection: &Option<MaskExpression>,
        filter: &Option<Box<Expression>>,
    ) -> datafusion::common::Result<LogicalPlan> {
        let schema = schema.replace_qualifier(table_ref.clone());

        let filters = if let Some(f) = filter {
            let filter_expr = consumer.consume_expression(f, &schema).await?;
            split_conjunction_owned(filter_expr)
        } else {
            vec![]
        };

        let plan = {
            let provider = match consumer.resolve_table_ref(&table_ref).await? {
                Some(ref provider) => Arc::clone(provider),
                _ => return plan_err!("No table named '{table_ref}'"),
            };

            LogicalPlanBuilder::scan_with_filters(
                table_ref,
                provider_as_source(Arc::clone(&provider)),
                None,
                filters,
            )?
            .build()?
        };

        ensure_schema_compatibility(plan.schema(), schema.clone())?;

        let schema = apply_masking(schema, projection)?;

        apply_projection(plan, schema)
    }

    let named_struct = read.base_schema.as_ref().ok_or_else(|| {
        substrait_datafusion_err!("No base schema provided for Read Relation")
    })?;

    let substrait_schema = from_substrait_named_struct(consumer, named_struct)?;

    match &read.read_type {
        Some(ReadType::NamedTable(nt)) => {
            // Check if this is a table function call by inspecting the advanced_extension
            if let Some(extension) = &nt.advanced_extension {
                if let Some(enhancement) = &extension.enhancement {
                    if enhancement.type_url == "datafusion.io/TableFunctionCall" {
                        // Decode table function metadata
                        let (function_name, arg_expressions) =
                            decode_table_function_metadata(enhancement)?;

                        // Convert Substrait expressions back to DataFusion Expr
                        let empty_schema = DFSchema::empty();
                        let mut args = Vec::new();
                        for arg_expr in arg_expressions {
                            let expr = consumer
                                .consume_expression(&arg_expr, &empty_schema)
                                .await?;
                            args.push(expr);
                        }

                        // Recreate the table function
                        let provider = match consumer
                            .resolve_table_function(&function_name, args.clone())
                            .await?
                        {
                            Some(provider) => provider,
                            None => {
                                return plan_err!(
                                "Table function '{function_name}' not found in function registry"
                            );
                            }
                        };

                        let table_reference = match nt.names.len() {
                            0 => TableReference::Bare {
                                table: format!("{function_name}()").into(),
                            },
                            1 => TableReference::Bare {
                                table: nt.names[0].clone().into(),
                            },
                            2 => TableReference::Partial {
                                schema: nt.names[0].clone().into(),
                                table: nt.names[1].clone().into(),
                            },
                            _ => TableReference::Full {
                                catalog: nt.names[0].clone().into(),
                                schema: nt.names[1].clone().into(),
                                table: nt.names[2].clone().into(),
                            },
                        };

                        // Use the provider we already have instead of looking it up again
                        let schema =
                            substrait_schema.replace_qualifier(table_reference.clone());

                        let filters = if let Some(f) = &read.filter {
                            let filter_expr =
                                consumer.consume_expression(f, &schema).await?;
                            split_conjunction_owned(filter_expr)
                        } else {
                            vec![]
                        };

                        let mut plan = LogicalPlanBuilder::scan_with_table_function_call(
                            table_reference,
                            provider_as_source(provider),
                            None,
                            function_name,
                            args,
                        )?;

                        // Apply filters if present
                        for f in filters {
                            plan = plan.filter(f)?;
                        }

                        let plan = plan.build()?;

                        ensure_schema_compatibility(plan.schema(), schema.clone())?;

                        let schema = apply_masking(schema, &read.projection)?;

                        return apply_projection(plan, schema);
                    }
                }
            } // Regular table scan
            let table_reference = match nt.names.len() {
                0 => {
                    return plan_err!("No table name found in NamedTable");
                }
                1 => TableReference::Bare {
                    table: nt.names[0].clone().into(),
                },
                2 => TableReference::Partial {
                    schema: nt.names[0].clone().into(),
                    table: nt.names[1].clone().into(),
                },
                _ => TableReference::Full {
                    catalog: nt.names[0].clone().into(),
                    schema: nt.names[1].clone().into(),
                    table: nt.names[2].clone().into(),
                },
            };

            read_with_schema(
                consumer,
                table_reference,
                substrait_schema,
                &read.projection,
                &read.filter,
            )
            .await
        }
        Some(ReadType::VirtualTable(vt)) => {
            if vt.values.is_empty() && vt.expressions.is_empty() {
                return Ok(LogicalPlan::EmptyRelation(EmptyRelation {
                    produce_one_row: false,
                    schema: DFSchemaRef::new(substrait_schema),
                }));
            }

            let values = if !vt.expressions.is_empty() {
                let mut exprs = vec![];
                for row in &vt.expressions {
                    let mut name_idx = 0;
                    let mut row_exprs = vec![];
                    for expression in &row.fields {
                        name_idx += 1;
                        let expr = consumer
                            .consume_expression(expression, &DFSchema::empty())
                            .await?;
                        row_exprs.push(expr);
                    }
                    if name_idx != named_struct.names.len() {
                        return substrait_err!(
                                "Names list must match exactly to nested schema, but found {} uses for {} names",
                                name_idx,
                                named_struct.names.len()
                            );
                    }
                    exprs.push(row_exprs);
                }
                exprs
            } else {
                vt
                .values
                .iter()
                .map(|row| {
                    let mut name_idx = 0;
                    let lits = row
                        .fields
                        .iter()
                        .map(|lit| {
                            name_idx += 1; // top-level names are provided through schema
                            Ok(Expr::Literal(from_substrait_literal(
                                consumer,
                                lit,
                                &named_struct.names,
                                &mut name_idx,
                            )?, None))
                        })
                        .collect::<datafusion::common::Result<_>>()?;
                    if name_idx != named_struct.names.len() {
                        return substrait_err!(
                                "Names list must match exactly to nested schema, but found {} uses for {} names",
                                name_idx,
                                named_struct.names.len()
                            );
                    }
                    Ok(lits)
                })
                .collect::<datafusion::common::Result<_>>()?
            };

            Ok(LogicalPlan::Values(Values {
                schema: DFSchemaRef::new(substrait_schema),
                values,
            }))
        }
        Some(ReadType::LocalFiles(lf)) => {
            fn extract_filename(name: &str) -> Option<String> {
                let corrected_url =
                    if name.starts_with("file://") && !name.starts_with("file:///") {
                        name.replacen("file://", "file:///", 1)
                    } else {
                        name.to_string()
                    };

                Url::parse(&corrected_url).ok().and_then(|url| {
                    let path = url.path();
                    std::path::Path::new(path)
                        .file_name()
                        .map(|filename| filename.to_string_lossy().to_string())
                })
            }

            // we could use the file name to check the original table provider
            // TODO: currently does not support multiple local files
            let filename: Option<String> =
                lf.items.first().and_then(|x| match x.path_type.as_ref() {
                    Some(UriFile(name)) => extract_filename(name),
                    _ => None,
                });

            if lf.items.len() > 1 || filename.is_none() {
                return not_impl_err!("Only single file reads are supported");
            }
            let name = filename.unwrap();
            // directly use unwrap here since we could determine it is a valid one
            let table_reference = TableReference::Bare { table: name.into() };

            read_with_schema(
                consumer,
                table_reference,
                substrait_schema,
                &read.projection,
                &read.filter,
            )
            .await
        }
        _ => {
            not_impl_err!("Unsupported Readtype: {:?}", read.read_type)
        }
    }
}

pub fn apply_masking(
    schema: DFSchema,
    mask_expression: &::core::option::Option<MaskExpression>,
) -> datafusion::common::Result<DFSchema> {
    match mask_expression {
        Some(MaskExpression { select, .. }) => match &select.as_ref() {
            Some(projection) => {
                let column_indices: Vec<usize> = projection
                    .struct_items
                    .iter()
                    .map(|item| item.field as usize)
                    .collect();

                let fields = column_indices
                    .iter()
                    .map(|i| schema.qualified_field(*i))
                    .map(|(qualifier, field)| {
                        (qualifier.cloned(), Arc::new(field.clone()))
                    })
                    .collect();

                Ok(DFSchema::new_with_metadata(
                    fields,
                    schema.metadata().clone(),
                )?)
            }
            None => Ok(schema),
        },
        None => Ok(schema),
    }
}

/// This function returns a DataFrame with fields adjusted if necessary in the event that the
/// Substrait schema is a subset of the DataFusion schema.
fn apply_projection(
    plan: LogicalPlan,
    substrait_schema: DFSchema,
) -> datafusion::common::Result<LogicalPlan> {
    let df_schema = plan.schema();

    if df_schema.logically_equivalent_names_and_types(&substrait_schema) {
        return Ok(plan);
    }

    let df_schema = df_schema.to_owned();

    match plan {
        LogicalPlan::TableScan(mut scan) => {
            let column_indices: Vec<usize> = substrait_schema
                .strip_qualifiers()
                .fields()
                .iter()
                .map(|substrait_field| {
                    Ok(df_schema
                        .index_of_column_by_name(None, substrait_field.name().as_str())
                        .unwrap())
                })
                .collect::<datafusion::common::Result<_>>()?;

            let fields = column_indices
                .iter()
                .map(|i| df_schema.qualified_field(*i))
                .map(|(qualifier, field)| (qualifier.cloned(), Arc::new(field.clone())))
                .collect();

            scan.projected_schema = DFSchemaRef::new(DFSchema::new_with_metadata(
                fields,
                df_schema.metadata().clone(),
            )?);
            scan.projection = Some(column_indices);

            Ok(LogicalPlan::TableScan(scan))
        }
        _ => plan_err!("DataFrame passed to apply_projection must be a TableScan"),
    }
}

/// Decode table function metadata from a Protobuf Any message
fn decode_table_function_metadata(
    extension: &ProtoAny,
) -> datafusion::common::Result<(String, Vec<Expression>)> {
    use prost::Message;

    let data = &extension.value;

    // Find the null terminator for the function name
    let null_pos = data.iter().position(|&b| b == 0).ok_or_else(|| {
        datafusion::common::plan_datafusion_err!(
            "Invalid table function metadata: missing null terminator"
        )
    })?;

    let function_name = String::from_utf8(data[..null_pos].to_vec()).map_err(|e| {
        datafusion::common::plan_datafusion_err!(
            "Invalid table function name encoding: {e}"
        )
    })?;

    let mut pos = null_pos + 1;

    // Read argument count
    if data.len() < pos + 4 {
        return plan_err!("Invalid table function metadata: missing argument count");
    }
    let arg_count =
        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    pos += 4;

    let mut arguments = Vec::new();
    for _ in 0..arg_count {
        // Read argument length
        if data.len() < pos + 4 {
            return plan_err!("Invalid table function metadata: missing argument length");
        }
        let arg_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
        pos += 4;

        // Read argument data
        if data.len() < pos + arg_len {
            return plan_err!("Invalid table function metadata: argument data truncated");
        }
        let arg_data = &data[pos..pos + arg_len];
        let expr = Expression::decode(arg_data).map_err(|e| {
            datafusion::common::plan_datafusion_err!(
                "Failed to decode table function argument: {e}"
            )
        })?;
        arguments.push(expr);
        pos += arg_len;
    }

    Ok((function_name, arguments))
}
