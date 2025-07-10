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

//! [`RequiredIndices`] helper for OptimizeProjection

use crate::optimize_projections::outer_columns;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Column, DFSchemaRef, Result};
use datafusion_expr::{Expr, LogicalPlan};
use log::debug;

/// Represents columns in a schema which are required (used) by a plan node
///
/// Also carries a flag indicating if putting a projection above children is
/// beneficial for the parent. For example `LogicalPlan::Filter` benefits from
/// small tables. Hence for filter child this flag would be `true`. Defaults to
/// `false`
///
/// # Invariant
///
/// Indices are always in order and without duplicates. For example, if these
/// indices were added `[3, 2, 4, 3, 6, 1]`,  the instance would be represented
/// by  `[1, 2, 3, 4, 6]`.
#[derive(Debug, Clone, Default)]
pub(super) struct RequiredIndices {
    /// The indices of the required columns in the
    indices: Vec<usize>,
    /// If putting a projection above children is beneficial for the parent.
    /// Defaults to false.
    projection_beneficial: bool,
}

impl RequiredIndices {
    /// Create a new, empty instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new instance that requires all columns from the specified plan
    pub fn new_for_all_exprs(plan: &LogicalPlan) -> Self {
        Self {
            indices: (0..plan.schema().fields().len()).collect(),
            projection_beneficial: false,
        }
    }

    /// Create a new instance with the specified indices as required
    pub fn new_from_indices(indices: Vec<usize>) -> Self {
        Self {
            indices,
            projection_beneficial: false,
        }
        .compact()
    }

    /// Convert the instance to its inner indices
    pub fn into_inner(self) -> Vec<usize> {
        self.indices
    }

    /// Set the projection beneficial flag
    pub fn with_projection_beneficial(mut self) -> Self {
        self.projection_beneficial = true;
        self
    }

    /// Return the value of projection beneficial flag
    pub fn projection_beneficial(&self) -> bool {
        self.projection_beneficial
    }

    /// Return a reference to the underlying indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Add required indices for all `exprs` used in plan
    pub fn with_plan_exprs(
        mut self,
        plan: &LogicalPlan,
        schema: &DFSchemaRef,
    ) -> Result<Self> {
        // Add indices of the child fields referred to by the expressions in the
        // parent
        plan.apply_expressions(|e| {
            self.add_expr(schema, e);
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(self.compact())
    }

    /// Adds the indices of the fields referred to by the given expression
    /// `expr` within the given schema (`input_schema`).
    ///
    /// Self is NOT compacted (and thus this method is not pub)
    ///
    /// # Parameters
    ///
    /// * `input_schema`: The input schema to analyze for index requirements.
    /// * `expr`: An expression for which we want to find necessary field indices.
    fn add_expr(&mut self, input_schema: &DFSchemaRef, expr: &Expr) {
        // TODO could remove these clones (and visit the expression directly)
        let mut cols = expr.column_refs();
        // Get outer-referenced (subquery) columns:
        outer_columns(expr, &mut cols);
        self.indices.reserve(cols.len());
        for col in cols {
            if let Some(idx) = input_schema.maybe_index_of_column(col) {
                self.indices.push(idx);
            }
        }
    }

    /// Similar to [`add_expr`] but ignores any qualifiers on the columns when
    /// locating them within `input_schema`.
    fn add_expr_ignore_qualifiers(&mut self, input_schema: &DFSchemaRef, expr: &Expr) {
        let mut cols = expr.column_refs();
        outer_columns(expr, &mut cols);
        self.indices.reserve(cols.len());
        for col in cols {
            // Debug: print column details
            debug!("Processing column: {col:?}");

            // For scalar subquery qualifiers, we need to preserve them as they represent
            // actual joined relations, not just table aliases
            if let Some(ref relation) = col.relation {
                if relation.to_string().starts_with("__scalar_sq_") {
                    debug!("Found scalar subquery column: {col:?}");
                    // Keep the scalar subquery qualifier
                    if let Some(idx) = input_schema.maybe_index_of_column(col) {
                        debug!("Found index {idx} for scalar subquery column: {col:?}");
                        self.indices.push(idx);
                    } else {
                        debug!("Could not find scalar subquery column {col:?} in schema");
                        debug!("Available fields in schema:");
                        for (i, field) in input_schema.fields().iter().enumerate() {
                            debug!("  [{i}] {field:?}");
                        }
                    }
                    continue;
                }
            }

            let unqualified = Column::new_unqualified(&col.name);
            if let Some(idx) = input_schema.maybe_index_of_column(&unqualified) {
                debug!("Found index {idx} for unqualified column: {}", col.name);
                self.indices.push(idx);
            } else {
                debug!("Could not find unqualified column: {}", col.name);
            }
        }
    }

    /// Adds the indices of the fields referred to by the given expressions
    /// `within the given schema.
    ///
    /// # Parameters
    ///
    /// * `input_schema`: The input schema to analyze for index requirements.
    /// * `exprs`: the expressions for which we want to find field indices.
    pub fn with_exprs<'a>(
        self,
        schema: &DFSchemaRef,
        exprs: impl IntoIterator<Item = &'a Expr>,
    ) -> Self {
        exprs
            .into_iter()
            .fold(self, |mut acc, expr| {
                acc.add_expr(schema, expr);
                acc
            })
            .compact()
    }

    /// Adds the indices of the fields referred to by `exprs` within
    /// `schema`, ignoring any qualifiers on the columns.
    pub fn with_exprs_ignore_qualifiers<'a>(
        self,
        schema: &DFSchemaRef,
        exprs: impl IntoIterator<Item = &'a Expr>,
    ) -> Self {
        exprs
            .into_iter()
            .fold(self, |mut acc, expr| {
                acc.add_expr_ignore_qualifiers(schema, expr);
                acc
            })
            .compact()
    }

    /// Adds all `indices` into this instance.
    pub fn append(mut self, indices: &[usize]) -> Self {
        self.indices.extend_from_slice(indices);
        self.compact()
    }

    /// Splits this instance into a tuple with two instances:
    /// * The first `n` indices
    /// * The remaining indices, adjusted down by n
    pub fn split_off(self, n: usize) -> (Self, Self) {
        let (l, r) = self.partition(|idx| idx < n);
        (l, r.map_indices(|idx| idx - n))
    }

    /// Partitions the indices in this instance into two groups based on the
    /// given predicate function `f`.
    fn partition<F>(&self, f: F) -> (Self, Self)
    where
        F: Fn(usize) -> bool,
    {
        let (l, r): (Vec<usize>, Vec<usize>) =
            self.indices.iter().partition(|&&idx| f(idx));
        let projection_beneficial = self.projection_beneficial;

        (
            Self {
                indices: l,
                projection_beneficial,
            },
            Self {
                indices: r,
                projection_beneficial,
            },
        )
    }

    /// Map the indices in this instance to a new set of indices based on the
    /// given function `f`, returning the mapped indices
    ///
    /// Not `pub` as it might not preserve the invariant of compacted indices
    fn map_indices<F>(mut self, f: F) -> Self
    where
        F: Fn(usize) -> usize,
    {
        self.indices.iter_mut().for_each(|idx| *idx = f(*idx));
        self
    }

    /// Apply the given function `f` to each index in this instance, returning
    /// the mapped indices
    pub fn into_mapped_indices<F>(self, f: F) -> Vec<usize>
    where
        F: Fn(usize) -> usize,
    {
        self.map_indices(f).into_inner()
    }

    /// Returns the `Expr`s from `exprs` that are at the indices in this instance
    pub fn get_at_indices(&self, exprs: &[Expr]) -> Vec<Expr> {
        self.indices.iter().map(|&idx| exprs[idx].clone()).collect()
    }

    /// Generates the required expressions (columns) that reside at `indices` of
    /// the given `input_schema`.
    pub fn get_required_exprs(&self, input_schema: &DFSchemaRef) -> Vec<Expr> {
        self.indices
            .iter()
            .map(|&idx| Expr::from(Column::from(input_schema.qualified_field(idx))))
            .collect()
    }

    /// Compacts the indices of this instance so they are sorted
    /// (ascending) and deduplicated.
    fn compact(mut self) -> Self {
        self.indices.sort_unstable();
        self.indices.dedup();
        self
    }
}
