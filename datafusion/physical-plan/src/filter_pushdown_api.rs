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

//! Simplified filter pushdown API.
//!
//! This module contains an experimental redesign of the
//! filter pushdown API aimed at reducing the number of
//! abstractions and making the flow easier to understand.
//! It is not yet hooked into the rest of the execution
//! engine but provides the core data structures that new
//! implementations can build on.

use std::sync::Arc;

use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

/// A predicate with its pushdown support status.
#[derive(Debug, Clone)]
pub enum PredicateWithSupport {
    /// The predicate can be pushed down.
    Supported(Arc<dyn PhysicalExpr>),
    /// The predicate must be evaluated locally.
    Unsupported(Arc<dyn PhysicalExpr>),
}

impl PredicateWithSupport {
    /// Returns a reference to the underlying expression.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        match self {
            PredicateWithSupport::Supported(expr)
            | PredicateWithSupport::Unsupported(expr) => expr,
        }
    }
}

/// Collection of predicates with convenience helpers.
#[derive(Debug, Clone, Default)]
pub struct Predicates(Vec<PredicateWithSupport>);

impl Predicates {
    /// Create a new collection from the provided predicates.
    pub fn new(preds: Vec<PredicateWithSupport>) -> Self {
        Self(preds)
    }

    /// Add a predicate to the collection returning a new instance.
    pub fn with_predicate(mut self, pred: PredicateWithSupport) -> Self {
        self.0.push(pred);
        self
    }

    /// Return all predicates marked as supported.
    pub fn collect_supported(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.0
            .iter()
            .filter_map(|p| match p {
                PredicateWithSupport::Supported(expr) => Some(Arc::clone(expr)),
                _ => None,
            })
            .collect()
    }

    /// Return all predicates marked as unsupported.
    pub fn collect_unsupported(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.0
            .iter()
            .filter_map(|p| match p {
                PredicateWithSupport::Unsupported(expr) => Some(Arc::clone(expr)),
                _ => None,
            })
            .collect()
    }

    /// Returns an iterator over all predicates.
    pub fn iter(&self) -> impl Iterator<Item = &PredicateWithSupport> {
        self.0.iter()
    }
}

/// Result of attempting to push predicates through a node.
#[derive(Debug, Clone)]
pub struct FilterPushdownResult<T> {
    /// Predicates that were pushed into the child node.
    pub pushed_predicates: Vec<Arc<dyn PhysicalExpr>>,
    /// Predicates that must remain on the current node.
    pub retained_predicates: Vec<Arc<dyn PhysicalExpr>>,
    /// Optional updated execution plan node.
    pub updated_plan: Option<T>,
}

impl<T> FilterPushdownResult<T> {
    /// Create a new result with the provided pushed and retained predicates.
    pub fn new(
        pushed_predicates: Vec<Arc<dyn PhysicalExpr>>,
        retained_predicates: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            pushed_predicates,
            retained_predicates,
            updated_plan: None,
        }
    }

    /// Attach an updated plan node.
    pub fn with_updated_plan(mut self, plan: T) -> Self {
        self.updated_plan = Some(plan);
        self
    }
}
