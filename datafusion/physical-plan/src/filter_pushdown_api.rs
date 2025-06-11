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

    /// Consume self and return the inner expression
    pub fn into_inner(self) -> Arc<dyn PhysicalExpr> {
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

    /// Create a new collection marking all predicates as supported
    pub fn all_supported(preds: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self(
            preds
                .into_iter()
                .map(PredicateWithSupport::Supported)
                .collect(),
        )
    }

    /// Create a new collection marking all predicates as unsupported
    pub fn all_unsupported(preds: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self(
            preds
                .into_iter()
                .map(PredicateWithSupport::Unsupported)
                .collect(),
        )
    }

    /// Create a new collection using the provided callback to determine support
    pub fn new_with_supported_check(
        preds: Vec<Arc<dyn PhysicalExpr>>,
        check: impl Fn(&Arc<dyn PhysicalExpr>) -> bool,
    ) -> Self {
        Self(
            preds
                .into_iter()
                .map(|p| {
                    if check(&p) {
                        PredicateWithSupport::Supported(p)
                    } else {
                        PredicateWithSupport::Unsupported(p)
                    }
                })
                .collect(),
        )
    }

    /// Add a predicate to the collection returning a new instance.
    pub fn with_predicate(mut self, pred: PredicateWithSupport) -> Self {
        self.0.push(pred);
        self
    }

    /// Transform all predicates to supported
    pub fn make_supported(self) -> Self {
        Self(
            self.0
                .into_iter()
                .map(|p| match p {
                    PredicateWithSupport::Supported(expr)
                    | PredicateWithSupport::Unsupported(expr) => {
                        PredicateWithSupport::Supported(expr)
                    }
                })
                .collect(),
        )
    }

    /// Transform all predicates to unsupported
    pub fn make_unsupported(self) -> Self {
        Self(
            self.0
                .into_iter()
                .map(|p| match p {
                    PredicateWithSupport::Supported(expr)
                    | PredicateWithSupport::Unsupported(expr) => {
                        PredicateWithSupport::Unsupported(expr)
                    }
                })
                .collect(),
        )
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

    /// Collect all predicates, discarding support information
    pub fn collect_all(self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.0.into_iter().map(|p| p.into_inner()).collect()
    }

    /// Consume self and return inner representation
    pub fn into_inner(self) -> Vec<PredicateWithSupport> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_all_supported(&self) -> bool {
        self.0
            .iter()
            .all(|p| matches!(p, PredicateWithSupport::Supported(_)))
    }

    pub fn is_all_unsupported(&self) -> bool {
        self.0
            .iter()
            .all(|p| matches!(p, PredicateWithSupport::Unsupported(_)))
    }

    /// Returns an iterator over all predicates.
    pub fn iter(&self) -> impl Iterator<Item = &PredicateWithSupport> {
        self.0.iter()
    }
}

impl IntoIterator for Predicates {
    type Item = PredicateWithSupport;
    type IntoIter = std::vec::IntoIter<PredicateWithSupport>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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
