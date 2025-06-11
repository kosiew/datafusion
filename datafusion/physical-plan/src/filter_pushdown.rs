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
pub enum PredicateSupport {
    /// The predicate can be pushed down.
    Supported(Arc<dyn PhysicalExpr>),
    /// The predicate must be evaluated locally.
    Unsupported(Arc<dyn PhysicalExpr>),
}

impl PredicateSupport {
    /// Returns a reference to the underlying expression.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        match self {
            PredicateSupport::Supported(expr) | PredicateSupport::Unsupported(expr) => {
                expr
            }
        }
    }

    /// Consume self and return the inner expression
    pub fn into_inner(self) -> Arc<dyn PhysicalExpr> {
        match self {
            PredicateSupport::Supported(expr) | PredicateSupport::Unsupported(expr) => {
                expr
            }
        }
    }
}

/// Collection of predicates with convenience helpers.
#[derive(Debug, Clone, Default)]
pub struct Predicates(Vec<PredicateSupport>);

impl Predicates {
    /// Create a new collection from the provided predicates.
    pub fn new(preds: Vec<PredicateSupport>) -> Self {
        Self(preds)
    }

    /// Create a new collection marking all predicates as supported
    pub fn all_supported(preds: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self(preds.into_iter().map(PredicateSupport::Supported).collect())
    }

    /// Create a new collection marking all predicates as unsupported
    pub fn all_unsupported(preds: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self(
            preds
                .into_iter()
                .map(PredicateSupport::Unsupported)
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
                        PredicateSupport::Supported(p)
                    } else {
                        PredicateSupport::Unsupported(p)
                    }
                })
                .collect(),
        )
    }

    /// Add a predicate to the collection returning a new instance.
    pub fn with_predicate(mut self, pred: PredicateSupport) -> Self {
        self.0.push(pred);
        self
    }

    /// Transform all predicates to supported
    pub fn make_supported(self) -> Self {
        Self(
            self.0
                .into_iter()
                .map(|p| match p {
                    PredicateSupport::Supported(expr)
                    | PredicateSupport::Unsupported(expr) => {
                        PredicateSupport::Supported(expr)
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
                    PredicateSupport::Supported(expr)
                    | PredicateSupport::Unsupported(expr) => {
                        PredicateSupport::Unsupported(expr)
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
                PredicateSupport::Supported(expr) => Some(Arc::clone(expr)),
                _ => None,
            })
            .collect()
    }

    /// Return all predicates marked as unsupported.
    pub fn collect_unsupported(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.0
            .iter()
            .filter_map(|p| match p {
                PredicateSupport::Unsupported(expr) => Some(Arc::clone(expr)),
                _ => None,
            })
            .collect()
    }

    /// Collect all predicates, discarding support information
    pub fn collect_all(self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.0.into_iter().map(|p| p.into_inner()).collect()
    }

    /// Consume self and return inner representation
    pub fn into_inner(self) -> Vec<PredicateSupport> {
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
            .all(|p| matches!(p, PredicateSupport::Supported(_)))
    }

    pub fn is_all_unsupported(&self) -> bool {
        self.0
            .iter()
            .all(|p| matches!(p, PredicateSupport::Unsupported(_)))
    }

    /// Returns an iterator over all predicates.
    pub fn iter(&self) -> impl Iterator<Item = &PredicateSupport> {
        self.0.iter()
    }
}

impl IntoIterator for Predicates {
    type Item = PredicateSupport;
    type IntoIter = std::vec::IntoIter<PredicateSupport>;

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

/// The result of pushing down filters into a child node.
/// This is the result provided to nodes in [`ExecutionPlan::handle_child_pushdown_result`].
/// Nodes process this result and convert it into a [`FilterPushdownPropagation`]
/// that is returned to their parent.
///
/// [`ExecutionPlan::handle_child_pushdown_result`]: crate::ExecutionPlan::handle_child_pushdown_result
#[derive(Debug, Clone)]
pub struct ChildPushdownResult {
    /// The combined result of pushing down each parent filter into each child.
    /// For example, given the fitlers `[a, b]` and children `[1, 2, 3]` the matrix of responses:
    ///
    // | filter | child 1     | child 2   | child 3   | result      |
    // |--------|-------------|-----------|-----------|-------------|
    // | a      | Supported   | Supported | Supported | Supported   |
    // | b      | Unsupported | Supported | Supported | Unsupported |
    ///
    /// That is: if any child marks a filter as unsupported or if the filter was not pushed
    /// down into any child then the result is unsupported.
    /// If at least one children and all children that received the filter mark it as supported
    /// then the result is supported.
    pub parent_filters: Predicates,
    /// The result of pushing down each filter this node provided into each of it's children.
    /// This is not combined with the parent filters so that nodes can treat each child independently.
    pub self_filters: Vec<Predicates>,
}

/// The result of pushing down filters into a node that it returns to its parent.
/// This is what nodes return from [`ExecutionPlan::handle_child_pushdown_result`] to communicate
/// to the optimizer:
///
/// 1. What to do with any parent filters that were not completely handled by the children.
/// 2. If the node needs to be replaced in the execution plan with a new node or not.
///
/// [`ExecutionPlan::handle_child_pushdown_result`]: crate::ExecutionPlan::handle_child_pushdown_result
#[derive(Debug, Clone)]
pub struct FilterPushdownPropagation<T> {
    pub filters: Predicates,
    pub updated_node: Option<T>,
}

impl<T> FilterPushdownPropagation<T> {
    /// Create a new [`FilterPushdownPropagation`] that tells the parent node
    /// that echoes back up to the parent the result of pushing down the filters
    /// into the children.
    pub fn transparent(child_pushdown_result: ChildPushdownResult) -> Self {
        Self {
            filters: child_pushdown_result.parent_filters,
            updated_node: None,
        }
    }

    /// Create a new [`FilterPushdownPropagation`] that tells the parent node
    /// that none of the parent filters were not pushed down.
    pub fn unsupported(parent_filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        let unsupported = Predicates::all_unsupported(parent_filters);
        Self {
            filters: unsupported,
            updated_node: None,
        }
    }

    /// Create a new [`FilterPushdownPropagation`] with the specified filter support.
    pub fn with_filters(filters: Predicates) -> Self {
        Self {
            filters,
            updated_node: None,
        }
    }

    /// Bind an updated node to the [`FilterPushdownPropagation`].
    pub fn with_updated_node(mut self, updated_node: T) -> Self {
        self.updated_node = Some(updated_node);
        self
    }
}

#[derive(Debug, Clone)]
struct ChildFilterDescription {
    /// Description of which parent filters can be pushed down into this node.
    /// Since we need to transmit filter pushdown results back to this node's parent
    /// we need to track each parent filter for each child, even those that are unsupported / won't be pushed down.
    /// We do this using a [`PredicateSupport`] which simplifies manipulating supported/unsupported filters.
    parent_filters: Predicates,
    /// Description of which filters this node is pushing down to its children.
    /// Since this is not transmitted back to the parents we can have variable sized inner arrays
    /// instead of having to track supported/unsupported.
    self_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl ChildFilterDescription {
    fn new() -> Self {
        Self {
            parent_filters: Predicates::new(vec![]),
            self_filters: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilterDescription {
    /// A filter description for each child.
    /// This includes which parent filters and which self filters (from the node in question)
    /// will get pushed down to each child.
    child_filter_descriptions: Vec<ChildFilterDescription>,
}

impl FilterDescription {
    pub fn new_with_child_count(num_children: usize) -> Self {
        Self {
            child_filter_descriptions: vec![ChildFilterDescription::new(); num_children],
        }
    }

    pub fn parent_filters(&self) -> Vec<Predicates> {
        self.child_filter_descriptions
            .iter()
            .map(|d| &d.parent_filters)
            .cloned()
            .collect()
    }

    pub fn self_filters(&self) -> Vec<Vec<Arc<dyn PhysicalExpr>>> {
        self.child_filter_descriptions
            .iter()
            .map(|d| &d.self_filters)
            .cloned()
            .collect()
    }

    /// Mark all parent filters as supported for all children.
    /// This is the case if the node allows filters to be pushed down through it
    /// without any modification.
    /// This broadcasts the parent filters to all children.
    /// If handling of parent filters is different for each child then you should set the
    /// field direclty.
    /// For example, nodes like [`RepartitionExec`] that let filters pass through it transparently
    /// use this to mark all parent filters as supported.
    ///
    /// [`RepartitionExec`]: crate::repartition::RepartitionExec
    pub fn all_parent_filters_supported(
        mut self,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        let supported = Predicates::all_supported(parent_filters);
        for child in &mut self.child_filter_descriptions {
            child.parent_filters = supported.clone();
        }
        self
    }

    /// Mark all parent filters as unsupported for all children.
    /// This is the case if the node does not allow filters to be pushed down through it.
    /// This broadcasts the parent filters to all children.
    /// If handling of parent filters is different for each child then you should set the
    /// field direclty.
    /// For example, the default implementation of filter pushdwon in [`ExecutionPlan`]
    /// assumes that filters cannot be pushed down to children.
    ///
    /// [`ExecutionPlan`]: crate::ExecutionPlan
    pub fn all_parent_filters_unsupported(
        mut self,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        let unsupported = Predicates::all_unsupported(parent_filters);
        for child in &mut self.child_filter_descriptions {
            child.parent_filters = unsupported.clone();
        }
        self
    }

    /// Add a filter generated / owned by the current node to be pushed down to all children.
    /// This assumes that there is a single filter that that gets pushed down to all children
    /// equally.
    /// If there are multiple filters or pushdown to children is not homogeneous then
    /// you should set the field directly.
    /// For example:
    /// - `TopK` uses this to push down a single filter to all children, it can use this method.
    /// - `HashJoinExec` pushes down a filter only to the probe side, it cannot use this method.
    pub fn with_self_filter(mut self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        for child in &mut self.child_filter_descriptions {
            child.self_filters = vec![Arc::clone(&predicate)];
        }
        self
    }

    pub fn with_self_filters_for_children(
        mut self,
        filters: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    ) -> Self {
        for (child, filters) in self.child_filter_descriptions.iter_mut().zip(filters) {
            child.self_filters = filters;
        }
        self
    }
}
