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

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_expr::{
    logical_plan::{Join, Limit, LogicalPlan, JoinType},
    optimizer::{OptimizerConfig, OptimizerRule},
};
use log::debug;

/// Optimization rule that pushes LIMIT down into JOIN operations
#[derive(Default)]
pub struct PushLimitIntoJoin {}

impl PushLimitIntoJoin {
    /// Create a new PushLimitIntoJoin optimizer rule
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for PushLimitIntoJoin {
    fn name(&self) -> &str {
        "push_limit_into_join"
    }

    fn optimize(
        &self,
        plan: &LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Limit(Limit {
                n,
                skip,
                input,
            }) => {
                // Check if the input is a Join operation
                match input.as_ref() {
                    LogicalPlan::Join(join) => {
                        // Limit push-down is most beneficial for non-equi joins that could produce large datasets
                        let can_push_limit = match join.join_type {
                            // Inner and left/right/full outer joins can benefit from this optimization
                            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => true,
                            // Semi and anti joins are already optimized for early exit
                            _ => false,
                        };

                        if can_push_limit {
                            debug!("Pushing LIMIT into JOIN: LIMIT {}, SKIP {}", n, skip);
                            
                            // Optimize the input plan first
                            let optimized_left = self.optimize(join.left.as_ref(), config)?;
                            let optimized_right = self.optimize(join.right.as_ref(), config)?;
                            
                            // Create a join with limit information
                            let limit_aware_join = LogicalPlan::Join(Join {
                                left: Arc::new(optimized_left),
                                right: Arc::new(optimized_right),
                                on: join.on.clone(),
                                filter: join.filter.clone(),
                                join_type: join.join_type,
                                join_constraint: join.join_constraint,
                                null_equals_null: join.null_equals_null,
                                schema: join.schema.clone(),
                                limit: Some(*n),
                                skip: *skip,
                            });
                            
                            return Ok(limit_aware_join);
                        }
                    }
                    _ => {}
                }
                
                // Apply the rule to the input plan
                let new_input = self.optimize(input, config)?;
                if new_input.same_as(input) {
                    Ok(plan.clone())
                } else {
                    Ok(LogicalPlan::Limit(Limit {
                        n: *n,
                        skip: *skip,
                        input: Arc::new(new_input),
                    }))
                }
            }
            _ => {
                // Apply the optimization to all inputs of the plan node
                plan.transform_inputs(|plan| self.optimize(plan, config))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::{
        col, lit,
        logical_plan::{table_scan, JoinConstraint, JoinType},
    };
    use datafusion_common::DFSchema;

    #[test]
    fn test_push_limit_into_join() -> Result<()> {
        // Create a simple test schema
        let schema = DFSchema::empty();
        
        // Create a test plan: LIMIT 10 (t1 JOIN t2 ON t1.id <= t2.id)
        let t1 = table_scan(Some("t1"), &schema, None, None, None)?;
        let t2 = table_scan(Some("t2"), &schema, None, None, None)?;
        
        let join = LogicalPlan::Join(Join {
            left: Arc::new(t1),
            right: Arc::new(t2),
            on: vec![(col("t1.id"), col("t2.id"))],
            filter: Some(col("t1.id").lt_eq(col("t2.id"))),
            join_type: JoinType::Inner,
            join_constraint: JoinConstraint::On,
            null_equals_null: false,
            schema: schema.clone(),
            limit: None,
            skip: 0,
        });
        
        let plan = LogicalPlan::Limit(Limit {
            n: 10,
            skip: 0,
            input: Arc::new(join),
        });
        
        // Apply the optimizer rule
        let rule = PushLimitIntoJoin::new();
        let optimized_plan = rule.optimize(&plan, &Default::default())?;
        
        // Verify the limit was pushed down
        match optimized_plan {
            LogicalPlan::Join(join) => {
                assert_eq!(join.limit, Some(10));
                assert_eq!(join.skip, 0);
                Ok(())
            }
            _ => Err(datafusion_common::DataFusionError::Internal(
                "Expected Join plan after optimization".to_string(),
            )),
        }
    }
}
