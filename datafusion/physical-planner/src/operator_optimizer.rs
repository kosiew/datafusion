// ... existing code ...

fn optimize_join(
    &self,
    join: &Join,
    optimizer_config: &dyn OptimizerConfig,
    config: &mut PlannerConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    // ... existing code ...

    // Create the appropriate join executor with limit awareness if provided
    let join_exec = match join_type {
        JoinType::Inner => {
            // ... existing code for inner join ...
            let mut join_exec = HashJoinExec::try_new(
                // ... existing parameters ...
            )?;

            // Add limit information if available
            if let Some(limit) = join.limit {
                join_exec = join_exec.with_limit(limit, join.skip);
            }

            join_exec.build()?
        }
        // Do the same for other join types...
        _ => {
            // ... existing code for other join types ...
        }
    };

    // ... existing code ...
}

// ... existing code ...
