# Column Generation and Pricing

## Overview

Column generation is a technique for solving large-scale linear programs where it's impractical to enumerate all variables explicitly. It generates variables (columns) on-demand by solving a pricing subproblem.

## Key Concepts

### Master Problem
- Contains a subset of variables (columns)
- Solved to optimality with current columns
- Provides dual values (shadow prices)

### Pricing Subproblem
- Uses dual values from master problem
- Identifies new columns with negative reduced cost
- Adds improving columns to master problem

### Reduced Cost
For a variable x_j with cost c_j and constraint coefficients a_ij:
```
reduced_cost_j = c_j - sum(dual_i * a_ij)
```

If reduced cost < 0, adding the column improves the objective.

## Algorithm

```
1. Initialize master problem with basic feasible solution

2. Repeat:
   a. Solve master problem (LP)
   b. Get dual values (π) from optimal solution
   
   c. Solve pricing subproblem:
      - Find column with most negative reduced cost
      - Use dual values π in objective
   
   d. If no column with negative reduced cost:
      - STOP (optimal solution found)
   
   e. Add new column(s) to master problem

3. Return optimal solution
```

## Example: Cutting Stock Problem

**Problem**: Cut large rolls into smaller pieces to minimize waste.

### Master Problem
```
Minimize: sum of patterns used
Subject to: demand for each size is met
```

### Pricing Subproblem (Knapsack)
```python
def pricing_subproblem(roll_width, sizes, demands, dual_values):
    """
    Find cutting pattern with most negative reduced cost
    
    roll_width: width of large roll
    sizes: list of piece sizes
    demands: demand for each size
    dual_values: from master problem
    """
    n = len(sizes)
    
    # Knapsack: maximize sum(dual_i * x_i)
    # subject to: sum(size_i * x_i) <= roll_width
    
    # Dynamic programming
    dp = [0] * (roll_width + 1)
    pattern = [[] for _ in range(roll_width + 1)]
    
    for width in range(1, roll_width + 1):
        for i, size in enumerate(sizes):
            if size <= width:
                value = dual_values[i] + dp[width - size]
                if value > dp[width]:
                    dp[width] = value
                    pattern[width] = pattern[width - size] + [i]
    
    # Reduced cost = 1 - max_dual_value
    reduced_cost = 1.0 - dp[roll_width]
    
    # Convert pattern to column
    column = [0] * n
    for piece_idx in pattern[roll_width]:
        column[piece_idx] += 1
    
    return column, reduced_cost


def cutting_stock_column_generation(roll_width, sizes, demands):
    """
    Solve cutting stock problem using column generation
    """
    n = len(sizes)
    
    # Initialize with simple patterns (one size per roll)
    patterns = []
    for i in range(n):
        pattern = [0] * n
        pattern[i] = roll_width // sizes[i]
        patterns.append(pattern)
    
    iteration = 0
    while True:
        iteration += 1
        
        # Solve master problem
        # Variables: y_p = number of times pattern p is used
        # Minimize: sum(y_p)
        # Subject to: sum(pattern_p[i] * y_p) >= demand[i]
        
        # (Using LP solver)
        solution, dual_values = solve_master_problem(patterns, demands)
        
        # Solve pricing subproblem
        new_pattern, reduced_cost = pricing_subproblem(
            roll_width, sizes, demands, dual_values
        )
        
        print(f"Iteration {iteration}: Reduced cost = {reduced_cost:.4f}")
        
        if reduced_cost >= -1e-6:  # Optimal (with tolerance)
            break
        
        # Add new pattern
        patterns.append(new_pattern)
    
    return patterns, solution
```

## Branch-and-Price

Extends column generation to integer programming:

```
1. Solve LP using column generation at each B&B node

2. Branching:
   - Cannot branch on individual variables (too many)
   - Branch on problem structure:
     * Ryan-Foster branching (for set partitioning)
     * Follow-on branching (for routing)
   
3. Pricing subproblem:
   - Must respect branching decisions
   - May become more complex in subtree

4. Column management:
   - Some columns valid only in subtree
   - Maintain separate column pools
```

## Example: Vehicle Routing Problem

### Master Problem
```
Minimize: sum of route costs
Subject to: each customer visited exactly once
```

### Pricing Subproblem (Shortest Path with Resource Constraints)
```python
def pricing_vrp(graph, customers, dual_values, vehicle_capacity):
    """
    Find route with most negative reduced cost
    
    Uses dynamic programming or labeling algorithm
    """
    # State: (current_node, remaining_capacity, visited_customers)
    # Cost: route_cost - sum(dual_value[customer])
    
    best_route = None
    best_reduced_cost = 0
    
    # Label-setting algorithm
    labels = {(depot, vehicle_capacity, frozenset()): 0}
    
    while labels:
        (node, capacity, visited), cost = min(labels.items(), key=lambda x: x[1])
        del labels[(node, capacity, visited)]
        
        # Try extending to each customer
        for next_node in customers:
            if next_node in visited:
                continue
            
            demand = get_demand(next_node)
            if demand > capacity:
                continue
            
            # Calculate cost
            edge_cost = graph[node][next_node]
            new_cost = cost + edge_cost - dual_values[next_node]
            new_capacity = capacity - demand
            new_visited = visited | {next_node}
            
            state = (next_node, new_capacity, new_visited)
            
            if state not in labels or new_cost < labels[state]:
                labels[state] = new_cost
        
        # Try returning to depot
        if node != depot:
            return_cost = cost + graph[node][depot]
            if return_cost < best_reduced_cost:
                best_reduced_cost = return_cost
                best_route = list(visited)
    
    return best_route, best_reduced_cost
```

## Stabilization Techniques

Column generation can suffer from slow convergence due to oscillating dual values.

### Dual Stabilization
```python
def stabilized_column_generation(master, pricing, alpha=0.5):
    """
    Stabilize dual values using smoothing
    
    alpha: smoothing parameter (0 = no smoothing, 1 = full smoothing)
    """
    dual_values = initialize_duals()
    dual_center = dual_values.copy()
    
    while True:
        # Solve master with stabilization
        solution, new_duals = solve_master_stabilized(
            master, dual_center, alpha
        )
        
        # Solve pricing
        columns, reduced_costs = pricing(new_duals)
        
        if all(rc >= -1e-6 for rc in reduced_costs):
            # Check optimality with original dual values
            _, true_reduced_costs = pricing(dual_values)
            if all(rc >= -1e-6 for rc in true_reduced_costs):
                break
        
        # Update dual center
        dual_center = alpha * dual_center + (1 - alpha) * new_duals
        dual_values = new_duals
        
        # Add columns
        master.add_columns(columns)
    
    return solution
```

## Implementation Tips

### Column Pool Management
- Store all generated columns
- Reuse columns across iterations
- Periodically remove unused columns

### Multiple Pricing
- Generate multiple columns per iteration
- Balance between pricing time and master problem size

### Heuristic Pricing
- Use heuristics to find good columns quickly
- Exact pricing only when needed

### Warm Starting
- Use previous solutions to initialize
- Maintain basis information

## Advantages

- Handles problems with exponentially many variables
- Often provides tight LP bounds
- Natural decomposition of problem structure

## Disadvantages

- Complex implementation
- Pricing subproblem can be difficult
- Convergence can be slow without stabilization
- Branching rules require careful design

## Applications

- **Cutting Stock**: Minimize waste in manufacturing
- **Crew Scheduling**: Airline/railway crew assignment
- **Vehicle Routing**: Delivery route optimization
- **Network Design**: Capacity planning
- **Bin Packing**: Container loading
- **Shift Scheduling**: Workforce planning

## Modern Extensions

- **Column-and-Row Generation**: Generate constraints dynamically
- **Nested Column Generation**: Pricing subproblem uses column generation
- **Parallel Pricing**: Solve multiple pricing problems simultaneously
- **Machine Learning**: Predict promising columns to generate
