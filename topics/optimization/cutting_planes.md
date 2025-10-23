# Cutting Planes

## Overview

Cutting planes are linear inequalities added to strengthen the LP relaxation of integer programming problems. They "cut off" fractional solutions without removing any integer feasible solutions.

## Key Concepts

### Valid Inequality
An inequality that is satisfied by all feasible integer solutions but may exclude some fractional solutions from the LP relaxation.

### Cutting Plane
A valid inequality that is violated by the current LP relaxation solution.

## Types of Cuts

### 1. Gomory Cuts
- **Source**: Derived from simplex tableau
- **Type**: General-purpose cuts for any IP
- **Method**: Round down coefficients of fractional basic variables

```python
def gomory_cut(tableau, basic_var_index):
    """
    Generate Gomory cut from simplex tableau
    """
    row = tableau[basic_var_index]
    rhs = row[-1]
    
    # Fractional part
    f0 = rhs - int(rhs)
    
    # Cut coefficients
    cut = []
    for coef in row[:-1]:
        fj = coef - int(coef)
        if fj <= f0:
            cut.append(fj / f0)
        else:
            cut.append((1 - fj) / (1 - f0))
    
    return cut, 1.0  # cut <= 1
```

### 2. Cover Cuts
- **Source**: Knapsack constraints
- **Application**: Set packing/covering problems
- **Example**: If items in set C exceed capacity, at most |C|-1 can be selected

```
Sum of x_i for i in C <= |C| - 1
```

### 3. Clique Cuts
- **Source**: Graph structure
- **Application**: Conflict graphs, coloring problems
- **Example**: In a clique of size k, at most 1 node can be selected

```
Sum of x_i for i in clique <= 1
```

### 4. Flow Cover Cuts
- **Source**: Network flow constraints
- **Application**: Lot-sizing, production planning
- **Method**: Identify minimal covers in flow networks

### 5. Mixed-Integer Rounding (MIR) Cuts
- **Source**: Mixed-integer constraints
- **Method**: Rounding and complementing variables
- **Strength**: Very effective for MIP problems

## Algorithm: Cut Generation

```
1. Solve LP relaxation
2. If solution is integer, STOP (optimal found)
3. Generate cutting planes:
   a. Identify violated inequalities
   b. Select most violated or most promising cuts
   c. Add cuts to LP
4. Resolve LP with added cuts
5. Repeat until:
   - Integer solution found
   - No more cuts can be generated
   - Iteration limit reached
```

## Cut Selection Strategies

### Efficacy
Measure of how much a cut improves the bound:
```
efficacy = |violation| / ||cut coefficients||
```

### Parallelism
Avoid adding cuts that are too similar (parallel):
```
parallelism = |dot_product(cut1, cut2)| / (||cut1|| * ||cut2||)
```

### Density
Prefer sparse cuts (fewer non-zero coefficients) for computational efficiency.

## Example: Knapsack Cover Cut

```python
def generate_cover_cut(items, capacity, solution):
    """
    Generate cover cut for knapsack constraint
    
    items: list of (weight, variable_index) tuples
    capacity: knapsack capacity
    solution: current LP solution values
    """
    # Find minimal cover: set of items exceeding capacity
    sorted_items = sorted(items, key=lambda x: solution[x[1]], reverse=True)
    
    cover = []
    total_weight = 0
    
    for weight, var_idx in sorted_items:
        cover.append(var_idx)
        total_weight += weight
        if total_weight > capacity:
            break
    
    if total_weight <= capacity:
        return None  # No cover found
    
    # Check if cover is minimal
    for var_idx in cover:
        weight = next(w for w, v in items if v == var_idx)
        if total_weight - weight > capacity:
            cover.remove(var_idx)
            total_weight -= weight
    
    # Generate cut: sum of variables in cover <= |cover| - 1
    cut_coefficients = {var_idx: 1.0 for var_idx in cover}
    cut_rhs = len(cover) - 1
    
    return cut_coefficients, cut_rhs
```

## Branch-and-Cut

Combines branch and bound with cutting planes:

```
1. At each B&B node:
   a. Solve LP relaxation
   b. Generate and add cuts
   c. Resolve LP
   d. If fractional, branch
   e. If integer, update incumbent

2. Cut management:
   - Local cuts: Valid only for subtree
   - Global cuts: Valid for entire tree
   - Age out inactive cuts
```

## Implementation Considerations

### Cut Pool
- Store generated cuts for reuse
- Periodically clean up ineffective cuts
- Limit pool size for memory management

### Cut Rounds
- Multiple rounds of cut generation per node
- Diminishing returns after several rounds
- Balance between cut generation time and bound improvement

### Numerical Stability
- Avoid cuts with very large coefficients
- Scale cuts appropriately
- Monitor LP solver stability

## Advantages

- Strengthens LP relaxation without branching
- Can dramatically reduce B&B tree size
- Provides better bounds early in search

## Disadvantages

- Cut generation can be time-consuming
- Too many cuts can slow down LP solver
- Numerical issues with dense cuts
- Requires problem-specific knowledge for best cuts

## Applications

- **Integer Programming**: Core component of modern MIP solvers
- **Traveling Salesman Problem**: Subtour elimination cuts
- **Vehicle Routing**: Capacity and route cuts
- **Scheduling**: Precedence and resource cuts
- **Network Design**: Connectivity and flow cuts

## Modern Solvers

Commercial solvers (CPLEX, Gurobi, Xpress) implement sophisticated cut generation:
- Automatic cut selection
- Problem-specific cut families
- Parallel cut generation
- Machine learning for cut selection
