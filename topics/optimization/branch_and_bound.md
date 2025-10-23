# Branch and Bound

## Overview

Branch and Bound (B&B) is a general algorithm for finding optimal solutions to optimization problems, particularly useful for integer and combinatorial problems.

## Key Concepts

### Branching
- **Purpose**: Divide the problem into smaller subproblems
- **Method**: Partition the solution space by fixing variables
- **Example**: For binary variable x, create two branches: x=0 and x=1

### Bounding
- **Purpose**: Establish upper/lower bounds on optimal solution
- **Method**: Solve relaxed versions of the problem (e.g., LP relaxation)
- **Use**: Prune branches that cannot improve the current best solution

### Pruning
Three types of pruning:
1. **By optimality**: Subproblem is solved optimally
2. **By bound**: Bound is worse than current best solution
3. **By infeasibility**: Subproblem has no feasible solution

## Algorithm

```
1. Initialize:
   - Best solution = None
   - Queue = [original problem]

2. While queue is not empty:
   a. Select and remove a node from queue
   b. Compute bound for this node
   c. If bound is worse than best solution, prune
   d. If node is feasible and better than best, update best
   e. If node is not fully explored, branch:
      - Create child nodes
      - Add to queue

3. Return best solution
```

## Example: Knapsack Problem

```python
def branch_and_bound_knapsack(items, capacity):
    """
    items: list of (value, weight) tuples
    capacity: maximum weight
    """
    best_value = 0
    best_solution = []
    
    def bound(node, remaining_capacity, remaining_items):
        # LP relaxation: take fractional items
        bound_value = node.value
        weight = node.weight
        
        for item in remaining_items:
            if weight + item.weight <= remaining_capacity:
                bound_value += item.value
                weight += item.weight
            else:
                # Take fraction of item
                fraction = (remaining_capacity - weight) / item.weight
                bound_value += item.value * fraction
                break
        
        return bound_value
    
    def branch(node, remaining_items):
        nonlocal best_value, best_solution
        
        if not remaining_items:
            if node.value > best_value:
                best_value = node.value
                best_solution = node.items
            return
        
        item = remaining_items[0]
        rest = remaining_items[1:]
        
        # Branch 1: Include item
        if node.weight + item.weight <= capacity:
            new_node = Node(
                node.value + item.value,
                node.weight + item.weight,
                node.items + [item]
            )
            if bound(new_node, capacity - new_node.weight, rest) > best_value:
                branch(new_node, rest)
        
        # Branch 2: Exclude item
        if bound(node, capacity - node.weight, rest) > best_value:
            branch(node, rest)
    
    # Sort items by value/weight ratio (greedy heuristic)
    sorted_items = sorted(items, key=lambda x: x[0]/x[1], reverse=True)
    initial_node = Node(0, 0, [])
    branch(initial_node, sorted_items)
    
    return best_value, best_solution
```

## Node Selection Strategies

1. **Depth-First Search (DFS)**: Explore deeply, memory efficient
2. **Best-First Search**: Select node with best bound, finds optimal faster
3. **Breadth-First Search (BFS)**: Explore level by level
4. **Hybrid**: Combine strategies for balance

## Advantages

- Guarantees optimal solution
- Can terminate early with proven optimality
- Flexible framework for many problem types

## Disadvantages

- Exponential worst-case complexity
- Memory intensive for large problems
- Performance depends heavily on bound quality

## Applications

- **Integer Programming**: General MIP solver component
- **Traveling Salesman Problem**: Route optimization
- **Job Scheduling**: Task assignment with constraints
- **Facility Location**: Optimal site selection
- **Portfolio Optimization**: Asset selection with constraints
