# Set Coverage Problems

## Overview

Set coverage problems involve selecting a subset of sets to cover all elements with minimum cost. These are fundamental combinatorial optimization problems with numerous applications.

## Problem Variants

### 1. Set Cover Problem (SCP)
**Goal**: Cover all elements using minimum number of sets

**Formulation**:
```
Minimize: sum of x_i (number of sets selected)
Subject to: sum of x_i for i in S_j >= 1  (each element j is covered)
            x_i ∈ {0, 1}
```

### 2. Weighted Set Cover
**Goal**: Minimize total cost of selected sets

**Formulation**:
```
Minimize: sum of c_i * x_i
Subject to: sum of x_i for i in S_j >= 1
            x_i ∈ {0, 1}
```

### 3. Set Packing Problem
**Goal**: Select maximum number of disjoint sets

**Formulation**:
```
Maximize: sum of x_i
Subject to: sum of x_i for i in S_j <= 1  (each element in at most one set)
            x_i ∈ {0, 1}
```

### 4. Set Partitioning Problem
**Goal**: Partition elements into disjoint sets with minimum cost

**Formulation**:
```
Minimize: sum of c_i * x_i
Subject to: sum of x_i for i in S_j = 1  (each element in exactly one set)
            x_i ∈ {0, 1}
```

## Solution Methods

### 1. Greedy Algorithm

```python
def greedy_set_cover(universe, sets, costs=None):
    """
    Greedy algorithm for set cover
    Approximation ratio: O(log n) where n is universe size
    
    universe: set of all elements to cover
    sets: list of sets (each set is a set of elements)
    costs: list of costs (default: uniform cost of 1)
    """
    if costs is None:
        costs = [1] * len(sets)
    
    uncovered = set(universe)
    selected_sets = []
    total_cost = 0
    
    while uncovered:
        # Find set with best cost-effectiveness
        best_ratio = float('inf')
        best_set_idx = None
        
        for i, s in enumerate(sets):
            if i in selected_sets:
                continue
            
            # Elements covered by this set
            newly_covered = s & uncovered
            
            if newly_covered:
                ratio = costs[i] / len(newly_covered)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_set_idx = i
        
        if best_set_idx is None:
            break  # No more sets can cover remaining elements
        
        # Select best set
        selected_sets.append(best_set_idx)
        total_cost += costs[best_set_idx]
        uncovered -= sets[best_set_idx]
    
    if uncovered:
        return None  # Cannot cover all elements
    
    return selected_sets, total_cost


# Example usage
universe = {1, 2, 3, 4, 5, 6, 7, 8}
sets = [
    {1, 2, 3},
    {2, 4, 5},
    {3, 6, 7},
    {5, 6, 8},
    {1, 4, 7, 8}
]
costs = [3, 2, 2, 3, 4]

solution, cost = greedy_set_cover(universe, sets, costs)
print(f"Selected sets: {solution}")
print(f"Total cost: {cost}")
```

### 2. Integer Programming

```python
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary

def ip_set_cover(universe, sets, costs=None):
    """
    Solve set cover using integer programming
    """
    if costs is None:
        costs = [1] * len(sets)
    
    # Create problem
    prob = LpProblem("SetCover", LpMinimize)
    
    # Decision variables: x[i] = 1 if set i is selected
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(len(sets))]
    
    # Objective: minimize total cost
    prob += lpSum([costs[i] * x[i] for i in range(len(sets))])
    
    # Constraints: each element must be covered
    for element in universe:
        # Find sets containing this element
        covering_sets = [i for i, s in enumerate(sets) if element in s]
        prob += lpSum([x[i] for i in covering_sets]) >= 1
    
    # Solve
    prob.solve()
    
    # Extract solution
    selected = [i for i in range(len(sets)) if x[i].varValue > 0.5]
    total_cost = sum(costs[i] for i in selected)
    
    return selected, total_cost
```

### 3. Branch and Bound with Cutting Planes

```python
class SetCoverBnB:
    def __init__(self, universe, sets, costs):
        self.universe = universe
        self.sets = sets
        self.costs = costs
        self.n_sets = len(sets)
        
        self.best_solution = None
        self.best_cost = float('inf')
    
    def lp_relaxation(self, fixed_vars):
        """
        Solve LP relaxation with some variables fixed
        Returns lower bound
        """
        # Use LP solver with 0 <= x_i <= 1
        # (Implementation using LP solver)
        pass
    
    def generate_cover_cuts(self, lp_solution):
        """
        Generate cutting planes for set cover
        """
        cuts = []
        
        # Minimal cover inequalities
        for element in self.universe:
            covering_sets = [i for i, s in enumerate(self.sets) if element in s]
            
            # If sum of LP values < 1, add cut
            lp_sum = sum(lp_solution[i] for i in covering_sets)
            if lp_sum < 0.99:  # Tolerance
                cut = {
                    'sets': covering_sets,
                    'rhs': 1.0,
                    'type': '>='
                }
                cuts.append(cut)
        
        return cuts
    
    def branch_and_bound(self, node, depth=0):
        """
        Branch and bound with cutting planes
        """
        # Solve LP relaxation
        lp_solution, lp_bound = self.lp_relaxation(node.fixed_vars)
        
        # Prune by bound
        if lp_bound >= self.best_cost:
            return
        
        # Generate and add cuts
        cuts = self.generate_cover_cuts(lp_solution)
        if cuts:
            # Resolve with cuts
            lp_solution, lp_bound = self.lp_relaxation_with_cuts(
                node.fixed_vars, cuts
            )
        
        # Check if integer solution
        if all(x in [0, 1] for x in lp_solution):
            cost = sum(self.costs[i] * lp_solution[i] 
                      for i in range(self.n_sets))
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = [i for i in range(self.n_sets) 
                                     if lp_solution[i] > 0.5]
            return
        
        # Branch on most fractional variable
        frac_var = max(range(self.n_sets),
                      key=lambda i: abs(lp_solution[i] - 0.5))
        
        # Branch 1: set variable to 1
        node1 = node.copy()
        node1.fixed_vars[frac_var] = 1
        self.branch_and_bound(node1, depth + 1)
        
        # Branch 2: set variable to 0
        node2 = node.copy()
        node2.fixed_vars[frac_var] = 0
        self.branch_and_bound(node2, depth + 1)
```

### 4. Column Generation

```python
def set_cover_column_generation(universe, set_generator, max_iterations=100):
    """
    Solve large-scale set cover using column generation
    
    set_generator: function that generates new sets given dual values
    """
    # Start with initial sets (e.g., singleton sets)
    current_sets = [{element} for element in universe]
    current_costs = [1] * len(current_sets)
    
    for iteration in range(max_iterations):
        # Solve master problem (LP relaxation)
        solution, dual_values = solve_master_lp(
            universe, current_sets, current_costs
        )
        
        # Pricing: find new set with negative reduced cost
        new_set, reduced_cost = set_generator(dual_values)
        
        if reduced_cost >= -1e-6:
            # Optimal solution found
            break
        
        # Add new set to master problem
        current_sets.append(new_set)
        current_costs.append(compute_set_cost(new_set))
    
    return solution, current_sets


def pricing_subproblem(dual_values, universe, max_set_size):
    """
    Pricing subproblem: find set with maximum dual value sum
    This is a knapsack-like problem
    """
    # Sort elements by dual value
    sorted_elements = sorted(universe, 
                            key=lambda e: dual_values[e], 
                            reverse=True)
    
    # Greedily select elements
    new_set = set()
    total_dual = 0
    
    for element in sorted_elements:
        if len(new_set) < max_set_size:
            new_set.add(element)
            total_dual += dual_values[element]
    
    # Reduced cost = cost - dual value sum
    cost = len(new_set)  # Or some other cost function
    reduced_cost = cost - total_dual
    
    return new_set, reduced_cost
```

## Applications

### 1. Crew Scheduling
**Problem**: Assign crew members to cover all flights
- Sets: Possible crew schedules (sequences of flights)
- Elements: Flights that must be covered
- Constraints: Crew regulations, rest periods

### 2. Facility Location
**Problem**: Select facilities to serve all customers
- Sets: Service areas of each facility
- Elements: Customers
- Objective: Minimize facility opening costs

### 3. Sensor Placement
**Problem**: Place sensors to monitor all locations
- Sets: Coverage areas of each sensor
- Elements: Locations to monitor
- Constraints: Budget, sensor types

### 4. Test Case Selection
**Problem**: Select test cases to cover all code paths
- Sets: Code paths covered by each test
- Elements: Code paths
- Objective: Minimize testing time

### 5. Feature Selection
**Problem**: Select features to represent all data aspects
- Sets: Data points covered by each feature
- Elements: Data points or patterns
- Objective: Minimize feature count

## Example: Crew Scheduling

```python
def crew_scheduling_example():
    """
    Airline crew scheduling as set cover problem
    """
    # Flights to cover
    flights = ['F1', 'F2', 'F3', 'F4', 'F5']
    
    # Possible crew schedules (pairings)
    # Each pairing is a sequence of flights one crew can fly
    pairings = [
        {'flights': {'F1', 'F2'}, 'cost': 1000},
        {'flights': {'F2', 'F3'}, 'cost': 1200},
        {'flights': {'F3', 'F4', 'F5'}, 'cost': 1500},
        {'flights': {'F1', 'F4'}, 'cost': 1100},
        {'flights': {'F4', 'F5'}, 'cost': 900},
    ]
    
    # Convert to set cover format
    universe = set(flights)
    sets = [p['flights'] for p in pairings]
    costs = [p['cost'] for p in pairings]
    
    # Solve
    solution, total_cost = ip_set_cover(universe, sets, costs)
    
    print("Selected pairings:")
    for idx in solution:
        print(f"  Pairing {idx}: {pairings[idx]['flights']} "
              f"(cost: {pairings[idx]['cost']})")
    print(f"Total cost: {total_cost}")
    
    return solution

crew_scheduling_example()
```

## Complexity and Approximation

### Complexity
- Set cover is NP-hard
- No polynomial-time algorithm unless P=NP
- Even approximation within (1-ε)ln(n) is NP-hard

### Approximation Algorithms
1. **Greedy**: O(log n) approximation
2. **LP Rounding**: O(log n) approximation
3. **Primal-Dual**: O(f) approximation where f is max frequency

### Special Cases
- **Vertex Cover**: 2-approximation (special case where each set has size 2)
- **Geometric Set Cover**: PTAS exists for geometric instances

## Heuristics and Metaheuristics

### Local Search
```python
def local_search_set_cover(universe, sets, costs, initial_solution):
    """
    Improve solution using local search
    """
    current = set(initial_solution)
    current_cost = sum(costs[i] for i in current)
    
    improved = True
    while improved:
        improved = False
        
        # Try removing each set
        for i in current:
            remaining = current - {i}
            if is_feasible(universe, sets, remaining):
                new_cost = sum(costs[j] for j in remaining)
                if new_cost < current_cost:
                    current = remaining
                    current_cost = new_cost
                    improved = True
                    break
        
        # Try swapping sets
        if not improved:
            for i in current:
                for j in range(len(sets)):
                    if j not in current:
                        candidate = (current - {i}) | {j}
                        if is_feasible(universe, sets, candidate):
                            new_cost = sum(costs[k] for k in candidate)
                            if new_cost < current_cost:
                                current = candidate
                                current_cost = new_cost
                                improved = True
                                break
                if improved:
                    break
    
    return list(current), current_cost
```

## Advanced Topics

### 1. Online Set Cover
Elements arrive online, must be covered immediately

### 2. Robust Set Cover
Uncertainty in element requirements or set availability

### 3. Multi-Objective Set Cover
Optimize multiple objectives (cost, reliability, etc.)

### 4. Capacitated Set Cover
Sets have limited capacity for covering elements
