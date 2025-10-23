# SAT Solving (Boolean Satisfiability)

## Overview

The Boolean Satisfiability Problem (SAT) asks whether there exists an assignment of truth values to variables that makes a Boolean formula evaluate to true. SAT is the first problem proven to be NP-complete and is fundamental to computer science.

## Problem Definition

### CNF (Conjunctive Normal Form)
Standard form for SAT: conjunction of clauses, where each clause is a disjunction of literals.

**Example**:
```
(x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂) ∧ (x₂ ∨ ¬x₃)
```

### Variants

1. **SAT**: General satisfiability
2. **3-SAT**: Each clause has exactly 3 literals
3. **MAX-SAT**: Maximize number of satisfied clauses
4. **Weighted MAX-SAT**: Maximize weighted sum of satisfied clauses
5. **#SAT**: Count number of satisfying assignments

## Basic SAT Solver Implementation

```python
class SATSolver:
    def __init__(self, clauses, n_vars):
        """
        clauses: list of lists, each inner list is a clause
                 positive integers represent variables
                 negative integers represent negated variables
        n_vars: number of variables
        
        Example: [[1, -2, 3], [-1, 2], [2, -3]]
        represents (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂) ∧ (x₂ ∨ ¬x₃)
        """
        self.clauses = [set(clause) for clause in clauses]
        self.n_vars = n_vars
        self.assignment = {}
        self.decision_level = 0
    
    def evaluate_clause(self, clause):
        """
        Evaluate clause under current assignment
        Returns: True (satisfied), False (unsatisfied), None (undetermined)
        """
        has_unassigned = False
        
        for literal in clause:
            var = abs(literal)
            sign = literal > 0
            
            if var not in self.assignment:
                has_unassigned = True
            elif self.assignment[var] == sign:
                return True  # Clause satisfied
        
        return None if has_unassigned else False
    
    def unit_propagation(self):
        """
        Unit propagation: if a clause has only one unassigned literal,
        assign it to satisfy the clause
        """
        changed = True
        while changed:
            changed = False
            
            for clause in self.clauses:
                result = self.evaluate_clause(clause)
                
                if result is False:
                    return False  # Conflict
                
                if result is None:
                    # Find unassigned literals
                    unassigned = [lit for lit in clause 
                                 if abs(lit) not in self.assignment]
                    
                    if len(unassigned) == 1:
                        # Unit clause: must assign this literal
                        literal = unassigned[0]
                        var = abs(literal)
                        value = literal > 0
                        self.assignment[var] = value
                        changed = True
        
        return True  # No conflict
    
    def pure_literal_elimination(self):
        """
        Pure literal: appears with only one polarity
        Can be assigned to satisfy all clauses containing it
        """
        literal_polarities = {}
        
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal)
                if var in self.assignment:
                    continue
                
                sign = literal > 0
                if var not in literal_polarities:
                    literal_polarities[var] = set()
                literal_polarities[var].add(sign)
        
        # Assign pure literals
        for var, polarities in literal_polarities.items():
            if len(polarities) == 1:
                self.assignment[var] = list(polarities)[0]
    
    def choose_variable(self):
        """
        Choose next variable to assign (branching heuristic)
        """
        # Simple heuristic: choose first unassigned variable
        for var in range(1, self.n_vars + 1):
            if var not in self.assignment:
                return var
        return None
    
    def solve(self):
        """
        DPLL algorithm: Davis-Putnam-Logemann-Loveland
        """
        return self.dpll()
    
    def dpll(self):
        """
        Recursive DPLL algorithm
        """
        # Unit propagation
        if not self.unit_propagation():
            return False  # Conflict
        
        # Pure literal elimination
        self.pure_literal_elimination()
        
        # Check if all clauses satisfied
        all_satisfied = all(
            self.evaluate_clause(clause) == True 
            for clause in self.clauses
        )
        if all_satisfied:
            return True
        
        # Choose variable to branch on
        var = self.choose_variable()
        if var is None:
            # All variables assigned but not all clauses satisfied
            return False
        
        # Try assigning True
        saved_assignment = self.assignment.copy()
        self.assignment[var] = True
        if self.dpll():
            return True
        
        # Backtrack and try False
        self.assignment = saved_assignment
        self.assignment[var] = False
        return self.dpll()


# Example usage
clauses = [
    [1, -2, 3],   # (x₁ ∨ ¬x₂ ∨ x₃)
    [-1, 2],      # (¬x₁ ∨ x₂)
    [2, -3],      # (x₂ ∨ ¬x₃)
    [-2, 3]       # (¬x₂ ∨ x₃)
]

solver = SATSolver(clauses, n_vars=3)
if solver.solve():
    print("SAT")
    print("Assignment:", solver.assignment)
else:
    print("UNSAT")
```

## Modern SAT Solving: CDCL

Conflict-Driven Clause Learning (CDCL) is the basis of modern SAT solvers.

```python
class CDCLSolver:
    def __init__(self, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars
        self.assignment = {}
        self.decision_level = {}
        self.antecedent = {}  # Clause that forced assignment
        self.current_level = 0
        self.learned_clauses = []
    
    def analyze_conflict(self, conflict_clause):
        """
        Analyze conflict and learn a new clause
        Returns: (learned_clause, backtrack_level)
        """
        if self.current_level == 0:
            return None, -1  # UNSAT
        
        # Build implication graph and find UIP
        # (First Unique Implication Point)
        learned = set(conflict_clause)
        current_level_lits = []
        
        for lit in learned:
            var = abs(lit)
            if self.decision_level.get(var) == self.current_level:
                current_level_lits.append(lit)
        
        # Resolve until one literal from current level
        while len(current_level_lits) > 1:
            # Choose literal to resolve on
            lit = current_level_lits[0]
            var = abs(lit)
            
            if var in self.antecedent:
                # Resolve with antecedent clause
                antecedent = self.antecedent[var]
                learned = self.resolve(learned, antecedent, lit)
                
                # Update current level literals
                current_level_lits = [
                    l for l in learned 
                    if self.decision_level.get(abs(l)) == self.current_level
                ]
            else:
                break
        
        # Compute backtrack level
        levels = [self.decision_level.get(abs(lit), 0) 
                 for lit in learned]
        levels = [l for l in levels if l < self.current_level]
        backtrack_level = max(levels) if levels else 0
        
        return list(learned), backtrack_level
    
    def resolve(self, clause1, clause2, literal):
        """
        Resolution: (A ∨ x) ∧ (B ∨ ¬x) → (A ∨ B)
        """
        result = set(clause1) | set(clause2)
        result.discard(literal)
        result.discard(-literal)
        return result
    
    def backtrack(self, level):
        """
        Backtrack to given decision level
        """
        vars_to_unassign = [
            var for var, lvl in self.decision_level.items() 
            if lvl > level
        ]
        
        for var in vars_to_unassign:
            del self.assignment[var]
            del self.decision_level[var]
            if var in self.antecedent:
                del self.antecedent[var]
        
        self.current_level = level
    
    def vsids_heuristic(self):
        """
        Variable State Independent Decaying Sum
        Modern branching heuristic
        """
        # Maintain activity scores for variables
        # Bump scores when variables appear in conflicts
        # Decay scores periodically
        # Choose variable with highest score
        pass
    
    def solve(self):
        """
        CDCL algorithm
        """
        while True:
            # Unit propagation
            conflict = self.unit_propagation()
            
            if conflict is not None:
                # Conflict occurred
                if self.current_level == 0:
                    return False  # UNSAT
                
                # Learn clause from conflict
                learned_clause, backtrack_level = self.analyze_conflict(conflict)
                
                if learned_clause is None:
                    return False  # UNSAT
                
                # Add learned clause
                self.learned_clauses.append(learned_clause)
                self.clauses.append(learned_clause)
                
                # Backtrack
                self.backtrack(backtrack_level)
            
            else:
                # No conflict
                if len(self.assignment) == self.n_vars:
                    return True  # SAT
                
                # Make decision
                var = self.choose_variable()
                self.current_level += 1
                self.assignment[var] = True  # Or use polarity heuristic
                self.decision_level[var] = self.current_level
```

## Advanced Techniques

### 1. Preprocessing

```python
def preprocess_cnf(clauses):
    """
    Simplify CNF formula before solving
    """
    # Remove tautologies: clauses containing both x and ¬x
    clauses = [c for c in clauses if not is_tautology(c)]
    
    # Subsumption: remove clauses subsumed by others
    # Clause A subsumes B if A ⊆ B
    clauses = remove_subsumed(clauses)
    
    # Variable elimination: eliminate variables that appear few times
    clauses = eliminate_variables(clauses)
    
    # Blocked clause elimination
    clauses = eliminate_blocked_clauses(clauses)
    
    return clauses

def is_tautology(clause):
    """Check if clause contains both x and ¬x"""
    literals = set(clause)
    for lit in literals:
        if -lit in literals:
            return True
    return False
```

### 2. Restart Strategies

```python
class RestartingSolver(CDCLSolver):
    def __init__(self, clauses, n_vars):
        super().__init__(clauses, n_vars)
        self.conflicts = 0
        self.restart_threshold = 100
    
    def solve_with_restarts(self):
        """
        Restart search periodically to escape bad search space regions
        """
        while True:
            # Solve with conflict limit
            result = self.solve_limited(self.restart_threshold)
            
            if result is not None:
                return result  # SAT or UNSAT
            
            # Restart
            self.backtrack(0)
            self.conflicts = 0
            
            # Increase restart threshold (Luby sequence or geometric)
            self.restart_threshold = int(self.restart_threshold * 1.5)
```

### 3. Clause Database Management

```python
def manage_learned_clauses(self):
    """
    Periodically remove less useful learned clauses
    """
    if len(self.learned_clauses) > self.max_learned:
        # Keep clauses with high activity (appeared in recent conflicts)
        # Remove clauses with low LBD (Literal Block Distance)
        self.learned_clauses.sort(key=lambda c: self.clause_activity[c])
        self.learned_clauses = self.learned_clauses[-self.max_learned:]
```

## Applications

### 1. Hardware Verification
```python
def verify_circuit(circuit_spec, property_spec):
    """
    Verify hardware design satisfies properties
    """
    # Encode circuit as CNF
    circuit_cnf = encode_circuit(circuit_spec)
    
    # Encode property violation as CNF
    property_cnf = encode_property_negation(property_spec)
    
    # Combine: circuit ∧ ¬property
    combined_cnf = circuit_cnf + property_cnf
    
    # Solve
    solver = SATSolver(combined_cnf, n_vars=...)
    if solver.solve():
        # Found counterexample
        return False, solver.assignment
    else:
        # Property holds
        return True, None
```

### 2. Software Verification

```python
def bounded_model_checking(program, assertion, depth):
    """
    Check if assertion can be violated within depth steps
    """
    # Encode program execution for depth steps
    cnf = []
    
    for step in range(depth):
        # Encode program state at step
        state_cnf = encode_program_state(program, step)
        cnf.extend(state_cnf)
        
        # Encode transition from step to step+1
        transition_cnf = encode_transition(program, step)
        cnf.extend(transition_cnf)
    
    # Encode assertion violation
    violation_cnf = encode_assertion_violation(assertion, depth)
    cnf.extend(violation_cnf)
    
    # Solve
    solver = SATSolver(cnf, n_vars=...)
    return solver.solve()
```

### 3. Planning

```python
def planning_as_sat(initial_state, goal_state, actions, horizon):
    """
    Encode planning problem as SAT
    """
    cnf = []
    
    # Encode initial state
    cnf.extend(encode_state(initial_state, time=0))
    
    # Encode goal state
    cnf.extend(encode_state(goal_state, time=horizon))
    
    # Encode actions and frame axioms
    for t in range(horizon):
        # At least one action per time step
        action_vars = [action_var(a, t) for a in actions]
        cnf.append(action_vars)
        
        # Action effects
        for action in actions:
            cnf.extend(encode_action_effects(action, t))
        
        # Frame axioms: things don't change unless caused by action
        cnf.extend(encode_frame_axioms(actions, t))
    
    # Solve
    solver = SATSolver(cnf, n_vars=...)
    if solver.solve():
        # Extract plan from assignment
        plan = extract_plan(solver.assignment, actions, horizon)
        return plan
    else:
        return None
```

### 4. Scheduling

```python
def job_shop_scheduling_sat(jobs, machines, deadline):
    """
    Encode job shop scheduling as SAT
    """
    cnf = []
    
    # Variables: job_i_on_machine_m_at_time_t
    # Constraints:
    # 1. Each job on exactly one machine at each time
    # 2. Each machine processes at most one job at each time
    # 3. Job precedence constraints
    # 4. Job completion before deadline
    
    for job in jobs:
        for time in range(deadline):
            # Job on exactly one machine
            machine_vars = [var(job, m, time) for m in machines]
            cnf.append(machine_vars)  # At least one
            cnf.extend(at_most_one(machine_vars))  # At most one
    
    for machine in machines:
        for time in range(deadline):
            # Machine processes at most one job
            job_vars = [var(j, machine, time) for j in jobs]
            cnf.extend(at_most_one(job_vars))
    
    # Precedence constraints
    for job in jobs:
        for task1, task2 in job.precedences:
            cnf.extend(encode_precedence(task1, task2))
    
    return cnf
```

## MAX-SAT

Maximize number of satisfied clauses when formula is unsatisfiable.

```python
def max_sat_branch_and_bound(clauses, n_vars):
    """
    Solve MAX-SAT using branch and bound
    """
    best_assignment = None
    best_satisfied = 0
    
    def count_satisfied(assignment):
        count = 0
        for clause in clauses:
            if any((lit > 0 and assignment.get(abs(lit)) == True) or
                   (lit < 0 and assignment.get(abs(lit)) == False)
                   for lit in clause):
                count += 1
        return count
    
    def branch(assignment, var_idx):
        nonlocal best_assignment, best_satisfied
        
        if var_idx > n_vars:
            satisfied = count_satisfied(assignment)
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = assignment.copy()
            return
        
        # Upper bound: current satisfied + remaining clauses
        current_satisfied = count_satisfied(assignment)
        upper_bound = current_satisfied + len(clauses)
        
        if upper_bound <= best_satisfied:
            return  # Prune
        
        # Branch on variable var_idx
        for value in [True, False]:
            assignment[var_idx] = value
            branch(assignment, var_idx + 1)
            del assignment[var_idx]
    
    branch({}, 1)
    return best_assignment, best_satisfied
```

## Modern SAT Solvers

Popular industrial-strength solvers:
- **MiniSat**: Educational solver, basis for many others
- **Glucose**: Improved clause learning
- **CryptoMiniSat**: Specialized for cryptographic problems
- **Lingeling**: Competition winner
- **Z3**: SMT solver with SAT core

## Performance Tips

1. **Good variable ordering**: Use VSIDS or similar heuristics
2. **Clause learning**: Learn short, relevant clauses
3. **Restarts**: Escape bad search regions
4. **Preprocessing**: Simplify before solving
5. **Incremental solving**: Reuse learned clauses across similar instances

## Complexity

- **SAT**: NP-complete
- **3-SAT**: NP-complete
- **2-SAT**: Polynomial time (linear)
- **MAX-SAT**: NP-hard
- **#SAT**: #P-complete (harder than NP)

## Extensions

- **SMT (Satisfiability Modulo Theories)**: SAT with theory reasoning
- **QBF (Quantified Boolean Formulas)**: SAT with quantifiers
- **Pseudo-Boolean Constraints**: Linear inequalities over Boolean variables
