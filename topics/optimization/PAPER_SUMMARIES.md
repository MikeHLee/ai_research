# Advanced Optimization Methods: Paper Summaries

This document provides concise, rigorous summaries of the advanced optimization methods from the research papers in the sources folder.

---

## 1. Large-Scale Zone-Based Evacuation Planning
**Paper**: 2003.11005v1.pdf (Hasan & Van Hentenryck, 2020)

### Problem Setup
Emergency evacuations require moving large populations from danger zones to safety through a transportation network with limited capacity. The challenge is to assign each residential zone an evacuation route and departure schedule that maximizes the number of people reaching safety before roads become blocked or disaster strikes.

### Key Constraints
- **Zone-based**: All residents from the same zone follow the same route
- **Time-expanded network**: Roads have time-varying capacities and block times
- **Contraflows**: Option to reverse inbound lanes for outbound traffic
- **Convergent paths**: Routes merge at intersections (no forks) to reduce confusion
- **Non-preemption**: Once a zone starts evacuating, it continues without interruption

### Solution Approaches

**1. Mixed Integer Programming (MIP)**
Formulates the problem with binary variables for path selection and continuous variables for flow quantities. Captures all constraints but becomes computationally expensive for large networks.

**2. Benders Decomposition**
Separates the problem into:
- **Master problem**: Selects evacuation paths for each zone
- **Subproblem**: Computes optimal departure schedules given paths
The master problem uses cuts from the subproblem to iteratively improve path selection until convergence.

**3. Conflict-Based Path Generation**
Heuristic approach that:
- Starts with simple initial paths
- Identifies capacity conflicts through simulation
- Generates alternative paths to resolve conflicts
- Iterates until a feasible solution is found

**4. Column Generation**
Treats each possible evacuation schedule as a "column":
- **Master problem**: Selects which schedules to use
- **Pricing subproblem**: Generates new promising schedules based on dual values
- Particularly effective for non-preemptive evacuations with response curves

### Key Insights
The paper demonstrates that contraflows can increase evacuation capacity by 20-30%, convergent paths reduce confusion but may increase total evacuation time by 5-10%, and non-preemptive evacuations are easier to enforce but require more sophisticated optimization algorithms. The choice of method depends on problem size and required features.

---

## 2. Optimization-Based Learning for Parcel Delivery Load Planning
**Paper**: 2307.04050v2.pdf (Ojha et al., 2023)

### Problem Setup
Parcel delivery companies must decide how many trailers to dispatch between terminals and how to route packages through the network. The challenge is that demand forecasts change daily, requiring frequent plan adjustments. Planners need consistent, explainable decisions that adapt to changing volumes while consolidating shipments efficiently.

### Core Challenge: Symmetry
The network has many equivalent solutions due to:
- Multiple paths between origin-destination pairs (primary and alternate routes)
- Interchangeable trailer assignments
- Equivalent consolidation options

This symmetry causes optimization solvers to return drastically different solutions for similar inputs, confusing planners and reducing trust.

### Solution Framework

**1. Lexicographical Optimization**
Eliminates symmetry by optimizing multiple objectives in priority order:
- **Primary**: Minimize total cost (trailer expenses + handling costs)
- **Secondary**: Minimize deviation from reference plan (previous day's plan)
- **Tertiary**: Prefer primary paths over alternates

This ensures solutions remain stable across days while still optimizing cost.

**2. Optimization Proxy (ML + Optimization Hybrid)**

The proxy combines machine learning and optimization:

**Step 1: Machine Learning Prediction**
- Train neural network on historical optimal solutions
- Input: demand forecast, network state, reference plan
- Output: predicted load assignments and flow routes

**Step 2: MIP-Based Repair**
- Fix ML predictions as initial solution
- Solve small MIP to repair infeasibilities
- Adjust flows to satisfy capacity and demand constraints
- Minimize changes to ML prediction

### Key Innovation
The optimization proxy is 10× faster than solving the full MIP while maintaining solution quality within 2-3% of optimal. More importantly, it generates consistent solutions across time periods, crucial for planner acceptance.

### Practical Impact
- Reduces computational time from minutes to seconds
- Maintains solution consistency day-to-day
- Enables real-time decision support for terminal planners
- Consolidates shipments more effectively than heuristic rules

---

## 3. Self-Supervised Learning for Security-Constrained Optimal Power Flow
**Paper**: 2311.18072v2.pdf (Park & Van Hentenryck, 2023)

### Problem Setup
Power grids must maintain supply-demand balance while ensuring the system remains stable even if any single generator or transmission line fails (N-1 security). The Security-Constrained Optimal Power Flow (SCOPF) problem finds the minimum-cost generator dispatch that satisfies:
- Power balance at all buses
- Transmission line capacity limits
- Generator capacity limits
- Feasibility under all single-component failures (contingencies)

### Computational Challenge
For a grid with G generators and K contingencies, traditional SCOPF requires solving a Mixed-Integer Linear Program with O(G×K) binary variables. For large grids (1000+ generators, 1000+ contingencies), this becomes computationally prohibitive for real-time operations.

### Solution: Primal-Dual Learning (PDL-SCOPF)

**Architecture**
The framework uses two neural networks:
- **Primal network P_θ**: Predicts generator dispatches
- **Dual network D_φ**: Predicts Lagrangian multipliers (shadow prices)

**Training Process (Self-Supervised)**
Unlike supervised learning (which requires optimal solutions for training), PDL-SCOPF trains using only:
- The SCOPF objective function (generation costs)
- Constraint violations (power balance, line limits)

The training mimics an Augmented Lagrangian Method:

1. **Forward pass**: 
   - Primal network predicts base case dispatch **g**
   - Repair layer enforces power balance: **g** ← **g** + correction
   - Binary search layer computes contingency dispatches using Automatic Primary Response
   - Dual network predicts multipliers **λ** for contingency constraints

2. **Loss computation**:
   ```
   Loss = generation_cost(**g**) 
        + penalty × constraint_violations
        + dual_term(**λ**, violations)
   ```

3. **Backpropagation**: Update both networks end-to-end

**Key Components**

1. **Repair Layer** (Differentiable)
   Ensures power balance by adjusting generator outputs:
   ```
   imbalance = sum(generation) - sum(demand)
   correction = -imbalance / num_generators
   g_i ← g_i + correction for all generators i
   ```

2. **Binary Search Layer** (Differentiable)
   Computes contingency dispatches using Automatic Primary Response:
   - When generator k fails, remaining generators increase output proportionally
   - Binary search finds dispatch that satisfies line limits
   - Implemented as differentiable operations for gradient flow

### Key Innovation
PDL-SCOPF solves large-scale SCOPF problems in **milliseconds** (vs. minutes for traditional methods) while achieving:
- Near-optimal costs (within 0.1-0.5% of optimal)
- 99%+ feasibility rate
- No need for training data with optimal solutions

### Practical Impact
Enables real-time SCOPF for large grids, replacing the current practice of using suboptimal reserve requirements. This improves grid efficiency and reliability, especially critical for integrating renewable energy.

---

## 4. Contextual Stochastic Optimization for Order Fulfillment
**Paper**: 2409.06918v3.pdf (Ye et al., 2024)

### Problem Setup
E-commerce companies must fulfill multi-item orders by selecting:
- **Fulfillment centers**: Warehouses or physical stores
- **Shipping carriers**: Traditional couriers or gig economy drivers (e.g., Roadie)

The challenge is balancing three objectives:
1. Minimize shipping costs
2. Meet customer delivery date expectations
3. Consolidate items when possible (ship together)

### Key Uncertainty
Delivery times are uncertain and depend on:
- Carrier reliability (varies by carrier and route)
- Weather conditions
- Traffic patterns
- Package characteristics (size, weight)

Traditional approaches use deterministic estimates, leading to frequent delivery delays.

### Solution: Contextual Stochastic Optimization (CSO)

**Framework Components**

**1. Distributional Forecasting**
Instead of predicting a single delivery time, predict the full probability distribution:

- **Input features (context)**: 
  - Package characteristics (weight, dimensions)
  - Route information (origin, destination, distance)
  - Carrier historical performance
  - Weather forecasts
  - Time of year, day of week

- **Model**: Quantile regression neural network
  - Outputs: 10th, 25th, 50th, 75th, 90th percentiles of delivery time
  - Captures uncertainty and tail risks

**2. Stochastic Optimization Model**

**Decision variables**:
- x_ijc = 1 if item i is fulfilled from center j using carrier c
- Consolidation: Items in same order can share shipments

**Objective** (risk-aware):
```
Minimize: shipping_costs + α × CVaR(delivery_time_deviation)
```

Where CVaR (Conditional Value at Risk) measures the expected deviation in the worst 10% of scenarios.

**Constraints**:
- Each item fulfilled exactly once
- Carrier capacity limits
- Consolidation rules (items must ship from same center)
- Service level: P(delivery ≤ expected_date) ≥ 95%

**3. Scenario-Based Formulation**

Generate S scenarios of delivery times from the distributional forecasts:
- Sample from predicted quantile distributions
- Each scenario represents a possible realization
- Optimization considers all scenarios simultaneously

**Robust Variant**:
For risk-averse decisions, use robust optimization:
```
Minimize: shipping_costs
Subject to: delivery ≤ expected_date in all scenarios
```

### Key Innovation
The CSO framework integrates machine learning predictions directly into optimization:
1. ML model learns delivery time distributions from historical data
2. Optimization uses these distributions to make risk-aware decisions
3. Context (weather, traffic, etc.) influences both prediction and decision

### Results
Compared to current heuristic practices:
- **15-20% reduction** in delivery delays
- **8-12% reduction** in shipping costs (through better consolidation)
- **Flexible risk management**: Adjust α to balance cost vs. reliability

### Practical Impact
First study to combine omnichannel fulfillment, multiple couriers, and delivery time uncertainty using contextual optimization. Provides actionable framework for retailers to improve customer satisfaction while controlling costs.

---

## 5. Weather-Informed Probabilistic Forecasting for Power Systems
**Paper**: 2409.07637v1.pdf (Zhang et al., 2024)

### Problem Setup
Power system operators need day-ahead forecasts for:
- **Load demand**: Electricity consumption at each location
- **Wind generation**: Output from wind farms
- **Solar generation**: Output from solar panels

These quantities are highly uncertain and correlated across space and time. Operators need not just point forecasts but probability distributions and realistic scenarios for optimization and risk assessment.

### Challenge: High Dimensionality
Modern power systems have:
- Hundreds of renewable generators
- Dozens of time periods (48 hours ahead)
- **Total dimension**: 10,000+ variables
- **Correlation matrix**: 100+ million coefficients

Traditional probabilistic forecasting methods fail at this scale.

### Solution Framework

**1. Weather-Informed Temporal Fusion Transformer (WI-TFT)**

**Architecture**:
- **Encoder**: Processes historical time series + weather covariates
- **Attention mechanism**: Learns temporal dependencies
- **Decoder**: Produces quantile forecasts for each time step

**Key innovation**: Incorporates weather information as covariates:
- Temperature, humidity, wind speed, cloud cover
- Weather forecasts for prediction horizon
- Spatial weather patterns across region

**Output**: For each variable and time step, predict quantiles (10%, 25%, 50%, 75%, 90%)

**2. Gaussian Copula for Scenario Generation**

Problem: Quantile forecasts don't capture correlations between variables.

**Solution**: Use Gaussian copula to restore spatio-temporal correlations:

**Step 1: Transform to uniform marginals**
```
For each variable i at time t:
  u_it = CDF_it(forecast_it)
```

**Step 2: Estimate correlation structure**
```
Compute correlation matrix Σ from historical data:
  Σ_ij = correlation between variables i and j
```

**Step 3: Generate correlated scenarios**
```
1. Sample z ~ N(0, Σ)  (multivariate normal)
2. Transform: u = Φ(z)  (to uniform marginals)
3. Invert: x = CDF^(-1)(u)  (to original scale)
```

This produces scenarios that:
- Match marginal distributions from WI-TFT
- Preserve spatial correlations (between locations)
- Preserve temporal correlations (across time)

**3. Validation Metrics**

**Forecast quality**:
- Pinball loss (quantile accuracy)
- Continuous Ranked Probability Score (CRPS)
- Calibration: Do 90% intervals contain 90% of observations?

**Scenario quality**:
- Energy score (multivariate calibration)
- Variogram score (spatial correlation accuracy)
- Temporal correlation preservation

### Key Findings

**Impact of weather information**:
- Including weather covariates improves forecast accuracy by 15-25%
- Most critical for solar (cloud cover) and wind (wind speed forecasts)
- Temperature important for load forecasting

**Copula effectiveness**:
- Gaussian copula accurately captures correlations
- Generated scenarios pass statistical tests for realism
- Enables downstream stochastic optimization

**Model comparison**:
- WI-TFT outperforms: ARIMA, DeepAR, standard TFT
- Attention mechanism crucial for long-horizon forecasting
- Quantile regression better than parametric distributions

### Practical Impact
Provides reliable probabilistic forecasts and scenarios for:
- **Unit commitment**: Decide which generators to start
- **Economic dispatch**: Allocate generation to meet demand
- **Reserve scheduling**: Set aside capacity for uncertainty
- **Risk assessment**: Quantify probability of constraint violations

The weather-informed approach is especially critical as renewable penetration increases, since renewable output depends directly on weather conditions.

---

## Common Themes Across Papers

### 1. Integration of Machine Learning and Optimization
All papers demonstrate the power of combining ML predictions with optimization models:
- **PDL-SCOPF**: Neural networks learn to solve optimization problems
- **Load planning**: ML predicts solutions, optimization repairs them
- **Order fulfillment**: ML forecasts uncertainty, optimization makes decisions
- **Power forecasting**: ML predicts distributions, copula generates scenarios

### 2. Handling Uncertainty
Multiple approaches to uncertainty:
- **Stochastic optimization**: Optimize expected value over scenarios
- **Robust optimization**: Ensure feasibility in worst-case scenarios
- **Probabilistic forecasting**: Quantify uncertainty explicitly
- **Risk measures**: CVaR, chance constraints

### 3. Computational Scalability
All papers address large-scale problems requiring fast solutions:
- **Decomposition**: Benders, column generation
- **Learning**: Neural networks for fast approximation
- **Heuristics**: Conflict-based search, local search
- **Hybrid methods**: ML + optimization repair

### 4. Practical Deployment Considerations
Beyond algorithmic performance:
- **Consistency**: Solutions should be stable across time
- **Explainability**: Decisions must be interpretable
- **Trust**: Methods must be reliable for adoption
- **Real-time**: Millisecond to second response times

### 5. Domain-Specific Modeling
Each application requires careful modeling of domain constraints:
- **Evacuation**: Contraflows, convergence, non-preemption
- **Load planning**: Symmetry breaking, reference plans
- **Power systems**: N-1 security, automatic primary response
- **Order fulfillment**: Consolidation, multi-courier options
- **Forecasting**: Spatio-temporal correlations, weather dependence

---

## Methodological Contributions

### Optimization Techniques
1. **Lexicographical optimization** for symmetry breaking
2. **Self-supervised learning** for optimization without training data
3. **Contextual stochastic optimization** integrating ML forecasts
4. **Differentiable optimization layers** for end-to-end learning
5. **Gaussian copula** for high-dimensional scenario generation

### Machine Learning Advances
1. **Quantile regression** for uncertainty quantification
2. **Temporal Fusion Transformer** with weather covariates
3. **Primal-dual networks** mimicking optimization algorithms
4. **Optimization proxies** combining prediction and repair
5. **Attention mechanisms** for long-horizon forecasting

### Hybrid Approaches
1. **Predict-then-optimize**: ML prediction → optimization decision
2. **Optimize-then-learn**: Optimization generates training data
3. **End-to-end learning**: Backpropagate through optimization
4. **Iterative refinement**: ML initialization + optimization repair
5. **Distributional forecasting**: ML predicts distributions for stochastic optimization

---

## Future Research Directions

Based on these papers, promising directions include:

1. **Generalization**: Transfer learning across problem instances and domains
2. **Online learning**: Adapt models during operations based on feedback
3. **Explainability**: Interpret learned optimization strategies
4. **Theoretical guarantees**: Combine learning with provable bounds
5. **Multi-stage problems**: Extend to sequential decision-making
6. **Fairness**: Incorporate equity considerations in optimization
7. **Robustness**: Handle distribution shift and adversarial scenarios
8. **Scalability**: Push to even larger problem sizes (millions of variables)
