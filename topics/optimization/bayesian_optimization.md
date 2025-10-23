# Bayesian Optimization

## Overview

Bayesian Optimization (BO) is a sequential design strategy for global optimization of black-box functions that are expensive to evaluate. It's particularly useful when:
- Function evaluations are costly (time, money, resources)
- Derivatives are unavailable
- Function is noisy
- Search space is continuous and relatively low-dimensional

## Key Concepts

### Surrogate Model
A probabilistic model (typically Gaussian Process) that approximates the true objective function based on observed data.

### Acquisition Function
A function that determines where to sample next by balancing:
- **Exploitation**: Sample where model predicts high values
- **Exploration**: Sample where uncertainty is high

### Gaussian Process (GP)
A distribution over functions specified by:
- **Mean function**: μ(x)
- **Covariance function (kernel)**: k(x, x')

## Algorithm

```
1. Initialize:
   - Sample initial points (random or space-filling)
   - Evaluate objective function at these points
   - Fit surrogate model (GP)

2. For iteration = 1 to max_iterations:
   a. Fit/update GP to all observed data
   b. Optimize acquisition function to find next point
   c. Evaluate objective at next point
   d. Update dataset with new observation

3. Return best observed point
```

## Acquisition Functions

### 1. Expected Improvement (EI)
```python
def expected_improvement(x, gp, best_y, xi=0.01):
    """
    Expected improvement acquisition function
    
    x: candidate point
    gp: Gaussian Process model
    best_y: best observed value so far
    xi: exploration parameter
    """
    mu, sigma = gp.predict(x, return_std=True)
    
    with np.errstate(divide='warn'):
        improvement = mu - best_y - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei
```

### 2. Upper Confidence Bound (UCB)
```python
def upper_confidence_bound(x, gp, kappa=2.0):
    """
    Upper confidence bound acquisition function
    
    x: candidate point
    gp: Gaussian Process model
    kappa: exploration-exploitation trade-off
    """
    mu, sigma = gp.predict(x, return_std=True)
    return mu + kappa * sigma
```

### 3. Probability of Improvement (PI)
```python
def probability_of_improvement(x, gp, best_y, xi=0.01):
    """
    Probability of improvement acquisition function
    """
    mu, sigma = gp.predict(x, return_std=True)
    
    with np.errstate(divide='warn'):
        Z = (mu - best_y - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0
    
    return pi
```

## Complete Implementation

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer:
    def __init__(self, bounds, acquisition='ei', xi=0.01, kappa=2.0):
        """
        bounds: list of (min, max) tuples for each dimension
        acquisition: 'ei', 'ucb', or 'pi'
        xi: exploration parameter for EI and PI
        kappa: exploration parameter for UCB
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.acquisition_type = acquisition
        self.xi = xi
        self.kappa = kappa
        
        # Gaussian Process with Matern kernel
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.X_observed = []
        self.y_observed = []
    
    def acquisition_function(self, x):
        """Compute acquisition function value"""
        x = x.reshape(1, -1)
        
        if self.acquisition_type == 'ei':
            return -expected_improvement(
                x, self.gp, np.max(self.y_observed), self.xi
            )[0]
        elif self.acquisition_type == 'ucb':
            return -upper_confidence_bound(x, self.gp, self.kappa)[0]
        elif self.acquisition_type == 'pi':
            return -probability_of_improvement(
                x, self.gp, np.max(self.y_observed), self.xi
            )[0]
    
    def propose_location(self):
        """Find next point to sample by optimizing acquisition function"""
        # Multi-start optimization
        best_x = None
        best_acquisition = float('inf')
        
        # Random restarts
        for _ in range(25):
            x0 = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.dim
            )
            
            result = minimize(
                self.acquisition_function,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x
        
        return best_x
    
    def observe(self, x, y):
        """Add observation to dataset"""
        self.X_observed.append(x)
        self.y_observed.append(y)
    
    def optimize(self, objective_function, n_iterations=50, n_initial=5):
        """
        Run Bayesian optimization
        
        objective_function: function to maximize
        n_iterations: number of iterations
        n_initial: number of random initial points
        """
        # Initial random sampling
        for _ in range(n_initial):
            x = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.dim
            )
            y = objective_function(x)
            self.observe(x, y)
        
        # Bayesian optimization loop
        for i in range(n_iterations):
            # Fit GP to observed data
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
            
            # Find next point to sample
            x_next = self.propose_location()
            
            # Evaluate objective
            y_next = objective_function(x_next)
            
            # Update observations
            self.observe(x_next, y_next)
            
            # Print progress
            best_y = np.max(self.y_observed)
            print(f"Iteration {i+1}/{n_iterations}: Best value = {best_y:.4f}")
        
        # Return best point found
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]


# Example usage
def objective(x):
    """Example: 2D Branin function (to minimize, so we negate)"""
    x1, x2 = x
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s
    
    return -(term1 + term2 + term3)  # Negate to maximize


# Run optimization
bounds = [(-5, 10), (0, 15)]
optimizer = BayesianOptimizer(bounds, acquisition='ei')
best_x, best_y = optimizer.optimize(objective, n_iterations=30)

print(f"\nBest point: {best_x}")
print(f"Best value: {best_y}")
```

## Variants and Extensions

### 1. Constrained Bayesian Optimization
Handle constraints using separate GP models:
```python
def constrained_acquisition(x, gp_objective, gp_constraint, threshold):
    """
    Acquisition that accounts for constraints
    """
    # Probability that constraint is satisfied
    mu_c, sigma_c = gp_constraint.predict(x, return_std=True)
    prob_feasible = norm.cdf((threshold - mu_c) / sigma_c)
    
    # Standard acquisition for objective
    acq_objective = expected_improvement(x, gp_objective, best_y)
    
    # Combined acquisition
    return acq_objective * prob_feasible
```

### 2. Multi-Objective Bayesian Optimization
Optimize multiple objectives simultaneously using Pareto frontiers.

### 3. Batch Bayesian Optimization
Select multiple points to evaluate in parallel:
- **Local penalization**: Penalize acquisition near previously selected points
- **Kriging believer**: Update GP with hallucinated observations
- **Thompson sampling**: Sample from GP posterior

### 4. High-Dimensional BO
- **Random embeddings**: Project to lower dimensions
- **Additive models**: Assume function decomposes
- **Trust regions**: Local optimization in subspaces

## Hyperparameter Tuning Example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective_ml(params):
    """
    Objective function for hyperparameter tuning
    params: [n_estimators, max_depth, min_samples_split]
    """
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Cross-validation score
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()

# Define search space
bounds = [
    (10, 200),    # n_estimators
    (2, 20),      # max_depth
    (2, 20)       # min_samples_split
]

# Optimize
optimizer = BayesianOptimizer(bounds, acquisition='ei')
best_params, best_score = optimizer.optimize(objective_ml, n_iterations=50)

print(f"Best hyperparameters: {best_params}")
print(f"Best CV score: {best_score}")
```

## Advantages

- **Sample efficient**: Finds good solutions with few evaluations
- **Handles noise**: GP naturally models uncertainty
- **No derivatives needed**: Black-box optimization
- **Principled exploration**: Balances exploration and exploitation

## Disadvantages

- **Scalability**: GP inference is O(n³) in number of observations
- **Dimensionality**: Performance degrades in high dimensions (>20)
- **Assumptions**: Assumes smoothness (encoded in kernel)
- **Local optima**: Acquisition optimization can get stuck

## Applications

- **Hyperparameter Tuning**: ML model optimization
- **A/B Testing**: Website/product optimization
- **Robotics**: Policy search, controller tuning
- **Materials Science**: Experimental design
- **Drug Discovery**: Molecule optimization
- **Engineering**: Design optimization (aerodynamics, structures)
- **AutoML**: Neural architecture search

## Popular Libraries

- **Scikit-Optimize**: General-purpose BO
- **GPyOpt**: Flexible BO framework
- **Optuna**: Hyperparameter optimization
- **Ax/BoTorch**: Facebook's BO platform (PyTorch-based)
- **Hyperopt**: Tree-structured Parzen estimator
- **SMAC**: Sequential Model-based Algorithm Configuration
