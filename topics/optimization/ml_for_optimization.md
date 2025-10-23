# Machine Learning for Optimization

## Overview

Machine learning is increasingly used to improve optimization algorithms by learning from data to make better decisions during the solution process. This includes learning heuristics, branching strategies, cut selection, and even end-to-end solution methods.

## Key Applications

### 1. Learning to Branch
Use ML to predict good branching decisions in branch-and-bound.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LearnedBranchingStrategy:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.features = []
        self.labels = []
    
    def extract_features(self, variable, lp_solution, node_info):
        """
        Extract features for branching decision
        
        Features commonly include:
        - Fractionality: distance from integer
        - Objective coefficient
        - Number of constraints variable appears in
        - Pseudocost: historical impact of branching
        - LP value and bounds
        - Node depth in B&B tree
        """
        features = [
            abs(lp_solution[variable] - round(lp_solution[variable])),  # fractionality
            node_info['objective_coef'][variable],
            node_info['constraint_count'][variable],
            node_info['pseudocost_up'][variable],
            node_info['pseudocost_down'][variable],
            lp_solution[variable],
            node_info['depth']
        ]
        return np.array(features)
    
    def select_branching_variable(self, fractional_vars, lp_solution, node_info):
        """
        Select variable to branch on using learned model
        """
        if not self.model:
            # Fallback to most fractional
            return max(fractional_vars, 
                      key=lambda v: abs(lp_solution[v] - 0.5))
        
        # Extract features for all candidates
        features_list = [
            self.extract_features(v, lp_solution, node_info)
            for v in fractional_vars
        ]
        
        # Predict quality scores
        scores = self.model.predict_proba(features_list)[:, 1]
        
        # Select variable with highest score
        best_idx = np.argmax(scores)
        return fractional_vars[best_idx]
    
    def collect_training_data(self, variable, features, outcome):
        """
        Collect data for training
        outcome: 1 if good branch, 0 if bad (based on tree size reduction)
        """
        self.features.append(features)
        self.labels.append(outcome)
    
    def train(self):
        """Train the branching model"""
        X = np.array(self.features)
        y = np.array(self.labels)
        self.model.fit(X, y)
```

### 2. Learning Cut Selection
Predict which cutting planes will be most effective.

```python
class LearnedCutSelector:
    def __init__(self):
        self.model = None  # Could be neural network, gradient boosting, etc.
    
    def extract_cut_features(self, cut, lp_solution, problem_info):
        """
        Features for cut selection:
        - Efficacy: violation / norm
        - Support: number of non-zeros
        - Parallelism: similarity to existing cuts
        - Expected improvement in bound
        - Cut density
        """
        violation = max(0, np.dot(cut['coeffs'], lp_solution) - cut['rhs'])
        norm = np.linalg.norm(cut['coeffs'])
        efficacy = violation / norm if norm > 0 else 0
        
        support = np.count_nonzero(cut['coeffs'])
        density = support / len(cut['coeffs'])
        
        # Parallelism with existing cuts
        parallelism = max([
            abs(np.dot(cut['coeffs'], existing['coeffs'])) / 
            (np.linalg.norm(cut['coeffs']) * np.linalg.norm(existing['coeffs']))
            for existing in problem_info['active_cuts']
        ]) if problem_info['active_cuts'] else 0
        
        features = [
            efficacy,
            support,
            density,
            parallelism,
            violation,
            problem_info['num_iterations']
        ]
        
        return np.array(features)
    
    def select_cuts(self, candidate_cuts, lp_solution, problem_info, max_cuts=10):
        """
        Select subset of cuts to add
        """
        # Extract features for all candidates
        features_list = [
            self.extract_cut_features(cut, lp_solution, problem_info)
            for cut in candidate_cuts
        ]
        
        # Predict quality scores
        scores = self.model.predict(features_list)
        
        # Select top-k cuts
        top_indices = np.argsort(scores)[-max_cuts:]
        return [candidate_cuts[i] for i in top_indices]
```

### 3. Learning Heuristics
Use ML to guide construction heuristics or local search.

```python
import torch
import torch.nn as nn

class HeuristicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class LearnedConstructionHeuristic:
    """
    Learn to construct solutions for combinatorial problems
    Example: TSP, knapsack, scheduling
    """
    def __init__(self, feature_dim):
        self.model = HeuristicNetwork(feature_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def construct_solution(self, problem_instance):
        """
        Iteratively build solution by selecting items/actions
        """
        solution = []
        available_items = list(range(problem_instance.n_items))
        state = problem_instance.initial_state()
        
        while available_items and not state.is_terminal():
            # Extract features for each available item
            features = torch.tensor([
                self.extract_features(item, state, problem_instance)
                for item in available_items
            ], dtype=torch.float32)
            
            # Predict selection probabilities
            with torch.no_grad():
                scores = self.model(features).squeeze()
            
            # Select item (greedy or sampling)
            selected_idx = torch.argmax(scores).item()
            selected_item = available_items[selected_idx]
            
            # Update solution and state
            solution.append(selected_item)
            state = state.add_item(selected_item)
            available_items.remove(selected_item)
        
        return solution
    
    def train_from_expert_solutions(self, training_data):
        """
        Imitation learning: learn from expert solutions
        """
        for instance, expert_solution in training_data:
            state = instance.initial_state()
            
            for step, expert_action in enumerate(expert_solution):
                # Get available actions
                available = state.get_available_actions()
                
                # Extract features
                features = torch.tensor([
                    self.extract_features(action, state, instance)
                    for action in available
                ], dtype=torch.float32)
                
                # Predict scores
                scores = self.model(features).squeeze()
                
                # Create target: 1 for expert action, 0 for others
                target = torch.zeros(len(available))
                expert_idx = available.index(expert_action)
                target[expert_idx] = 1.0
                
                # Loss: cross-entropy
                loss = nn.BCELoss()(scores, target)
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update state
                state = state.add_action(expert_action)
```

### 4. Graph Neural Networks for Combinatorial Optimization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
    
    def forward(self, x, adj):
        """
        x: node features [n_nodes, in_features]
        adj: adjacency matrix [n_nodes, n_nodes]
        """
        h = self.W(x)  # [n_nodes, out_features]
        n = h.size(0)
        
        # Compute attention coefficients
        h_i = h.repeat(1, n).view(n * n, -1)
        h_j = h.repeat(n, 1)
        attention_input = torch.cat([h_i, h_j], dim=1)
        e = F.leaky_relu(self.a(attention_input).view(n, n))
        
        # Mask attention for non-edges
        e = e.masked_fill(adj == 0, float('-inf'))
        
        # Softmax attention
        alpha = F.softmax(e, dim=1)
        
        # Aggregate neighbors
        h_prime = torch.matmul(alpha, h)
        
        return h_prime


class GNNOptimizer(nn.Module):
    """
    Graph Neural Network for optimization problems
    Example: TSP, graph coloring, max-cut
    """
    def __init__(self, node_features, hidden_dim=64, num_layers=3):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                node_features if i == 0 else hidden_dim,
                hidden_dim
            )
            for i in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, node_features, adjacency):
        """
        Predict node scores for solution construction
        """
        x = node_features
        
        for layer in self.layers:
            x = F.relu(layer(x, adjacency))
        
        scores = self.output(x).squeeze()
        return scores
    
    def solve_tsp(self, cities, distances):
        """
        Construct TSP tour using learned policy
        """
        n = len(cities)
        
        # Create graph representation
        node_features = torch.tensor(cities, dtype=torch.float32)
        adjacency = torch.ones(n, n) - torch.eye(n)  # Fully connected
        
        tour = []
        available = set(range(n))
        current = 0  # Start from city 0
        tour.append(current)
        available.remove(current)
        
        while available:
            # Mask for available cities
            mask = torch.zeros(n)
            for city in available:
                mask[city] = 1.0
            
            # Predict scores
            with torch.no_grad():
                scores = self.forward(node_features, adjacency)
                scores = scores * mask + (1 - mask) * float('-inf')
            
            # Select next city
            next_city = torch.argmax(scores).item()
            tour.append(next_city)
            available.remove(next_city)
            
            # Update features (encode current position)
            node_features[current, -1] = 0  # Mark as visited
            current = next_city
        
        return tour
```

### 5. Reinforcement Learning for Optimization

```python
class RLOptimizationAgent:
    """
    Use RL to learn optimization strategies
    Example: Learning variable selection, cut selection, etc.
    """
    def __init__(self, state_dim, action_dim):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters())
        )
    
    def select_action(self, state, available_actions):
        """
        Select action using policy network
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_network(state_tensor)
        
        # Mask unavailable actions
        mask = torch.zeros(len(action_probs))
        mask[available_actions] = 1.0
        masked_probs = action_probs * mask
        masked_probs = masked_probs / masked_probs.sum()
        
        # Sample action
        action = torch.multinomial(masked_probs, 1).item()
        return action, masked_probs[action]
    
    def train_episode(self, states, actions, rewards):
        """
        Train using policy gradient (REINFORCE or A2C)
        """
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G  # discount factor
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = 0
        value_loss = 0
        
        for state, action, G in zip(states, actions, returns):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Value prediction
            value = self.value_network(state_tensor)
            advantage = G - value.item()
            
            # Policy loss
            action_probs = self.policy_network(state_tensor)
            log_prob = torch.log(action_probs[action])
            policy_loss -= log_prob * advantage
            
            # Value loss
            value_loss += F.mse_loss(value, torch.tensor([G]))
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Hybrid Approaches

### 1. Predict-and-Optimize
Learn to predict problem parameters, then solve optimization problem.

### 2. Learning-Augmented Algorithms
Use ML predictions to guide traditional algorithms with theoretical guarantees.

### 3. Neural Combinatorial Optimization
End-to-end learning of solution methods using deep learning.

## Advantages

- **Adaptivity**: Learn from problem-specific patterns
- **Speed**: Fast inference after training
- **Generalization**: Transfer learning across similar problems
- **Automation**: Reduce need for manual tuning

## Disadvantages

- **Training data**: Requires large datasets
- **Generalization**: May not work on different problem types
- **Guarantees**: No optimality guarantees (usually)
- **Interpretability**: Black-box decisions

## Applications

- **MIP Solving**: Branching, cutting, primal heuristics
- **Routing**: Vehicle routing, TSP
- **Scheduling**: Job shop, flow shop
- **Packing**: Bin packing, knapsack
- **Graph Problems**: Coloring, partitioning, max-cut
- **Resource Allocation**: Cloud computing, network design

## Research Directions

- **Generalization**: Transfer across problem instances and types
- **Sample Efficiency**: Learn from fewer examples
- **Theoretical Guarantees**: Combine learning with provable bounds
- **Explainability**: Understand learned strategies
- **Online Learning**: Adapt during solving process

## Tools and Libraries

- **SCIP-ML**: ML integration with SCIP solver
- **Ecole**: RL environment for combinatorial optimization
- **OR-Tools**: Google's optimization library
- **PyTorch Geometric**: GNN library
- **Stable-Baselines3**: RL algorithms
