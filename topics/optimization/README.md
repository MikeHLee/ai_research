# Optimization Problems and Methods

This directory contains an overview of common optimization problems and solution methods used in operations research, computer science, and machine learning.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Types](#problem-types)
3. [Solution Methods](#solution-methods)
4. [Applications](#applications)

## Introduction

Optimization is the process of finding the best solution from a set of feasible alternatives. Problems typically involve:
- **Objective function**: What to maximize or minimize
- **Decision variables**: What we can control
- **Constraints**: Limitations on feasible solutions

## Problem Types

### Linear Programming (LP)
- Objective and constraints are linear functions
- Efficiently solvable with simplex or interior-point methods
- Example: Resource allocation, production planning

### Integer Programming (IP)
- Some or all variables must be integers
- NP-hard in general
- Example: Scheduling, facility location

### Mixed-Integer Programming (MIP)
- Combination of continuous and integer variables
- Most flexible but computationally challenging

### Combinatorial Optimization
- Discrete decision variables
- Often involves graph theory
- Example: TSP, matching problems

## Solution Methods

See individual files for detailed explanations:
- [Branch and Bound](./branch_and_bound.md)
- [Cutting Planes](./cutting_planes.md)
- [Column Generation](./column_generation.md)
- [Bayesian Optimization](./bayesian_optimization.md)
- [Machine Learning for Optimization](./ml_for_optimization.md)
- [Set Coverage Problems](./set_coverage.md)
- [SAT Solving](./sat_solving.md)
- [Graph Traversal](./graph_traversal.md)

## Applications

- **Supply Chain**: Routing, inventory management
- **Finance**: Portfolio optimization, risk management
- **Healthcare**: Staff scheduling, treatment planning
- **Manufacturing**: Production scheduling, quality control
- **Machine Learning**: Hyperparameter tuning, neural architecture search
