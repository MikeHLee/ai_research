# Graph Traversal and Optimization

## Overview

Graph traversal problems involve finding paths, cycles, or structures in graphs subject to various constraints. These problems are fundamental to network optimization, routing, and scheduling.

## Basic Graph Representations

```python
class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj_list = {i: [] for i in range(n_vertices)}
        self.edges = []
    
    def add_edge(self, u, v, weight=1):
        """Add edge from u to v with optional weight"""
        self.adj_list[u].append((v, weight))
        self.edges.append((u, v, weight))
        
        if not self.directed:
            self.adj_list[v].append((u, weight))
    
    def get_neighbors(self, u):
        """Get neighbors of vertex u"""
        return self.adj_list[u]
```

## Classic Traversal Algorithms

### 1. Depth-First Search (DFS)

```python
def dfs(graph, start, visited=None):
    """
    Depth-first search traversal
    Time: O(V + E)
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(f"Visiting: {start}")
    
    for neighbor, _ in graph.get_neighbors(start):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited


def dfs_iterative(graph, start):
    """
    Iterative DFS using stack
    """
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            print(f"Visiting: {vertex}")
            
            # Add neighbors to stack
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited
```

### 2. Breadth-First Search (BFS)

```python
from collections import deque

def bfs(graph, start):
    """
    Breadth-first search traversal
    Time: O(V + E)
    """
    visited = set([start])
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        print(f"Visiting: {vertex}")
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited


def bfs_shortest_path(graph, start, end):
    """
    Find shortest path using BFS (unweighted graph)
    """
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex == end:
            return path
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

## Shortest Path Problems

### 1. Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm for single-source shortest paths
    Time: O((V + E) log V) with binary heap
    
    Returns: distances and predecessors
    """
    distances = {i: float('inf') for i in range(graph.n)}
    distances[start] = 0
    predecessors = {i: None for i in range(graph.n)}
    
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        
        # Relax edges
        for v, weight in graph.get_neighbors(u):
            if v in visited:
                continue
            
            new_dist = current_dist + weight
            
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    return distances, predecessors


def reconstruct_path(predecessors, start, end):
    """Reconstruct path from predecessors"""
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    path.reverse()
    
    if path[0] == start:
        return path
    else:
        return None  # No path exists
```

### 2. A* Search

```python
def a_star(graph, start, goal, heuristic):
    """
    A* algorithm with heuristic function
    Time: O(E) in best case, O(b^d) in worst case
    
    heuristic: function that estimates distance to goal
    """
    # g_score: actual distance from start
    g_score = {i: float('inf') for i in range(graph.n)}
    g_score[start] = 0
    
    # f_score: g_score + heuristic
    f_score = {i: float('inf') for i in range(graph.n)}
    f_score[start] = heuristic(start, goal)
    
    # Priority queue: (f_score, vertex)
    open_set = [(f_score[start], start)]
    came_from = {}
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + weight
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found


# Example heuristic for grid graphs (Manhattan distance)
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### 3. Bellman-Ford Algorithm

```python
def bellman_ford(graph, start):
    """
    Bellman-Ford algorithm (handles negative weights)
    Time: O(VE)
    
    Returns: distances, predecessors, or None if negative cycle exists
    """
    distances = {i: float('inf') for i in range(graph.n)}
    distances[start] = 0
    predecessors = {i: None for i in range(graph.n)}
    
    # Relax edges V-1 times
    for _ in range(graph.n - 1):
        for u, v, weight in graph.edges:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
    
    # Check for negative cycles
    for u, v, weight in graph.edges:
        if distances[u] + weight < distances[v]:
            return None  # Negative cycle detected
    
    return distances, predecessors
```

### 4. Floyd-Warshall Algorithm

```python
def floyd_warshall(graph):
    """
    All-pairs shortest paths
    Time: O(V³)
    """
    n = graph.n
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Distance from vertex to itself is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Add edges
    for u, v, weight in graph.edges:
        dist[u][v] = weight
        if not graph.directed:
            dist[v][u] = weight
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist
```

## Constrained Path Problems

### 1. Resource-Constrained Shortest Path

```python
def resource_constrained_shortest_path(graph, start, end, max_resource):
    """
    Find shortest path with resource constraint
    Example: shortest path with limited fuel/time/cost
    
    State: (vertex, resource_used)
    """
    # Priority queue: (distance, vertex, resource_used, path)
    pq = [(0, start, 0, [start])]
    
    # Best distance for each (vertex, resource) state
    best = {}
    
    while pq:
        dist, u, resource, path = heapq.heappop(pq)
        
        if u == end:
            return path, dist
        
        state = (u, resource)
        if state in best and best[state] <= dist:
            continue
        best[state] = dist
        
        for v, (edge_cost, edge_resource) in graph.get_neighbors(u):
            new_resource = resource + edge_resource
            
            if new_resource <= max_resource:
                new_dist = dist + edge_cost
                new_path = path + [v]
                heapq.heappush(pq, (new_dist, v, new_resource, new_path))
    
    return None, float('inf')  # No feasible path
```

### 2. Time-Dependent Shortest Path

```python
def time_dependent_shortest_path(graph, start, end, start_time):
    """
    Shortest path where edge costs depend on time
    Example: traffic-aware routing
    """
    # Priority queue: (arrival_time, vertex, path)
    pq = [(start_time, start, [start])]
    best_arrival = {start: start_time}
    
    while pq:
        arrival_time, u, path = heapq.heappop(pq)
        
        if u == end:
            return path, arrival_time
        
        if arrival_time > best_arrival.get(u, float('inf')):
            continue
        
        for v, travel_time_func in graph.get_neighbors(u):
            # Travel time depends on departure time
            travel_time = travel_time_func(arrival_time)
            new_arrival = arrival_time + travel_time
            
            if new_arrival < best_arrival.get(v, float('inf')):
                best_arrival[v] = new_arrival
                heapq.heappush(pq, (new_arrival, v, path + [v]))
    
    return None, float('inf')
```

### 3. k-Shortest Paths

```python
def k_shortest_paths(graph, start, end, k):
    """
    Find k shortest paths using Yen's algorithm
    Time: O(kV(E + V log V))
    """
    # Find first shortest path
    distances, predecessors = dijkstra(graph, start)
    first_path = reconstruct_path(predecessors, start, end)
    
    if first_path is None:
        return []
    
    paths = [first_path]
    candidates = []
    
    for k_i in range(1, k):
        prev_path = paths[-1]
        
        # For each node in previous path
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i+1]
            
            # Remove edges that would create duplicate paths
            removed_edges = []
            for path in paths:
                if len(path) > i and path[:i+1] == root_path:
                    u, v = path[i], path[i+1]
                    if graph.has_edge(u, v):
                        removed_edges.append((u, v))
                        graph.remove_edge(u, v)
            
            # Find shortest path from spur node to end
            distances, predecessors = dijkstra(graph, spur_node)
            spur_path = reconstruct_path(predecessors, spur_node, end)
            
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                candidates.append(total_path)
            
            # Restore removed edges
            for u, v in removed_edges:
                graph.add_edge(u, v)
        
        if not candidates:
            break
        
        # Select shortest candidate
        candidates.sort(key=lambda p: path_length(graph, p))
        paths.append(candidates.pop(0))
    
    return paths
```

## Cycle Detection and Analysis

### 1. Detect Cycle (DFS-based)

```python
def has_cycle_directed(graph):
    """
    Detect cycle in directed graph using DFS
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in range(graph.n)}
    
    def visit(u):
        color[u] = GRAY
        
        for v, _ in graph.get_neighbors(u):
            if color[v] == GRAY:
                return True  # Back edge found (cycle)
            if color[v] == WHITE and visit(v):
                return True
        
        color[u] = BLACK
        return False
    
    for vertex in range(graph.n):
        if color[vertex] == WHITE:
            if visit(vertex):
                return True
    
    return False


def has_cycle_undirected(graph):
    """
    Detect cycle in undirected graph
    """
    visited = set()
    
    def visit(u, parent):
        visited.add(u)
        
        for v, _ in graph.get_neighbors(u):
            if v not in visited:
                if visit(v, u):
                    return True
            elif v != parent:
                return True  # Cycle found
        
        return False
    
    for vertex in range(graph.n):
        if vertex not in visited:
            if visit(vertex, None):
                return True
    
    return False
```

### 2. Find All Cycles

```python
def find_all_cycles(graph):
    """
    Find all simple cycles in directed graph
    Using Johnson's algorithm
    """
    def find_cycles_from(start, current, path, blocked, stack):
        cycles = []
        path.append(current)
        blocked.add(current)
        
        for neighbor, _ in graph.get_neighbors(current):
            if neighbor == start:
                # Found cycle
                cycles.append(path[:])
            elif neighbor not in blocked:
                cycles.extend(find_cycles_from(
                    start, neighbor, path, blocked, stack
                ))
        
        if cycles:
            # Unblock vertices
            unblock(current, blocked, stack)
        else:
            # Add to stack for later unblocking
            for neighbor, _ in graph.get_neighbors(current):
                if current not in stack[neighbor]:
                    stack[neighbor].add(current)
        
        path.pop()
        return cycles
    
    def unblock(vertex, blocked, stack):
        blocked.remove(vertex)
        for w in stack[vertex]:
            if w in blocked:
                unblock(w, blocked, stack)
        stack[vertex].clear()
    
    all_cycles = []
    
    for start in range(graph.n):
        blocked = set()
        stack = {i: set() for i in range(graph.n)}
        cycles = find_cycles_from(start, start, [], blocked, stack)
        all_cycles.extend(cycles)
    
    return all_cycles
```

## Minimum Spanning Tree

### 1. Kruskal's Algorithm

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal_mst(graph):
    """
    Kruskal's algorithm for minimum spanning tree
    Time: O(E log E)
    """
    # Sort edges by weight
    edges = sorted(graph.edges, key=lambda e: e[2])
    
    uf = UnionFind(graph.n)
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == graph.n - 1:
                break
    
    return mst, total_weight
```

### 2. Prim's Algorithm

```python
def prim_mst(graph, start=0):
    """
    Prim's algorithm for minimum spanning tree
    Time: O(E log V)
    """
    mst = []
    total_weight = 0
    visited = {start}
    
    # Priority queue: (weight, u, v)
    edges = [(weight, start, v) for v, weight in graph.get_neighbors(start)]
    heapq.heapify(edges)
    
    while edges and len(visited) < graph.n:
        weight, u, v = heapq.heappop(edges)
        
        if v in visited:
            continue
        
        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight
        
        # Add new edges
        for next_v, next_weight in graph.get_neighbors(v):
            if next_v not in visited:
                heapq.heappush(edges, (next_weight, v, next_v))
    
    return mst, total_weight
```

## Network Flow Problems

### 1. Maximum Flow (Ford-Fulkerson)

```python
def max_flow(graph, source, sink):
    """
    Maximum flow using Ford-Fulkerson with BFS (Edmonds-Karp)
    Time: O(VE²)
    """
    # Create residual graph
    residual = {i: {} for i in range(graph.n)}
    for u, v, capacity in graph.edges:
        residual[u][v] = capacity
        if v not in residual or u not in residual[v]:
            residual[v][u] = 0
    
    def bfs_find_path():
        """Find augmenting path using BFS"""
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            u, path = queue.popleft()
            
            if u == sink:
                return path
            
            for v in residual[u]:
                if v not in visited and residual[u][v] > 0:
                    visited.add(v)
                    queue.append((v, path + [v]))
        
        return None
    
    max_flow_value = 0
    
    while True:
        path = bfs_find_path()
        if path is None:
            break
        
        # Find minimum capacity along path
        flow = min(residual[path[i]][path[i+1]] 
                  for i in range(len(path) - 1))
        
        # Update residual graph
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            residual[u][v] -= flow
            residual[v][u] += flow
        
        max_flow_value += flow
    
    return max_flow_value
```

## Applications

### 1. Traveling Salesman Problem (TSP)

```python
def tsp_dynamic_programming(distances):
    """
    TSP using dynamic programming (Held-Karp)
    Time: O(n² 2ⁿ)
    """
    n = len(distances)
    
    # dp[mask][i] = minimum cost to visit cities in mask, ending at i
    dp = {}
    
    # Base case: start from city 0
    for i in range(1, n):
        dp[(1 << i, i)] = distances[0][i]
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            
            prev_mask = mask ^ (1 << last)
            
            for prev in range(n):
                if not (prev_mask & (1 << prev)):
                    continue
                
                if (prev_mask, prev) in dp:
                    cost = dp[(prev_mask, prev)] + distances[prev][last]
                    
                    if (mask, last) not in dp or cost < dp[(mask, last)]:
                        dp[(mask, last)] = cost
    
    # Find minimum cost to return to start
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    
    for last in range(1, n):
        if (full_mask, last) in dp:
            cost = dp[(full_mask, last)] + distances[last][0]
            min_cost = min(min_cost, cost)
    
    return min_cost
```

### 2. Vehicle Routing with Time Windows

```python
def vrp_with_time_windows(customers, vehicles, depot):
    """
    Vehicle routing with time windows constraint
    Each customer must be visited within their time window
    """
    # This is typically solved using:
    # - Column generation
    # - Branch and price
    # - Metaheuristics (genetic algorithms, simulated annealing)
    pass
```

## Summary

Graph traversal and optimization encompasses:
- **Basic traversal**: DFS, BFS
- **Shortest paths**: Dijkstra, A*, Bellman-Ford, Floyd-Warshall
- **Constrained paths**: Resource limits, time-dependent, k-shortest
- **Cycles**: Detection, enumeration
- **Spanning trees**: Kruskal, Prim
- **Network flows**: Max flow, min cost flow
- **Complex routing**: TSP, VRP variants
