"""
Simulated Annealing Solver for Water Network Sensor Placement
Solves the same optimization problem as the AMPL model using metaheuristic
"""

import numpy as np
import random
import math
import json
from typing import Dict, List, Tuple, Set
import time


class SensorPlacementSA:
    """
    Simulated Annealing solver for sensor placement optimization
    
    Minimizes: sum_{i,p,j} alpha[i,p] * c[i,p,j] * delta[j,p]
    
    Subject to:
    - c[i,p,i] = 1 (attack node is contaminated)
    - s[i,j] = s[j,i] (sensor symmetry)
    - c[i,p,j] >= c[i,p,k] - s[k,j] for flow from k to j (contamination propagation)
    - sum s[i,j] <= S_max (sensor budget)
    """
    
    def __init__(self, data_file: str = '../simulation/data/network_data.dat'):
        """Load problem data from AMPL .dat file"""
        print("Loading problem data...")
        self.load_data(data_file)
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Attack nodes: {self.num_attack_nodes}")
        print(f"  - Edges: {self.num_edges}")
        print(f"  - Patterns: {self.num_patterns}")
        
    def load_data(self, data_file: str):
        """Parse AMPL .dat file and extract problem data"""
        with open(data_file, 'r') as f:
            content = f.read()
        
        # Extract scalar parameters
        self.num_nodes = int(self._extract_param(content, 'num_nodes'))
        self.num_attack_nodes = int(self._extract_param(content, 'num_attack_nodes'))
        self.num_edges = int(self._extract_param(content, 'num_edges'))
        self.num_patterns = int(self._extract_param(content, 'num_patterns'))
        self.S_max = int(self._extract_param(content, 'S_max', default=5))
        
        # Extract sets
        self.attack_nodes = self._extract_set(content, 'ATTACK_NODES')
        self.edges = self._extract_edges(content)
        
        # Extract matrices
        self.alpha = self._extract_matrix(content, 'alpha', (self.num_attack_nodes, self.num_patterns))
        self.delta = self._extract_matrix(content, 'delta', (self.num_nodes, self.num_patterns))
        self.flow = self._extract_flow_matrix(content)
        
        # Extract node names
        self.node_names = self._extract_node_names(content)
        
        # Build edge index mapping
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        
        # Build adjacency for contamination propagation
        self._build_flow_graph()
        
    def _extract_param(self, content: str, param_name: str, default=None) -> str:
        """Extract scalar parameter value"""
        import re
        pattern = f'param {param_name} := ([^;]+);'
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
        return default
    
    def _extract_node_names(self, content: str) -> Dict[int, str]:
        """Extract node names mapping"""
        import re
        node_names = {}
        pattern = r'param node_name :=\s*([\s\S]*?);'
        match = re.search(pattern, content)
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    node_idx = int(parts[0])
                    node_name = parts[1]
                    node_names[node_idx] = node_name
        return node_names
        
    def _extract_set(self, content: str, set_name: str) -> List[int]:
        """Extract set elements"""
        import re
        pattern = f'set {set_name} := ([^;]+);'
        match = re.search(pattern, content)
        if match:
            return [int(x) for x in match.group(1).split()]
        return []
        
    def _extract_edges(self, content: str) -> List[Tuple[int, int]]:
        """Extract edge list"""
        import re
        edges = []
        
        # Find edge_from parameter
        pattern = r'param edge_from :=\s*([\s\S]*?);'
        match = re.search(pattern, content)
        if match:
            lines = match.group(1).strip().split('\n')
            edge_from = {}
            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    edge_from[int(parts[0])] = int(parts[1])
        
        # Find edge_to parameter
        pattern = r'param edge_to :=\s*([\s\S]*?);'
        match = re.search(pattern, content)
        if match:
            lines = match.group(1).strip().split('\n')
            edge_to = {}
            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    edge_to[int(parts[0])] = int(parts[1])
        
        # Build edge list
        for idx in sorted(edge_from.keys()):
            edges.append((edge_from[idx], edge_to[idx]))
            
        return edges
        
    def _extract_matrix(self, content: str, param_name: str, shape: Tuple[int, int]) -> np.ndarray:
        """Extract 2D matrix parameter"""
        import re
        pattern = f'param {param_name} :[^:]*:=([^;]+);'
        match = re.search(pattern, content, re.DOTALL)
        
        matrix = np.zeros(shape)
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    row_idx = int(parts[0])
                    if row_idx < shape[0]:
                        values = [float(x) for x in parts[1:]]
                        matrix[row_idx, :len(values)] = values
        
        return matrix
        
    def _extract_flow_matrix(self, content: str) -> np.ndarray:
        """Extract flow matrix"""
        flow = np.zeros((self.num_edges, self.num_patterns))
        
        import re
        pattern = r'param flow :[^:]*:=([^;]+);'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    edge_idx = int(parts[0])
                    if edge_idx < self.num_edges:
                        values = [float(x) for x in parts[1:]]
                        flow[edge_idx, :len(values)] = values
        
        return flow
        
    def _build_flow_graph(self):
        """Build graph structure for contamination propagation"""
        # For each pattern, build adjacency list of edges with flow
        self.flow_graph = []  # flow_graph[p] = list of (from_node, to_node) with flow
        
        for p in range(self.num_patterns):
            pattern_edges = []
            for e, (i, j) in enumerate(self.edges):
                if self.flow[e, p] > 0:
                    pattern_edges.append((i, j))
            self.flow_graph.append(pattern_edges)
    
    def create_initial_solution(self) -> Set[Tuple[int, int]]:
        """Create random initial sensor placement"""
        if self.S_max == 0:
            return set()
        
        # Randomly select S_max edges
        num_sensors = min(self.S_max, len(self.edges))
        selected_edges = random.sample(self.edges, num_sensors)
        
        # Apply symmetry: if (i,j) selected and (j,i) exists, add it
        sensors = set(selected_edges)
        for (i, j) in selected_edges:
            if (j, i) in self.edge_to_idx:
                sensors.add((j, i))
        
        return sensors
    
    def get_neighbor(self, sensors: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Generate neighbor solution by swapping one sensor"""
        new_sensors = sensors.copy()
        
        if random.random() < 0.5 and len(new_sensors) > 0:
            # Remove a random sensor (and its symmetric pair)
            sensor_to_remove = random.choice(list(new_sensors))
            new_sensors.discard(sensor_to_remove)
            reverse = (sensor_to_remove[1], sensor_to_remove[0])
            new_sensors.discard(reverse)
            
            # Add a new random sensor (with symmetry)
            available = [e for e in self.edges if e not in new_sensors]
            if available and len(new_sensors) < self.S_max:
                new_sensor = random.choice(available)
                new_sensors.add(new_sensor)
                reverse = (new_sensor[1], new_sensor[0])
                if reverse in self.edge_to_idx:
                    new_sensors.add(reverse)
        else:
            # Add a new sensor if below budget
            available = [e for e in self.edges if e not in new_sensors]
            if available and len(new_sensors) < self.S_max:
                new_sensor = random.choice(available)
                new_sensors.add(new_sensor)
                reverse = (new_sensor[1], new_sensor[0])
                if reverse in self.edge_to_idx:
                    new_sensors.add(reverse)
        
        return new_sensors
    
    def calculate_contamination(self, sensors: Set[Tuple[int, int]], 
                                attack_node_idx: int, pattern: int) -> np.ndarray:
        """
        Calculate contamination spread for a given attack and pattern
        Uses BFS to propagate contamination through network
        
        Returns: binary array c[j] indicating if node j is contaminated
        """
        attack_node = self.attack_nodes[attack_node_idx]
        c = np.zeros(self.num_nodes)
        c[attack_node] = 1  # Attack node is contaminated
        
        # BFS to propagate contamination
        queue = [attack_node]
        visited = {attack_node}
        
        while queue:
            node = queue.pop(0)
            
            # Check all edges with flow in this pattern
            for (from_node, to_node) in self.flow_graph[pattern]:
                if from_node == node and to_node not in visited:
                    # Check if sensor blocks this edge
                    if (from_node, to_node) not in sensors:
                        # Contamination propagates
                        c[to_node] = 1
                        queue.append(to_node)
                        visited.add(to_node)
        
        return c
    
    def evaluate(self, sensors: Set[Tuple[int, int]]) -> float:
        """
        Calculate objective function value for given sensor placement
        
        Objective: sum_{i,p,j} alpha[i,p] * c[i,p,j] * delta[j,p]
        """
        obj = 0.0
        
        for i in range(self.num_attack_nodes):
            for p in range(self.num_patterns):
                # Calculate contamination for this attack and pattern
                c = self.calculate_contamination(sensors, i, p)
                
                # Vectorized calculation of weighted contamination
                obj += np.sum(self.alpha[i, p] * c * self.delta[:, p])
        
        return obj
    
    def solve(self, 
              initial_temp: float = 100.0,
              cooling_rate: float = 0.95,
              min_temp: float = 0.01,
              iterations_per_temp: int = 100,
              max_time: float = 100.0,
              verbose: bool = True) -> Dict:
        """
        Solve using Simulated Annealing
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            min_temp: Stopping temperature
            iterations_per_temp: Number of iterations at each temperature
            max_time: Maximum time in seconds (default: 100)
            verbose: Print progress
            
        Returns:
            Dictionary with solution and statistics
        """
        print("\n" + "="*70)
        print("SIMULATED ANNEALING SOLVER")
        print("="*70)
        print(f"Initial temperature: {initial_temp}")
        print(f"Cooling rate: {cooling_rate}")
        print(f"Minimum temperature: {min_temp}")
        print(f"Iterations per temperature: {iterations_per_temp}")
        print(f"Maximum time: {max_time} seconds")
        print(f"Sensor budget: {self.S_max}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        # Initialize
        current_solution = self.create_initial_solution()
        current_obj = self.evaluate(current_solution)
        
        best_solution = current_solution.copy()
        best_obj = current_obj
        
        temperature = initial_temp
        iteration = 0
        
        # Statistics
        accepts = 0
        rejects = 0
        improvements = 0
        
        if verbose:
            print(f"Initial solution: {len(current_solution)} sensors, objective = {current_obj:.6f}\n")
        
        # Simulated Annealing loop
        while temperature > min_temp and (time.time() - start_time) < max_time:
            for _ in range(iterations_per_temp):
                # Check time limit
                if (time.time() - start_time) >= max_time:
                    break
                    
                iteration += 1
                
                # Generate neighbor
                neighbor = self.get_neighbor(current_solution)
                neighbor_obj = self.evaluate(neighbor)
                
                # Calculate acceptance probability
                delta = neighbor_obj - current_obj
                
                if delta < 0:  # Better solution (minimization)
                    current_solution = neighbor
                    current_obj = neighbor_obj
                    accepts += 1
                    improvements += 1
                    
                    # Update best
                    if current_obj < best_obj:
                        best_solution = current_solution.copy()
                        best_obj = current_obj
                        if verbose:
                            print(f"Iter {iteration:5d} | T={temperature:7.3f} | New best: {best_obj:.6f} ({len(best_solution)} sensors)")
                else:
                    # Accept worse solution with probability
                    accept_prob = math.exp(-delta / temperature)
                    if random.random() < accept_prob:
                        current_solution = neighbor
                        current_obj = neighbor_obj
                        accepts += 1
                    else:
                        rejects += 1
            
            # Cool down
            temperature *= cooling_rate
        
        elapsed_time = time.time() - start_time
        
        # Print results
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Best objective value: {best_obj:.6f}")
        print(f"Number of sensors placed: {len(best_solution)}")
        print(f"Sensor budget used: {len(best_solution)}/{self.S_max}")
        print(f"\nStatistics:")
        print(f"  Total iterations: {iteration}")
        print(f"  Solutions accepted: {accepts}")
        print(f"  Solutions rejected: {rejects}")
        print(f"  Improvements found: {improvements}")
        print(f"  Acceptance rate: {100*accepts/(accepts+rejects):.2f}%")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        print("="*70 + "\n")
        
        return {
            'sensors': best_solution,
            'objective': best_obj,
            'num_sensors': len(best_solution),
            'iterations': iteration,
            'time': elapsed_time,
            'accepts': accepts,
            'rejects': rejects,
            'improvements': improvements
        }
    
    def save_results(self, result: Dict, output_file: str = 'data/sa_results.txt'):
        """Save results to file"""
        with open(output_file, 'w') as f:
            f.write("SIMULATED ANNEALING RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Objective Value: {result['objective']:.6f}\n\n")
            f.write(f"Sensors Placed: {result['num_sensors']} out of {self.S_max} allowed\n\n")
            f.write("Sensor Locations:\n")
            f.write("-"*70 + "\n")
            f.write("From_Index\tTo_Index\tFrom_Name\tTo_Name\n")
            for (i, j) in sorted(result['sensors']):
                from_name = self.node_names.get(i, str(i))
                to_name = self.node_names.get(j, str(j))
                f.write(f"{i}\t\t{j}\t\t{from_name}\t\t{to_name}\n")
            f.write("\n")
            f.write("Statistics:\n")
            f.write("-"*70 + "\n")
            f.write(f"Total iterations: {result['iterations']}\n")
            f.write(f"Solutions accepted: {result['accepts']}\n")
            f.write(f"Solutions rejected: {result['rejects']}\n")
            f.write(f"Improvements found: {result['improvements']}\n")
            f.write(f"Time elapsed: {result['time']:.2f} seconds\n")
        
        print(f"Results saved to {output_file}")


def main():
    """Main entry point"""
    # Create solver
    solver = SensorPlacementSA('../simulation/data/network_data.dat')
    
    # Solve using Simulated Annealing
    result = solver.solve(
        initial_temp=100.0,
        cooling_rate=0.95,
        min_temp=0.01,
        iterations_per_temp=100,
        max_time=100.0,  # Stop after 100 seconds
        verbose=True
    )
    
    # Save results
    solver.save_results(result)
    
    print("\nSensor placement:")
    print("From_Index\tTo_Index\tFrom_Name\tTo_Name")
    print("-" * 60)
    for i, j in sorted(result['sensors']):
        from_name = solver.node_names.get(i, str(i))
        to_name = solver.node_names.get(j, str(j))
        print(f"{i}\t\t{j}\t\t{from_name}\t\t{to_name}")


if __name__ == "__main__":
    main()
