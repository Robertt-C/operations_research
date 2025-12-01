"""
Data Extraction Pipeline for Water Network Sensor Placement Problem
Extracts flow patterns and network structure from EPANET .inp file
"""

import numpy as np
import pandas as pd
from epyt import epanet
import pickle
import os
from typing import Dict, List, Tuple, Set


class WaterNetworkDataExtractor:
    """
    Extracts and processes data from EPANET network files for sensor placement optimization
    """
    
    def __init__(self, inp_file_path: str):
        """
        Initialize the data extractor
        
        Args:
            inp_file_path: Path to the EPANET .inp file
        """
        self.inp_file = inp_file_path
        self.network = None
        self.node_ids = []
        self.node_indices = {}
        self.link_ids = []
        self.link_indices = {}
        self.edges = []
        self.edge_to_index = {}
        
    def load_network(self):
        """Load the EPANET network"""
        print(f"Loading network from {self.inp_file}...")
        self.network = epanet(self.inp_file)
        print("Network loaded successfully!")
        
    def extract_network_structure(self):
        """Extract nodes, links, and edges from the network"""
        print("\nExtracting network structure...")
        
        # Get all nodes (junctions, reservoirs, tanks)
        self.node_ids = self.network.getNodeNameID()
        self.node_ids.sort()
        self.node_indices = {node: idx for idx, node in enumerate(self.node_ids)}
        
        print(f"  - Total nodes: {len(self.node_ids)}")
        
        # Get all links (pipes, pumps, valves)
        self.link_ids = self.network.getLinkNameID()
        self.link_indices = {link: idx for idx, link in enumerate(self.link_ids)}
        
        print(f"  - Total links: {len(self.link_ids)}")
        
        # Extract edges (node pairs connected by links)
        self._extract_edges()
        
        return {
            'nodes': self.node_ids,
            'node_indices': self.node_indices,
            'links': self.link_ids,
            'edges': self.edges
        }
    
    def _extract_edges(self):
        """Extract edges (i,j) from links with node indices"""
        print("\nExtracting edges from links...")
        
        # Get all connecting nodes at once (much more efficient for large networks)
        all_connecting_nodes = self.network.getNodesConnectingLinksID()
        
        self.edges = []
        self.edge_to_index = {}
        self.link_to_edge = {}  # Map link index to edge index
        edge_set = set()  # Track unique edges to avoid duplicates (for parallel pipes)
        
        for link_idx, link_name in enumerate(self.link_ids):
            # Get the link's connecting nodes by name
            connecting_nodes = all_connecting_nodes[link_idx]
            
            # connecting_nodes is a list like ['NodeName1', 'NodeName2']
            start_node_name = connecting_nodes[0]
            end_node_name = connecting_nodes[1]
            
            # Get node indices from names
            start_node_idx = self.node_indices[start_node_name]
            end_node_idx = self.node_indices[end_node_name]
            
            # Store edge as tuple of node indices
            edge = (start_node_idx, end_node_idx)
            reverse_edge = (end_node_idx, start_node_idx)
            
            # Check if this edge already exists (handles parallel pipes)
            if edge in edge_set or reverse_edge in edge_set:
                # Parallel pipe - map to existing edge
                if edge in self.edge_to_index:
                    self.link_to_edge[link_idx] = self.edge_to_index[edge]
                else:
                    self.link_to_edge[link_idx] = self.edge_to_index[reverse_edge]
            else:
                # New unique edge
                edge_idx = len(self.edges)
                self.edges.append(edge)
                self.edge_to_index[edge] = edge_idx
                self.link_to_edge[link_idx] = edge_idx
                edge_set.add(edge)
                
                # Also store reverse edge mapping for symmetry
                if reverse_edge not in self.edge_to_index:
                    self.edge_to_index[reverse_edge] = edge_idx
        
        print(f"  - Total unique edges: {len(self.edges)} (from {len(self.link_ids)} links)")
        if len(self.edges) < len(self.link_ids):
            print(f"  - Found {len(self.link_ids) - len(self.edges)} parallel pipes")
        
    def run_hydraulic_simulation(self, save_timeseries: bool = False):
        """
        Run hydraulic simulation to get flow patterns
        
        Args:
            save_timeseries: Whether to save time series data (default: False)
            
        Returns:
            Dictionary with simulation results
        """
        print("\nRunning hydraulic simulation...")
        
        # Get simulation time parameters first
        sim_duration = self.network.getTimeSimulationDuration()
        hydraulic_timestep = self.network.getTimeHydraulicStep()
        num_timesteps = int(sim_duration / hydraulic_timestep) + 1
        
        print(f"  - Simulation duration: {sim_duration} seconds")
        print(f"  - Hydraulic timestep: {hydraulic_timestep} seconds")
        print(f"  - Number of timesteps: {num_timesteps}")
        
        # Run hydraulic analysis step by step
        print("\nExtracting flow data...")
        
        # Initialize arrays to store link flows (only data needed)
        num_links = len(self.link_ids)
        link_flows = np.zeros((num_links, num_timesteps))
        
        # Open hydraulic analysis
        self.network.openHydraulicAnalysis()
        self.network.initializeHydraulicAnalysis(0)  # 0 = save to file
        
        tstep = 1
        t = 0
        timestep_idx = 0
        
        # Run through each timestep
        while tstep > 0 and timestep_idx < num_timesteps:
            # Run hydraulic analysis for current timestep
            t = self.network.runHydraulicAnalysis()
            
            # Get link flows for current timestep
            link_flows[:, timestep_idx] = self.network.getLinkFlows().flatten()
            
            # Advance to next timestep
            tstep = self.network.nextHydraulicAnalysisStep()
            timestep_idx += 1
        
        # Close hydraulic analysis
        self.network.closeHydraulicAnalysis()
        
        print(f"  - Completed {timestep_idx} timesteps")
        
        results = {
            'num_timesteps': timestep_idx,
            'sim_duration': sim_duration,
            'hydraulic_timestep': hydraulic_timestep,
            'link_flows': link_flows
        }
        
        # Timeseries CSV files are not needed for optimization
        
        return results
    def build_flow_patterns(self, results: Dict, flow_threshold: float = 0.0):
        """
        Build flow pattern matrix from simulation results
        
        Args:
            results: Dictionary from run_hydraulic_simulation
            flow_threshold: Minimum absolute flow to consider (GPM) - default 0.0 means any positive flow
            
        Returns:
            Dictionary with flow pattern data
        """
        print("\nBuilding flow patterns...")
        
        num_edges = len(self.edges)
        num_timesteps = results['num_timesteps']
        link_flows = results['link_flows']
        
        # Create flow pattern matrix: flow[edge_idx, timestep]
        # Value: 1 if positive flow (contamination can spread), 0 otherwise
        # For parallel pipes, aggregate their flows
        flow_pattern = np.zeros((num_edges, num_timesteps), dtype=int)
        
        # Create actual flow values matrix for reference
        flow_values = np.zeros((num_edges, num_timesteps), dtype=float)
        
        # Process each link and map to its corresponding edge
        for link_idx, link_name in enumerate(self.link_ids):
            edge_idx = self.link_to_edge[link_idx]
            edge = self.edges[edge_idx]
            start_node, end_node = edge
            
            for timestep in range(num_timesteps):
                flow_value = link_flows[link_idx][timestep]
                
                # Aggregate flow values for parallel pipes (sum absolute values for reference)
                flow_values[edge_idx, timestep] += abs(flow_value)
                
                # Mark as 1 ONLY if flow is positive (in forward direction)
                # Positive flow = contamination CAN spread along this edge
                # Negative flow = water flows backwards, contamination CANNOT spread
                # Zero flow = no flow = no contamination spread
                if flow_value > flow_threshold:
                    flow_pattern[edge_idx, timestep] = 1
        
        print(f"  - Flow pattern matrix shape: {flow_pattern.shape}")
        print(f"  - Non-zero flow patterns: {np.sum(flow_pattern)}/{flow_pattern.size}")
        
        # Calculate flow statistics
        flow_stats = self._calculate_flow_statistics(flow_pattern, flow_values)
        
        return {
            'flow_pattern': flow_pattern,
            'flow_values': flow_values,
            'num_patterns': num_timesteps,
            'num_edges': num_edges,
            'flow_threshold': flow_threshold,
            'statistics': flow_stats
        }
    
    def _calculate_flow_statistics(self, flow_pattern: np.ndarray, flow_values: np.ndarray):
        """Calculate statistics about flow patterns"""
        stats = {
            'total_patterns': flow_pattern.shape[1],
            'total_edges': flow_pattern.shape[0],
            'active_flows_per_pattern': [],
            'pattern_diversity': 0,
            'average_flow_magnitude': np.mean(np.abs(flow_values)),
            'max_flow_magnitude': np.max(np.abs(flow_values)),
            'min_nonzero_flow': np.min(np.abs(flow_values[flow_values != 0])) if np.any(flow_values != 0) else 0
        }
        
        # Count active flows per pattern
        for p in range(flow_pattern.shape[1]):
            active_flows = np.sum(flow_pattern[:, p])
            stats['active_flows_per_pattern'].append(int(active_flows))
        
        # Calculate pattern diversity (number of unique patterns)
        unique_patterns = len(set([tuple(flow_pattern[:, p]) for p in range(flow_pattern.shape[1])]))
        stats['pattern_diversity'] = unique_patterns
        
        return stats
    
    def build_attack_scenarios(self, attack_nodes: List[str] = None):
        """
        Build attack scenario data structure
        
        Args:
            attack_nodes: List of node names to consider as potential attack points
                         If None, all nodes (junctions, tanks, reservoirs) are considered
                         
        Returns:
            Dictionary with attack scenario data
        """
        print("\nBuilding attack scenarios...")
        
        if attack_nodes is None:
            # Include ALL nodes as potential attack points (junctions, tanks, reservoirs)
            attack_node_indices = list(range(len(self.node_ids)))
            attack_nodes = [self.node_ids[idx] for idx in attack_node_indices]
        else:
            attack_node_indices = [self.node_indices[node] for node in attack_nodes]
        
        print(f"  - Number of potential attack nodes: {len(attack_nodes)} (all nodes)")
        
        # Create alpha matrix: alpha[i,p]
        # For simplicity, assume uniform probability across all patterns
        # This can be customized based on specific scenarios
        num_attack_nodes = len(attack_node_indices)
        
        return {
            'attack_nodes': attack_nodes,
            'attack_node_indices': attack_node_indices,
            'num_attack_nodes': num_attack_nodes
        }
    
    def build_optimization_data_structures(self, 
                                          flow_pattern_data: Dict,
                                          attack_data: Dict,
                                          alpha: np.ndarray = None,
                                          delta: np.ndarray = None):
        """
        Build complete data structures needed for optimization model
        
        Args:
            flow_pattern_data: Output from build_flow_patterns()
            attack_data: Output from build_attack_scenarios()
            alpha: Attack probability matrix [i,p] - if None, uniform distribution
            delta: Pattern weight matrix [j,p] - if None, uniform weights
            
        Returns:
            Dictionary with all optimization data structures
        """
        print("\nBuilding optimization data structures...")
        
        num_nodes = len(self.node_ids)
        num_patterns = flow_pattern_data['num_patterns']
        num_attack_nodes = attack_data['num_attack_nodes']
        attack_node_indices = attack_data['attack_node_indices']
        
        # Build alpha[i,p] - probability of attack at node i under pattern p
        # Normalize so sum over all nodes and patterns equals 1
        if alpha is None:
            total_scenarios = num_attack_nodes * num_patterns
            alpha = np.ones((num_attack_nodes, num_patterns)) / total_scenarios
            print(f"  - Using uniform attack probability distribution")
            print(f"  - Total attack scenarios: {total_scenarios} ({num_attack_nodes} nodes × {num_patterns} patterns)")
            print(f"  - Each scenario probability: {1/total_scenarios:.6f}")
        
        # Build delta[j,p] - weight/importance of node j under pattern p
        if delta is None:
            # Following the paper: assign 500 people to each demand node
            # Tanks/reservoirs get 0 population
            node_type_codes = self.network.getNodeTypeIndex()  # Returns numeric codes
            delta = np.zeros((num_nodes, num_patterns))
            
            for i in range(num_nodes):
                node_type = node_type_codes[i]
                # Node types: 0=Junction (demand node), 1=Reservoir, 2=Tank
                if node_type == 0:  # Junction nodes get population
                    delta[i, :] = 500  # 500 people per demand node
                else:  # Tanks/Reservoirs get 0
                    delta[i, :] = 0
            
            num_demand_nodes = np.sum(delta[:, 0] > 0)
            total_population = np.sum(delta[:, 0])
            print(f"  - Assigned 500 people to each of {num_demand_nodes} demand nodes (junctions)")
            print(f"  - Tanks/Reservoirs have 0 population")
            print(f"  - Total population: {total_population:.0f}")
        
        # Normalize delta by total population to get proportion of population at each node
        # This makes the objective value represent expected proportion of population exposed
        total_population = np.sum(delta[:, 0])  # Same across all patterns
        if total_population > 0:
            delta = delta / total_population
            print(f"  - Normalized delta by total population ({total_population:.0f})")
        
        # Build flow matrix for optimization: flow[k,j,p]
        # This is the flow_pattern matrix but indexed by edges
        flow = flow_pattern_data['flow_pattern']
        
        # Build edge list with node pairs
        edge_list = []
        for edge_idx, (i, j) in enumerate(self.edges):
            edge_list.append({
                'edge_index': edge_idx,
                'from_node': i,
                'to_node': j,
                'from_node_name': self.node_ids[i],
                'to_node_name': self.node_ids[j],
                'link_name': self.link_ids[edge_idx]
            })
        
        optimization_data = {
            'num_nodes': num_nodes,
            'num_patterns': num_patterns,
            'num_edges': flow_pattern_data['num_edges'],
            'num_attack_nodes': num_attack_nodes,
            'node_ids': self.node_ids,
            'node_indices': self.node_indices,
            'attack_node_indices': attack_node_indices,
            'edges': self.edges,
            'edge_list': edge_list,
            'edge_to_index': self.edge_to_index,
            'alpha': alpha,
            'delta': delta,
            'flow': flow,
            'flow_values': flow_pattern_data['flow_values'],
            'flow_threshold': flow_pattern_data['flow_threshold']
        }
        
        print(f"\nData structure summary:")
        print(f"  - Nodes: {num_nodes}")
        print(f"  - Attack nodes: {num_attack_nodes}")
        print(f"  - Edges: {optimization_data['num_edges']}")
        print(f"  - Flow patterns (timesteps): {num_patterns}")
        print(f"  - Alpha matrix shape: {alpha.shape}")
        print(f"  - Delta matrix shape: {delta.shape}")
        print(f"  - Flow matrix shape: {flow.shape}")
        
        return optimization_data
    
    def save_data_structures(self, optimization_data: Dict, output_file: str = None):
        """
        Save data structures to AMPL format
        
        Args:
            optimization_data: Output from build_optimization_data_structures()
            output_file: Full path to output .dat file (default: data/network_data.dat in same dir as input)
        """
        if output_file is None:
            # Get directory of inp_file - if already in data/, use that
            base_dir = os.path.dirname(self.inp_file)
            if os.path.basename(base_dir) == 'data':
                output_dir = base_dir
            else:
                output_dir = os.path.join(base_dir, 'data')
            output_file = os.path.join(output_dir, 'network_data.dat')
        else:
            output_dir = os.path.dirname(output_file)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving data structures to {output_file}...")
        
        # Save as AMPL .dat format
        self._save_ampl_dat(optimization_data, output_file)
        
        print(f"\nData structures saved successfully!")
    
    def _save_ampl_dat(self, optimization_data: Dict, output_file: str):
        """
        Save data in AMPL .dat format
        
        Args:
            optimization_data: Output from build_optimization_data_structures()
            output_file: Full path to output .dat file
        """
        print(f"\nSaving AMPL data file...")
        
        with open(output_file, 'w') as f:
            f.write("# AMPL Data File for Water Network Sensor Placement\n")
            f.write("# Generated automatically from EPANET network file\n")
            f.write(f"# Source: {os.path.basename(self.inp_file)}\n")
            f.write(f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Parameters (scalars)
            f.write("# ===== PARAMETERS (Scalars) =====\n\n")
            f.write(f"param num_nodes := {optimization_data['num_nodes']};\n")
            f.write(f"param num_attack_nodes := {optimization_data['num_attack_nodes']};\n")
            f.write(f"param num_edges := {optimization_data['num_edges']};\n")
            f.write(f"param num_patterns := {optimization_data['num_patterns']};\n")
            f.write(f"param flow_threshold := {optimization_data['flow_threshold']};\n")
            f.write("\n")
            
            # Sets
            f.write("# ===== SETS =====\n\n")
            
            # Set of all nodes (0-indexed)
            f.write("set NODES := ")
            f.write(" ".join(str(i) for i in range(optimization_data['num_nodes'])))
            f.write(";\n\n")
            
            # Set of attack nodes (0-indexed)
            f.write("set ATTACK_NODES := ")
            f.write(" ".join(str(i) for i in optimization_data['attack_node_indices']))
            f.write(";\n\n")
            
            # Set of patterns (0-indexed)
            f.write("set PATTERNS := ")
            f.write(" ".join(str(p) for p in range(optimization_data['num_patterns'])))
            f.write(";\n\n")
            
            # Set of edges (as tuples)
            f.write("set EDGES := \n")
            for edge_idx, (i, j) in enumerate(optimization_data['edges']):
                f.write(f"  ({i},{j})")
                if edge_idx < len(optimization_data['edges']) - 1:
                    f.write("\n")
            f.write("\n;\n\n")
            
            # Node names mapping
            f.write("# Node index to name mapping\n")
            f.write("param node_name :=\n")
            for idx, name in enumerate(self.node_ids):
                f.write(f"  {idx}  {name}\n")
            f.write(";\n\n")
            
            # Alpha matrix - attack probabilities [attack_node_idx, pattern]
            f.write("# ===== ALPHA: Attack probabilities [i,p] =====\n")
            f.write("# alpha[i,p] = probability of attack at node i under pattern p\n\n")
            f.write("param alpha : ")
            # Column headers (patterns)
            f.write(" ".join(str(p) for p in range(optimization_data['num_patterns'])))
            f.write(" :=\n")
            
            alpha = optimization_data['alpha']
            attack_indices = optimization_data['attack_node_indices']
            for attack_idx, node_idx in enumerate(attack_indices):
                f.write(f"  {node_idx}  ")
                f.write("  ".join(f"{alpha[attack_idx, p]:.6f}" for p in range(optimization_data['num_patterns'])))
                f.write("\n")
            f.write(";\n\n")
            
            # Delta matrix - node weights [node, pattern]
            f.write("# ===== DELTA: Node importance weights [j,p] =====\n")
            f.write("# delta[j,p] = importance/weight of node j under pattern p\n\n")
            f.write("param delta : ")
            # Column headers (patterns)
            f.write(" ".join(str(p) for p in range(optimization_data['num_patterns'])))
            f.write(" :=\n")
            
            delta = optimization_data['delta']
            for node_idx in range(optimization_data['num_nodes']):
                f.write(f"  {node_idx}  ")
                f.write("  ".join(f"{delta[node_idx, p]:.6f}" for p in range(optimization_data['num_patterns'])))
                f.write("\n")
            f.write(";\n\n")
            
            # Flow pattern matrix - binary flow indicators [edge_idx, pattern]
            f.write("# ===== FLOW: Binary flow pattern matrix [edge,p] =====\n")
            f.write("# flow[e,p] = 1 if positive flow (contamination can spread), 0 otherwise\n")
            f.write("# Negative flows ignored (contamination cannot spread backwards)\n")
            f.write("# Edges are indexed 0 to num_edges-1\n\n")
            f.write("param flow : ")
            # Column headers (patterns)
            f.write(" ".join(str(p) for p in range(optimization_data['num_patterns'])))
            f.write(" :=\n")
            
            flow = optimization_data['flow']
            for edge_idx in range(optimization_data['num_edges']):
                f.write(f"  {edge_idx}  ")
                f.write("  ".join(str(int(flow[edge_idx, p])) for p in range(optimization_data['num_patterns'])))
                f.write("\n")
            f.write(";\n\n")
            
            # Edge connectivity - which nodes does each edge connect
            f.write("# ===== EDGE CONNECTIVITY =====\n")
            f.write("# Maps edge index to (from_node, to_node)\n\n")
            f.write("param edge_from :=\n")
            for edge_idx, (i, j) in enumerate(optimization_data['edges']):
                f.write(f"  {edge_idx}  {i}\n")
            f.write(";\n\n")
            
            f.write("param edge_to :=\n")
            for edge_idx, (i, j) in enumerate(optimization_data['edges']):
                f.write(f"  {edge_idx}  {j}\n")
            f.write(";\n\n")
            
            # Alternative format: flow indexed by (from_node, to_node, pattern)
            f.write("# ===== ALTERNATIVE: Flow indexed by (from, to, pattern) =====\n")
            f.write("# flow_edge[i,j,p] = 1 if there is flow from node i to j in pattern p\n\n")
            f.write("param flow_edge :=\n")
            for edge_idx, (i, j) in enumerate(optimization_data['edges']):
                for p in range(optimization_data['num_patterns']):
                    if flow[edge_idx, p] == 1:
                        f.write(f"  {i} {j} {p}  1\n")
            f.write(";\n\n")
            
            f.write("# End of AMPL data file\n")
        
        print(f"  - Saved {os.path.basename(output_file)} (AMPL format)")
    
    def close(self):
        """Close the EPANET network"""
        if self.network is not None:
            self.network.unload()
            print("\nNetwork unloaded.")
