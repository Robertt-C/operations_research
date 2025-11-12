# ============================================================================
# AMPL Model for Water Network Sensor Placement Optimization
# ============================================================================
# This model uses data extracted from EPANET network file (net3.inp)
# Data file: network_data.dat
# ============================================================================

# ===== SETS =====

set NODES;                  # Set of all nodes (0-indexed)
set ATTACK_NODES;           # Set of nodes that can be attacked
set PATTERNS;               # Set of flow patterns (timesteps)
set EDGES within {NODES, NODES};  # Set of edges (directed)

# ===== PARAMETERS =====

# Dimensions
param num_nodes >= 0;
param num_attack_nodes >= 0;
param num_edges >= 0;
param num_patterns >= 0;
param flow_threshold >= 0;

# Node names (for reference)
param node_name {NODES} symbolic;

# Edge connectivity
param edge_from {0..num_edges-1} >= 0;
param edge_to {0..num_edges-1} >= 0;

# Attack probabilities: alpha[i,p] = probability of attack at node i in pattern p
param alpha {ATTACK_NODES, PATTERNS} >= 0, <= 1;

# Node importance weights: delta[j,p] = importance of node j in pattern p
param delta {NODES, PATTERNS} >= 0;

# Flow patterns: flow[e,p] = 1 if flow exists on edge e in pattern p
param flow {0..num_edges-1, PATTERNS} binary;

# Alternative flow representation: flow_edge[i,j,p]
param flow_edge {i in NODES, j in NODES, p in PATTERNS} binary default 0;

# Sensor budget (set this based on your constraint)
param S_max >= 0 default 5;

# ===== DECISION VARIABLES =====

# Sensor placement: s[i,j] = 1 if sensor placed on edge (i,j)
var s {EDGES} binary;

# Contamination: c[i,p,j] = 1 if node j contaminated when i attacked in pattern p
var c {i in ATTACK_NODES, p in PATTERNS, j in NODES} binary;

# ===== OBJECTIVE FUNCTION =====

# Minimize expected contamination weighted by node importance
minimize Expected_Contamination:
    sum {i in ATTACK_NODES, p in PATTERNS, j in NODES} 
        alpha[i,p] * c[i,p,j] * delta[j,p];

# ===== CONSTRAINTS =====

# Constraint 1: Attack initialization
# A node directly attacked is always contaminated
subject to Attack_Init {i in ATTACK_NODES, p in PATTERNS}:
    c[i,p,i] = 1;

# Constraint 2: Sensor symmetry
# A sensor on edge (i,j) also covers edge (j,i)
subject to Sensor_Symmetry {(i,j) in EDGES: (j,i) in EDGES}:
    s[i,j] = s[j,i];

# Constraint 3: Contamination propagation
# If node k is contaminated and flow goes from k to j without a sensor,
# then j becomes contaminated
subject to Contamination_Propagation {e in 0..num_edges-1, p in PATTERNS, i in ATTACK_NODES: flow[e,p] = 1}:
    c[i,p,edge_to[e]] >= c[i,p,edge_from[e]] - s[edge_from[e],edge_to[e]];

# Constraint 4: Sensor budget
# Total number of sensors cannot exceed the budget
subject to Sensor_Limit:
    sum {(i,j) in EDGES} s[i,j] <= S_max;