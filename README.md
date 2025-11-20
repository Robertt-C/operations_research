# Water Network Sensor Placement Optimization

Sensor placement optimization for contamination detection in water distribution networks using the EPANET net3 benchmark.

## Problem

**Minimize** expected contamination (population exposed) by placing sensors on network edges.

- 97 nodes (92 junctions + 5 tanks/reservoirs)
- 119 edges (pipes)
- 25 time patterns
- 500 people per junction node
- All nodes can be contamination sources

## Quick Start

### 1. Setup

```bash
# Activate virtual environment
venv_operations_research\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
cd simulation
python run_pipeline.py
```
Output: `data/network_data.dat`

### 3. Solve

**Option A - AMPL (Exact)**:
```bash
cd model
ampl solve.run
```
Results: `data/sensor_placements.txt`, `data/contamination_results.txt`

**Option B - Simulated Annealing (Metaheuristic)**:
```bash
cd model
python simulated_annealing_solver.py
```
Results: `data/sa_results.txt`

## Model

**Objective**: `minimize Σ α[i,p] × c[i,p,j] × δ[j,p]`

Where:
- `α[i,p]`: Attack probability at node i in pattern p
- `δ[j,p]`: Population at node j in pattern p
- `c[i,p,j]`: 1 if node j contaminated when i attacked in pattern p

**Constraints**:
1. Attack node always contaminated
2. Sensor symmetry: s[i,j] = s[j,i]
3. Contamination spreads through pipes without sensors
4. Maximum S_max sensors

## Configuration

**Sensor Budget** (in `model/sensor_placement.mod`):
```ampl
param S_max := 5;  # Change this value
```

**SA Parameters** (in `model/simulated_annealing_solver.py`):
```python
max_time=100.0,        # Time limit (seconds)
initial_temp=100.0,    # Starting temperature
cooling_rate=0.95      # Cooling factor
```

## Project Structure

```
operations_research/
├── simulation/
│   ├── data_extraction.py          # Extract data from EPANET
│   ├── run_pipeline.py             # Generate AMPL data file
│   └── data/
│       ├── net3.inp                # EPANET input (network definition)
│       └── network_data.dat        # AMPL data (generated)
├── model/
│   ├── sensor_placement.mod        # AMPL optimization model
│   ├── solve.run                   # AMPL execution script
│   ├── simulated_annealing_solver.py  # SA metaheuristic solver
│   └── data/
│       ├── sensor_placements.txt   # AMPL results
│       ├── contamination_results.txt  # AMPL contamination analysis
│       └── sa_results.txt          # SA results
├── paper/                          # Project documentation
└── venv_operations_research/       # Virtual environment
```

## Dependencies

- Python 3.11+
- epyt (EPANET toolkit)
- numpy, pandas
- AMPL + Gurobi solver