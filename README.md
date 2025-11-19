# Grover's Algorithm Experiment

## Overview
This experiment compares **Grover's Quantum Search Algorithm** with **Classical Linear Search** for unstructured database search.

## Requirements
- Python 3.8+
- macOS (tested on macOS 14+)

## Setup Instructions

### 1. Install Python Dependencies
```bash
# Make sure you're in the paper_experiment directory
cd ~/Desktop/fall25/applied-ds/ads-project/paper_experiment

# Install required packages
pip3 install -r requirements.txt
```

### 2. Run the Experiment
```bash
python3 run_experiment.py
```

### 3. Generate Visualizations
```bash
python3 visualize_results.py
```

## Output Files
- `results/experiment_results.csv` - Raw data
- `results/latex_table.tex` - LaTeX table for paper
- `visualizations/grover_vs_classical.png` - Charts

## File Descriptions
- `classical_search.py` - Classical O(N) linear search
- `grover_simulator.py` - Grover's O(√N) quantum search
- `run_experiment.py` - Main experiment driver
- `visualize_results.py` - Data visualization