# Grover's Algorithm Experiment - Qiskit Implementation

**Scientific Comparison of Quantum Search vs Classical Methods**

## Authors
- Abdullah Ahmed (@abdullah-ax) - Qiskit Implementation, Experiment Design
- Linah Aboudoma (@linah1604) - Data Analysis, Visualization

## Overview
Rigorous experimental validation of Grover's quantum search algorithm using IBM's Qiskit framework, comparing against classical linear search across database sizes from 2^4 to 2^20 elements.

## Requirements
```bash
pip install -r requirements.txt
```

## Experimental Protocol

### Parameters
- **Database Sizes:** 2^4 to 2^20 (16 to 1,048,576 elements)
- **Target Positions:** Middle (N/2), End (N-1)
- **Repetitions:** 10 trials per configuration
- **Qiskit Shots:** 1,000 measurements per run
- **Total Experiments:** ~340 trials

### Running the Experiment
```bash
# Full experiment (takes several hours)
python experiment_runner.py

# Statistical analysis
python statistical_analysis.py

# Generate visualizations
python visualize_results.py
```

## Citation
Implementation based on:
- Qiskit Grover's Algorithm Tutorial: https://qiskit.org/textbook/ch-algorithms/grover.html
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

## Results Structure
```
results/
├── raw_data/           # Trial-level data
├── processed/          # Aggregated statistics
├── figures/            # Publication-quality plots
└── statistical_report.txt
```

## Key Findings
- Grover's algorithm achieves O(√N) query complexity
- Speedup scales from 3x (N=16) to >600x (N=2^20)
- Success probability consistently >95% across all database sizes
- Experimental results match theoretical predictions within 1%
