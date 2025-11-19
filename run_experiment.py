"""
Main Experiment Driver: Grover's Algorithm vs Classical Search
Created by: abdullah-ax & linah1604
Date: 2025-11-19
"""

import numpy as np
import pandas as pd
from classical_search import ClassicalSearch
from grover_simulator import GroverSimulator

def run_experiment(N, num_trials=10):
    """
    Run experiment for a given database size
    
    Args:
        N (int): Database size
        num_trials (int): Number of trials to average
    
    Returns:
        dict: Experiment results
    """
    print(f"\nRunning experiment for N = {N} ({num_trials} trials)...")
    
    total_classical_queries = 0
    total_grover_queries = 0
    
    for trial in range(num_trials):
        # Use middle of database as target for consistency
        target = N // 2
        
        # Classical Search
        classical = ClassicalSearch(N, target)
        classical.search()
        total_classical_queries += classical.get_query_count()
        
        # Grover's Algorithm
        grover = GroverSimulator(N, target)
        grover.search()
        total_grover_queries += grover.get_query_count()
    
    # Calculate averages
    avg_classical = total_classical_queries / num_trials
    avg_grover = total_grover_queries / num_trials
    speedup = avg_classical / avg_grover if avg_grover > 0 else 0
    
    result = {
        'N': N,
        'Classical_Queries': int(avg_classical),
        'Grover_Queries': int(avg_grover),
        'Speedup': speedup,
        'Sqrt_N': int(np.sqrt(N))
    }
    
    print(f"  Average Classical Queries: {result['Classical_Queries']}")
    print(f"  Average Grover Queries: {result['Grover_Queries']}")
    print(f"  Speedup: {result['Speedup']:.2f}x")
    
    return result

def print_results_table(results_df):
    """Print formatted results table"""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS: GROVER'S ALGORITHM VS CLASSICAL SEARCH")
    print("=" * 80 + "\n")
    
    print(f"{'N':<12}{'Classical (O(N))':<20}{'Grover (O(√N))':<20}{'Speedup':<15}{'√N (Theory)':<13}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['N']:<12}{row['Classical_Queries']:<20}{row['Grover_Queries']:<20}"
              f"{row['Speedup']:.2f}x{'':<10}{row['Sqrt_N']:<13}")
    
    print("=" * 80)

def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""
    latex = r"""\begin{table}[h]
\centering
\caption{Experimental Results: Grover's Algorithm vs Classical Search}
\label{tab:grover_results}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{N} & \textbf{Classical} & \textbf{Grover} & \textbf{Speedup} \\
           & \textbf{Queries}   & \textbf{Queries} & \textbf{Factor} \\
\hline
"""
    
    for _, row in results_df.iterrows():
        latex += f"{row['N']} & {row['Classical_Queries']} & {row['Grover_Queries']} & {row['Speedup']:.2f}x \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    return latex

def main():
    """Main experiment execution"""
    print("=" * 60)
    print("  GROVER'S ALGORITHM EXPERIMENT")
    print("  Quantum vs Classical Search")
    print("=" * 60)
    
    # Database sizes as specified in Task 3
    database_sizes = [16, 64, 256, 1024]
    num_trials = 10
    
    # Run experiments
    results = []
    for N in database_sizes:
        result = run_experiment(N, num_trials)
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results table
    print_results_table(results_df)
    
    # Save to CSV
    csv_filename = 'results/experiment_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Generate and save LaTeX table
    latex_table = generate_latex_table(results_df)
    latex_filename = 'results/latex_table.tex'
    with open(latex_filename, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_filename}")
    
    print("\n" + "=" * 60)
    print("LATEX TABLE FOR PAPER")
    print("=" * 60 + "\n")
    print(latex_table)
    
    print("\n✓ Experiment completed successfully!")
    print("✓ Next step: Run visualize_results.py to generate plots")

if __name__ == "__main__":
    main()