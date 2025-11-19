"""
Experiment Runner: Grover's Algorithm vs Classical Search
Scientific experiment with proper statistical methodology

Authors: abdullah-ax, linah1604,
Date: 2025-11-19
"""

from classical_search import ClassicalSearch
from grover_qiskit import QiskitGroverSearch
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GroverExperiment:
    """
    Comprehensive experimental framework for comparing Grover's algorithm
    with classical search across multiple database sizes.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize experiment framework.

        Args:
            output_dir: Directory for storing results
        """
        self.output_dir = output_dir
        self.create_directories()

        # Experiment parameters
        self.repetitions = 10  # Statistical repetitions per configuration
        self.qiskit_shots = 1000  # Measurement shots per Qiskit run

        # Database sizes: 2^4 to 2^20
        self.qubit_range = list(range(4, 21))  # 4 to 20 qubits

        # Results storage
        self.results = []

    def create_directories(self):
        """Create directory structure for results."""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/raw_data",
            f"{self.output_dir}/processed",
            f"{self.output_dir}/figures"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def run_single_trial(self, num_qubits: int, target_position: str,
                         trial_num: int) -> Dict:
        """
        Run a single experimental trial.

        Args:
            num_qubits: Number of qubits (N = 2^num_qubits)
            target_position: 'middle' or 'end'
            trial_num: Trial number for this configuration

        Returns:
            Dictionary with trial results
        """
        N = 2 ** num_qubits

        # Determine target index based on position
        if target_position == 'middle':
            target_index = N // 2
        else:  # 'end'
            target_index = N - 1

        # Run Classical Search
        classical = ClassicalSearch(N, target_index, target_position)
        classical_result = classical.search()
        classical_queries = classical.get_query_count()

        # Run Grover's Algorithm (Qiskit)
        grover = QiskitGroverSearch(num_qubits, target_index)
        grover_result, success_prob = grover.search(shots=self.qiskit_shots)
        grover_queries = grover.get_oracle_queries()
        grover_iterations = grover.get_optimal_iterations()

        # Calculate metrics
        speedup = classical_queries / grover_queries if grover_queries > 0 else 0
        correctness = 1 if grover_result == target_index else 0

        return {
            'num_qubits': num_qubits,
            'N': N,
            'sqrt_N': int(np.sqrt(N)),
            'target_position': target_position,
            'target_index': target_index,
            'trial': trial_num,
            'classical_queries': classical_queries,
            'grover_queries': grover_queries,
            'grover_iterations': grover_iterations,
            'grover_success_prob': success_prob,
            'grover_correct': correctness,
            'speedup': speedup,
            'timestamp': datetime.now().isoformat()
        }

    def run_experiment_set(self, num_qubits: int, target_position: str) -> List[Dict]:
        """
        Run complete experiment set for given parameters.

        Args:
            num_qubits: Number of qubits
            target_position: Target position ('middle' or 'end')

        Returns:
            List of trial results
        """
        N = 2 ** num_qubits
        results = []

        desc = f"N={N:>7} | Target: {target_position:6}"
        for trial in tqdm(range(self.repetitions), desc=desc, leave=False):
            try:
                result = self.run_single_trial(
                    num_qubits, target_position, trial + 1)
                results.append(result)
            except Exception as e:
                print(f"\nError in trial {trial + 1}: {e}")
                continue

        return results

    def run_full_experiment(self):
        """Execute complete experimental protocol."""
        print("=" * 70)
        print("GROVER'S ALGORITHM EXPERIMENT - SCIENTIFIC PROTOCOL")
        print("=" * 70)
        print(f"Qubit Range: {self.qubit_range[0]} to {self.qubit_range[-1]}")
        print(
            f"Database Sizes: 2^{self.qubit_range[0]} to 2^{self.qubit_range[-1]}")
        print(f"Repetitions per configuration: {self.repetitions}")
        print(f"Target Positions: middle, end")
        print(
            f"Total Experiments: {len(self.qubit_range) * 2 * self.repetitions}")
        print("=" * 70 + "\n")

        # Run experiments for each qubit count
        for num_qubits in tqdm(self.qubit_range, desc="Overall Progress"):
            # Middle target
            middle_results = self.run_experiment_set(num_qubits, 'middle')
            self.results.extend(middle_results)

            # End target
            end_results = self.run_experiment_set(num_qubits, 'end')
            self.results.extend(end_results)

            # Save intermediate results
            self.save_raw_results()

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

        # Process and save results
        self.process_results()

    def save_raw_results(self):
        """Save raw experimental data."""
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/raw_data/experiment_raw_{timestamp}.csv"
        df.to_csv(filename, index=False)

    def process_results(self):
        """Process and aggregate experimental results."""
        df = pd.DataFrame(self.results)

        # Group by configuration and calculate statistics
        grouped = df.groupby(['num_qubits', 'N', 'target_position']).agg({
            'classical_queries': ['mean', 'std'],
            'grover_queries': ['mean', 'std'],
            'grover_iterations': 'first',
            'grover_success_prob': ['mean', 'std'],
            'grover_correct': 'mean',
            'speedup': ['mean', 'std'],
            'sqrt_N': 'first'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_')
                           for col in grouped.columns.values]

        # Save processed results
        output_file = f"{self.output_dir}/processed/experiment_summary.csv"
        grouped.to_csv(output_file, index=False)

        print(f"\n✓ Processed results saved to: {output_file}")

        # Save metadata
        metadata = {
            'experiment_date': datetime.now().isoformat(),
            'total_trials': len(self.results),
            'repetitions_per_config': self.repetitions,
            'qiskit_shots': self.qiskit_shots,
            'qubit_range': self.qubit_range,
            'database_sizes': [2**q for q in self.qubit_range]
        }

        metadata_file = f"{self.output_dir}/processed/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_file}")

        # Print summary statistics
        self.print_summary(grouped)

    def print_summary(self, summary_df):
        """Print experimental summary."""
        print("\n" + "=" * 70)
        print("EXPERIMENTAL SUMMARY")
        print("=" * 70)

        # Filter for middle target only for cleaner display
        middle_df = summary_df[summary_df['target_position'] == 'middle']

        print(
            f"\n{'N':>10} {'Classical':>15} {'Grover':>15} {'Speedup':>15} {'Success %':>15}")
        print("-" * 70)

        for _, row in middle_df.iterrows():
            print(f"{row['N']:>10.0f} "
                  f"{row['classical_queries_mean']:>10.1f}±{row['classical_queries_std']:>4.1f} "
                  f"{row['grover_queries_mean']:>10.1f}±{row['grover_queries_std']:>4.1f} "
                  f"{row['speedup_mean']:>10.2f}±{row['speedup_std']:>4.2f} "
                  f"{row['grover_success_prob_mean']*100:>10.1f}%")

        print("=" * 70)


def main():
    """Main execution function."""
    experiment = GroverExperiment(output_dir="results")
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
