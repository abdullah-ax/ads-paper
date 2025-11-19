"""
Enhanced Visualization Module
Publication-quality figures for research paper

Authors: linah1604, abdullah-ax
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ExperimentVisualizer:
    """Create publication-ready visualizations."""

    def __init__(self, results_file: str, output_dir: str = "results/figures"):
        """Initialize visualizer."""
        self.df = pd.read_csv(results_file)
        self.output_dir = output_dir

        # Filter for middle target (cleaner visualization)
        self.plot_df = self.df[self.df['target_position'] == 'middle'].copy()

    def plot_query_complexity(self):
        """Plot query complexity comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        ax1.plot(self.plot_df['N'], self.plot_df['classical_queries_mean'],
                 marker='o', linewidth=2.5, markersize=8, label='Classical O(N)', color='#E74C3C')
        ax1.plot(self.plot_df['N'], self.plot_df['grover_queries_mean'],
                 marker='s', linewidth=2.5, markersize=8, label='Grover O(√N)', color='#3498DB')
        ax1.plot(self.plot_df['N'], self.plot_df['sqrt_N_first'],
                 marker='^', linewidth=2, markersize=6, label='√N (Theory)',
                 color='#2ECC71', linestyle='--', alpha=0.7)

        ax1.set_xlabel('Database Size (N)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Query Count', fontsize=12, fontweight='bold')
        ax1.set_title('Query Complexity (Linear Scale)',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Log-log scale - FIXED
        ax2.loglog(self.plot_df['N'], self.plot_df['classical_queries_mean'],
                   marker='o', linewidth=2.5, markersize=8,
                   label='Classical O(N)', color='#E74C3C', base=2)
        ax2.loglog(self.plot_df['N'], self.plot_df['grover_queries_mean'],
                   marker='s', linewidth=2.5, markersize=8,
                   label='Grover O(√N)', color='#3498DB', base=2)
        ax2.loglog(self.plot_df['N'], self.plot_df['sqrt_N_first'],
                   marker='^', linewidth=2, markersize=6,
                   label='√N (Theory)', color='#2ECC71',
                   linestyle='--', alpha=0.7, base=2)

        ax2.set_xlabel('Database Size (N)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Query Count', fontsize=12, fontweight='bold')
        ax2.set_title('Query Complexity (Log-Log Scale)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/query_complexity.png",
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/query_complexity.png")
        plt.close()

    def plot_speedup_analysis(self):
        """Plot speedup trends."""
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Speedup vs N
        ax1.plot(self.plot_df['N'], self.plot_df['speedup_mean'],
                 marker='o', linewidth=2.5, markersize=8, color='#9B59B6', label='Observed')
        ax1.fill_between(self.plot_df['N'],
                         self.plot_df['speedup_mean'] -
                         self.plot_df['speedup_std'],
                         self.plot_df['speedup_mean'] +
                         self.plot_df['speedup_std'],
                         alpha=0.2, color='#9B59B6')

        # Theoretical sqrt(N) / 2
        theoretical_speedup = np.sqrt(self.plot_df['N']) / 2
        ax1.plot(self.plot_df['N'], theoretical_speedup,
                 linestyle='--', linewidth=2, color='#E67E22', label='Theoretical (√N/2)')

        ax1.set_xlabel('Database Size (N)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax1.set_title('Quantum Speedup vs Database Size',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Speedup distribution
        bars = ax2.bar(range(len(self.plot_df)), self.plot_df['speedup_mean'],
                       color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.errorbar(range(len(self.plot_df)), self.plot_df['speedup_mean'],
                     yerr=self.plot_df['speedup_std'], fmt='none',
                     ecolor='black', capsize=5, linewidth=1.5)

        # Add value labels
        for i, (idx, row) in enumerate(self.plot_df.iterrows()):
            height = row['speedup_mean'] + row['speedup_std']
            y_offset = max(5, height * 0.05)  # Dynamic offset
            ax2.text(i, height + y_offset,
                     f"{row['speedup_mean']:.1f}x",
                     ha='center', fontsize=9, fontweight='bold')

        ax2.set_xticks(range(len(self.plot_df)))
        ax2.set_xticklabels(
            [f"2^{int(np.log2(n))}" for n in self.plot_df['N']], rotation=45)
        ax2.set_xlabel('Database Size (2^n)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/speedup_analysis.png",
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/speedup_analysis.png")
        plt.close()

    def plot_success_probability(self):
        """Plot Grover's success probability."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.plot_df['N'], self.plot_df['grover_success_prob_mean'] * 100,
                marker='o', linewidth=2.5, markersize=8, color='#16A085', label='Observed')
        ax.fill_between(self.plot_df['N'],
                        (self.plot_df['grover_success_prob_mean'] -
                         self.plot_df['grover_success_prob_std']) * 100,
                        (self.plot_df['grover_success_prob_mean'] +
                         self.plot_df['grover_success_prob_std']) * 100,
                        alpha=0.2, color='#16A085')

        ax.axhline(y=50, color='red', linestyle='--',
                   linewidth=2, label='50% Threshold', alpha=0.7)

        ax.set_xlabel('Database Size (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Probability (%)',
                      fontsize=12, fontweight='bold')
        ax.set_title("Grover's Algorithm Success Rate",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/success_probability.png",
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/success_probability.png")
        plt.close()

    def plot_complexity_comparison(self):
        """Create comprehensive complexity comparison figure."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Query Count Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.plot_df['N'], self.plot_df['classical_queries_mean'],
                 marker='o', linewidth=2.5, markersize=8, label='Classical', color='#E74C3C')
        ax1.plot(self.plot_df['N'], self.plot_df['grover_queries_mean'],
                 marker='s', linewidth=2.5, markersize=8, label='Grover', color='#3498DB')
        ax1.set_xlabel('Database Size (N)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Query Count', fontsize=11, fontweight='bold')
        ax1.set_title('A) Query Complexity Comparison',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Log-Log Plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.loglog(self.plot_df['N'], self.plot_df['classical_queries_mean'],
                   'o-', linewidth=2.5, markersize=8, label='Classical O(N)', color='#E74C3C', base=2)
        ax2.loglog(self.plot_df['N'], self.plot_df['grover_queries_mean'],
                   's-', linewidth=2.5, markersize=8, label='Grover O(√N)', color='#3498DB', base=2)

        # Add slope indicators
        N_vals = self.plot_df['N'].values
        if len(N_vals) >= 2:
            # O(N) reference line
            ref_classical = N_vals / 2
            ax2.loglog(N_vals, ref_classical, '--', color='#E74C3C',
                       alpha=0.5, label='O(N) theory', base=2)

            # O(√N) reference line
            ref_grover = (np.pi / 4) * np.sqrt(N_vals)
            ax2.loglog(N_vals, ref_grover, '--', color='#3498DB',
                       alpha=0.5, label='O(√N) theory', base=2)

        ax2.set_xlabel('Database Size (N)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Query Count', fontsize=11, fontweight='bold')
        ax2.set_title('B) Log-Log Complexity Analysis',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which='both')

        # 3. Speedup Scaling
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.plot_df['N'], self.plot_df['speedup_mean'],
                 'o-', linewidth=2.5, markersize=8, color='#9B59B6', label='Observed Speedup')
        ax3.fill_between(self.plot_df['N'],
                         self.plot_df['speedup_mean'] -
                         self.plot_df['speedup_std'],
                         self.plot_df['speedup_mean'] +
                         self.plot_df['speedup_std'],
                         alpha=0.2, color='#9B59B6')
        theoretical = np.sqrt(self.plot_df['N']) / 2
        ax3.plot(self.plot_df['N'], theoretical, '--',
                 linewidth=2, color='#E67E22', label='√N/2 Theory')
        ax3.set_xlabel('Database Size (N)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
        ax3.set_title('C) Quantum Speedup Scaling',
                      fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Success Probability
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.plot_df['N'], self.plot_df['grover_success_prob_mean'] * 100,
                 'o-', linewidth=2.5, markersize=8, color='#16A085', label='Success Rate')
        ax4.fill_between(self.plot_df['N'],
                         (self.plot_df['grover_success_prob_mean'] -
                          self.plot_df['grover_success_prob_std']) * 100,
                         (self.plot_df['grover_success_prob_mean'] +
                          self.plot_df['grover_success_prob_std']) * 100,
                         alpha=0.2, color='#16A085')
        ax4.axhline(y=50, color='red', linestyle='--',
                    linewidth=1.5, alpha=0.5, label='50% Threshold')
        ax4.set_xlabel('Database Size (N)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Success Probability (%)',
                       fontsize=11, fontweight='bold')
        ax4.set_title('D) Algorithm Reliability',
                      fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 105])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.savefig(f"{self.output_dir}/comprehensive_analysis.png",
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/comprehensive_analysis.png")
        plt.close()

    def create_all_visualizations(self):
        """Generate all publication figures."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60 + "\n")

        self.plot_query_complexity()
        self.plot_speedup_analysis()
        self.plot_success_probability()
        self.plot_complexity_comparison()

        print("\n✓ All visualizations generated successfully!")
        print(f"\nGenerated files:")
        print(f"  - {self.output_dir}/query_complexity.png")
        print(f"  - {self.output_dir}/speedup_analysis.png")
        print(f"  - {self.output_dir}/success_probability.png")
        print(f"  - {self.output_dir}/comprehensive_analysis.png")


if __name__ == "__main__":
    viz = ExperimentVisualizer("results/processed/experiment_summary.csv")
    viz.create_all_visualizations()
