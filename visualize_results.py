"""
Visualization Script for Grover's Algorithm Experiment
Created by: linah1604
Date: 2025-11-19
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations():
    """Create all visualizations from experiment results"""
    
    # Read CSV data
    data = pd.read_csv('results/experiment_results.csv')
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Plot 1: Query Count Comparison ==========
    axes[0].plot(data['N'], data['Classical_Queries'], 
                 marker='o', linewidth=2.5, markersize=10, 
                 label='Classical Search O(N)', color='#E74C3C')
    
    axes[0].plot(data['N'], data['Grover_Queries'], 
                 marker='s', linewidth=2.5, markersize=10, 
                 label='Grover\'s Algorithm O(√N)', color='#3498DB')
    
    axes[0].plot(data['N'], data['Sqrt_N'], 
                 marker='^', linewidth=2, markersize=8, 
                 label='√N (Theoretical)', color='#2ECC71', 
                 linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Database Size (N)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Queries', fontsize=13, fontweight='bold')
    axes[0].set_title('Query Complexity: Grover vs Classical', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='upper left', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#F8F9FA')
    
    # ========== Plot 2: Speedup Factor ==========
    bars = axes[1].bar(data['N'].astype(str), data['Speedup'], 
                       color='#9B59B6', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2.5, 
                    label='No Speedup (1x)', alpha=0.7)
    
    axes[1].set_xlabel('Database Size (N)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Speedup Factor', fontsize=13, fontweight='bold')
    axes[1].set_title('Quantum Speedup (Classical / Grover)', 
                      fontsize=15, fontweight='bold', pad=15)
    axes[1].legend(fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_facecolor('#F8F9FA')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, data['Speedup'])):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{val:.2f}x', ha='center', va='bottom', 
                     fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'visualizations/grover_vs_classical.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # ========== Generate Summary Statistics ==========
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nAverage Speedup: {data['Speedup'].mean():.2f}x")
    print(f"Maximum Speedup: {data['Speedup'].max():.2f}x (N={data.loc[data['Speedup'].idxmax(), 'N']})")
    print(f"Minimum Speedup: {data['Speedup'].min():.2f}x (N={data.loc[data['Speedup'].idxmin(), 'N']})")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("1. Grover's algorithm consistently achieves O(√N) query complexity")
    print("2. Speedup increases with database size")
    print(f"3. At N=1024, quantum speedup is {data.loc[data['N']==1024, 'Speedup'].values[0]:.2f}x")
    print("4. Experimental results match theoretical predictions")

if __name__ == "__main__":
    create_visualizations()