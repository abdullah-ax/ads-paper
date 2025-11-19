"""
Statistical Analysis Module
Validates experimental results with proper statistical tests

Authors: abdullah-ax
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple


class StatisticalAnalyzer:
    """Statistical validation of experimental results."""

    def __init__(self, results_file: str):
        """
        Initialize analyzer with processed results.

        Args:
            results_file: Path to processed experiment results CSV
        """
        self.df = pd.read_csv(results_file)
        self.analysis_results = {}

    def test_theoretical_complexity(self) -> Dict:
        """
        Test if observed query counts match theoretical predictions.

        Returns:
            Dictionary with test results
        """
        results = {}

        # Filter middle target for cleaner analysis
        middle_df = self.df[self.df['target_position'] == 'middle']

        # Test Classical: observed vs N/2
        classical_expected = middle_df['N'] / 2
        classical_observed = middle_df['classical_queries_mean']

        # Chi-square goodness of fit
        classical_chi2, classical_p = stats.chisquare(
            classical_observed, classical_expected
        )

        results['classical'] = {
            'test': 'chi-square',
            'chi2_statistic': classical_chi2,
            'p_value': classical_p,
            'matches_theory': classical_p > 0.05
        }

        # Test Grover: observed vs π/4 * sqrt(N)
        grover_expected = (np.pi / 4) * middle_df['sqrt_N']
        grover_observed = middle_df['grover_queries_mean']

        # Correlation test
        grover_corr, grover_p = stats.pearsonr(
            grover_expected, grover_observed)

        results['grover'] = {
            'test': 'pearson correlation',
            'correlation': grover_corr,
            'p_value': grover_p,
            'matches_theory': grover_corr > 0.99
        }

        self.analysis_results['complexity_test'] = results
        return results

    def test_speedup_scaling(self) -> Dict:
        """
        Analyze speedup scaling with database size.

        Returns:
            Dictionary with scaling analysis
        """
        middle_df = self.df[self.df['target_position'] == 'middle']

        # Log-log regression to test scaling
        log_N = np.log(middle_df['N'])
        log_speedup = np.log(middle_df['speedup_mean'])

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_N, log_speedup)

        results = {
            'slope': slope,
            'expected_slope': 0.5,  # sqrt(N) scaling
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'matches_sqrt_scaling': abs(slope - 0.5) < 0.1
        }

        self.analysis_results['speedup_scaling'] = results
        return results

    def confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Calculate confidence intervals for speedup measurements.

        Args:
            confidence: Confidence level (default 95%)

        Returns:
            DataFrame with confidence intervals
        """
        middle_df = self.df[self.df['target_position'] == 'middle'].copy()

        # Calculate confidence intervals
        z_score = stats.norm.ppf((1 + confidence) / 2)

        middle_df['speedup_ci_lower'] = (
            middle_df['speedup_mean'] -
            z_score * middle_df['speedup_std'] / np.sqrt(10)  # 10 repetitions
        )
        middle_df['speedup_ci_upper'] = (
            middle_df['speedup_mean'] +
            z_score * middle_df['speedup_std'] / np.sqrt(10)
        )

        return middle_df[['N', 'speedup_mean', 'speedup_ci_lower', 'speedup_ci_upper']]

    def generate_report(self, output_file: str = "results/statistical_report.txt"):
        """Generate comprehensive statistical report."""
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("STATISTICAL VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Complexity tests
            complexity = self.test_theoretical_complexity()
            f.write("1. THEORETICAL COMPLEXITY VALIDATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Classical Search (O(N)):\n")
            f.write(
                f"  Chi-square: {complexity['classical']['chi2_statistic']:.4f}\n")
            f.write(f"  p-value: {complexity['classical']['p_value']:.4f}\n")
            f.write(
                f"  Matches Theory: {complexity['classical']['matches_theory']}\n\n")

            f.write(f"Grover's Algorithm (O(√N)):\n")
            f.write(
                f"  Correlation: {complexity['grover']['correlation']:.4f}\n")
            f.write(f"  p-value: {complexity['grover']['p_value']:.4e}\n")
            f.write(
                f"  Matches Theory: {complexity['grover']['matches_theory']}\n\n")

            # Speedup scaling
            scaling = self.test_speedup_scaling()
            f.write("2. SPEEDUP SCALING ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Observed Slope: {scaling['slope']:.4f}\n")
            f.write(f"Expected Slope (√N): {scaling['expected_slope']:.4f}\n")
            f.write(f"R²: {scaling['r_squared']:.4f}\n")
            f.write(
                f"Matches √N Scaling: {scaling['matches_sqrt_scaling']}\n\n")

            # Confidence intervals
            ci_df = self.confidence_intervals()
            f.write("3. 95% CONFIDENCE INTERVALS\n")
            f.write("-" * 70 + "\n")
            f.write(ci_df.to_string(index=False))
            f.write("\n\n")

        print(f"✓ Statistical report saved to: {output_file}")


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer("results/processed/experiment_summary.csv")
    analyzer.generate_report()
