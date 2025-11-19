"""
Grover's Algorithm Simulation
Created by: abdullah-ax
Date: 2025-11-19
"""

import numpy as np
import math

class GroverSimulator:
    """Quantum search using Grover's algorithm simulation"""
    
    def __init__(self, N, target):
        """
        Initialize Grover's algorithm simulator
        
        Args:
            N (int): Database size
            target (int): Index of target element
        """
        self.N = N
        self.target = target
        self.query_count = 0
        
        # Calculate number of qubits needed: ceil(log2(N))
        self.num_qubits = math.ceil(math.log2(N))
        
        # Calculate optimal iterations: π/4 * sqrt(N)
        self.optimal_iterations = int(round((math.pi / 4) * math.sqrt(N)))
        
        # Initialize quantum state vector (2^num_qubits states)
        self.state_size = 2 ** self.num_qubits
        self.state = np.zeros(self.state_size, dtype=complex)
        
        self._initialize_superposition()
    
    def _initialize_superposition(self):
        """Initialize quantum state to uniform superposition |ψ⟩ = (1/√N) Σ|x⟩"""
        amplitude = 1.0 / math.sqrt(self.N)
        
        # Set amplitudes for valid states
        for i in range(self.N):
            self.state[i] = amplitude
        
        # Remaining states are zero (if N is not a power of 2)
        for i in range(self.N, self.state_size):
            self.state[i] = 0.0
    
    def _apply_oracle(self):
        """
        Apply the Oracle operator (marks the target state)
        Oracle: O|x⟩ = -|x⟩ if x is target, otherwise |x⟩
        """
        self.query_count += 1  # Each oracle call is a query
        self.state[self.target] *= -1.0
    
    def _apply_diffusion(self):
        """
        Apply the Diffusion operator (amplitude amplification)
        Diffusion: D = 2|s⟩⟨s| - I
        """
        # Calculate average amplitude
        avg = np.sum(self.state[:self.N]) / self.N
        
        # Inversion about average
        for i in range(self.N):
            self.state[i] = 2 * avg - self.state[i]
    
    def _measure(self):
        """
        Measure the quantum state (collapse to classical result)
        
        Returns:
            int: Index of measured state
        """
        # Calculate probabilities: P(i) = |amplitude_i|^2
        probabilities = np.abs(self.state[:self.N]) ** 2
        
        # Normalize probabilities
        probabilities /= np.sum(probabilities)
        
        # Sample from probability distribution
        result = np.random.choice(self.N, p=probabilities)
        return result
    
    def search(self):
        """
        Run Grover's algorithm
        
        Returns:
            int: Index of found element
        """
        self.query_count = 0
        
        # Run Grover iterations
        for _ in range(self.optimal_iterations):
            self._apply_oracle()
            self._apply_diffusion()
        
        # Measure the final state
        return self._measure()
    
    def get_query_count(self):
        """Return the number of oracle queries"""
        return self.query_count
    
    def get_optimal_iterations(self):
        """Return the optimal number of iterations"""
        return self.optimal_iterations
    
    def print_stats(self):
        """Print search statistics"""
        print(f"Grover's Algorithm Statistics:")
        print(f"  Database Size (N): {self.N}")
        print(f"  Number of Qubits: {self.num_qubits}")
        print(f"  Target Index: {self.target}")
        print(f"  Optimal Iterations: {self.optimal_iterations}")
        print(f"  Oracle Queries: {self.query_count}")
        print(f"  Theoretical Complexity: O(√N) = {int(math.sqrt(self.N))}")


if __name__ == "__main__":
    # Test Grover's algorithm
    grover = GroverSimulator(16, 8)
    result = grover.search()
    grover.print_stats()
    print(f"  Target found at index: {result}")