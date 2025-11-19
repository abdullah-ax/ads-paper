"""
Grover's Algorithm Implementation using Qiskit
Based on Qiskit documentation and Nielsen & Chuang's textbook

Reference:
- Qiskit Grover's Algorithm Tutorial:
  https://qiskit.org/textbook/ch-algorithms/grover.html
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and 
  Quantum Information. Cambridge University Press.

Authors: abdullah-ax, linah1604, 
Date: 2025-11-19
"""

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class QiskitGroverSearch:
    """
    Grover's Algorithm implementation using Qiskit for unstructured search.

    Attributes:
        num_qubits (int): Number of qubits required
        N (int): Database size (search space)
        target_index (int): Index of the target element
        optimal_iterations (int): Theoretical optimal iterations
    """

    def __init__(self, num_qubits: int, target_index: int):
        """
        Initialize Grover's algorithm.

        Args:
            num_qubits: Number of qubits (N = 2^num_qubits)
            target_index: Target element index in search space
        """
        self.num_qubits = num_qubits
        self.N = 2 ** num_qubits
        self.target_index = target_index

        # Calculate optimal iterations: π/4 * sqrt(N)
        self.optimal_iterations = int(np.round(np.pi / 4 * np.sqrt(self.N)))

        # Initialize simulator
        self.simulator = AerSimulator()

    def create_oracle(self) -> QuantumCircuit:
        """
        Create the Oracle circuit that marks the target state.
        Oracle: |x⟩ → -|x⟩ if x = target, |x⟩ otherwise

        Returns:
            QuantumCircuit: Oracle circuit
        """
        oracle_circuit = QuantumCircuit(self.num_qubits, name='Oracle')

        # Convert target index to binary
        target_binary = format(self.target_index, f'0{self.num_qubits}b')

        # Flip qubits where target has 0 (for MCZ gate preparation)
        for i, bit in enumerate(target_binary):
            if bit == '0':
                oracle_circuit.x(i)

        # Multi-controlled Z gate (marks target with phase flip)
        if self.num_qubits == 1:
            oracle_circuit.z(0)
        elif self.num_qubits == 2:
            oracle_circuit.cz(0, 1)
        else:
            # For n > 2, use multi-controlled Z
            oracle_circuit.h(self.num_qubits - 1)
            oracle_circuit.mcx(
                list(range(self.num_qubits - 1)), self.num_qubits - 1)
            oracle_circuit.h(self.num_qubits - 1)

        # Unflip qubits
        for i, bit in enumerate(target_binary):
            if bit == '0':
                oracle_circuit.x(i)

        return oracle_circuit

    def create_diffusion(self) -> QuantumCircuit:
        """
        Create the Diffusion operator (inversion about average).
        Diffusion: D = 2|s⟩⟨s| - I

        Returns:
            QuantumCircuit: Diffusion circuit
        """
        diffusion_circuit = QuantumCircuit(self.num_qubits, name='Diffusion')

        # Apply Hadamard gates
        diffusion_circuit.h(range(self.num_qubits))

        # Apply X gates
        diffusion_circuit.x(range(self.num_qubits))

        # Multi-controlled Z gate
        if self.num_qubits == 1:
            diffusion_circuit.z(0)
        elif self.num_qubits == 2:
            diffusion_circuit.cz(0, 1)
        else:
            diffusion_circuit.h(self.num_qubits - 1)
            diffusion_circuit.mcx(
                list(range(self.num_qubits - 1)), self.num_qubits - 1)
            diffusion_circuit.h(self.num_qubits - 1)

        # Apply X gates
        diffusion_circuit.x(range(self.num_qubits))

        # Apply Hadamard gates
        diffusion_circuit.h(range(self.num_qubits))

        return diffusion_circuit

    def build_circuit(self) -> QuantumCircuit:
        """
        Build the complete Grover's algorithm circuit.

        Returns:
            QuantumCircuit: Complete Grover circuit
        """
        # Create quantum and classical registers
        qr = QuantumRegister(self.num_qubits, 'q')
        cr = ClassicalRegister(self.num_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Initialize to uniform superposition
        circuit.h(range(self.num_qubits))
        circuit.barrier()

        # Apply Grover iterations
        oracle = self.create_oracle()
        diffusion = self.create_diffusion()

        for _ in range(self.optimal_iterations):
            circuit.compose(oracle, inplace=True)
            circuit.barrier()
            circuit.compose(diffusion, inplace=True)
            circuit.barrier()

        # Measure all qubits
        circuit.measure(qr, cr)

        return circuit

    def search(self, shots: int = 1000) -> Tuple[int, float]:
        """
        Execute Grover's search algorithm.

        Args:
            shots: Number of measurement repetitions

        Returns:
            Tuple of (measured_index, success_probability)
        """
        # Build and execute circuit
        circuit = self.build_circuit()
        job = self.simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Find most frequent measurement (reversed bit order)
        max_count = 0
        measured_index = 0

        for bitstring, count in counts.items():
            # Qiskit uses reversed bit ordering
            index = int(bitstring[::-1], 2)
            if count > max_count:
                max_count = count
                measured_index = index

        success_probability = max_count / shots

        return measured_index, success_probability

    def get_oracle_queries(self) -> int:
        """
        Return the number of oracle queries made.
        In Grover's algorithm, this equals the number of iterations.
        """
        return self.optimal_iterations

    def get_optimal_iterations(self) -> int:
        """Return the theoretical optimal number of iterations."""
        return self.optimal_iterations

    def print_stats(self):
        """Print algorithm statistics."""
        print(f"Qiskit Grover's Algorithm Statistics:")
        print(f"  Database Size (N): {self.N}")
        print(f"  Number of Qubits: {self.num_qubits}")
        print(f"  Target Index: {self.target_index}")
        print(f"  Optimal Iterations: {self.optimal_iterations}")
        print(f"  Oracle Queries: {self.get_oracle_queries()}")
        print(f"  Theoretical Complexity: O(√N) = {int(np.sqrt(self.N))}")


if __name__ == "__main__":
    # Test with small example
    print("Testing Qiskit Grover's Algorithm\n")

    # 4-qubit example (N = 16)
    num_qubits = 4
    target = 11

    grover = QiskitGroverSearch(num_qubits, target)
    found_index, prob = grover.search(shots=10000)

    grover.print_stats()
    print(f"\n  Measured Index: {found_index}")
    print(f"  Success Probability: {prob:.2%}")
    print(f"  Correct: {found_index == target}")

    # Verify query count
    expected_queries = int(np.round((np.pi / 4) * np.sqrt(16)))
    actual_queries = grover.get_oracle_queries()
    print(f"\n  Expected Queries: {expected_queries}")
    print(f"  Actual Queries: {actual_queries}")
    print(f"  Match: {expected_queries == actual_queries}")
