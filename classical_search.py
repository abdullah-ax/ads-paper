"""
Classical Linear Search Implementation
Baseline comparison for Grover's algorithm experiment

Authors: abdullah-ax, linah1604
Date: 2025-11-19
"""

from typing import Literal


class ClassicalSearch:
    """Classical linear search with query counting"""

    def __init__(self, N: int, target_index: int,
                 target_position: Literal['middle', 'end'] = 'middle'):
        """
        Initialize classical search.

        Args:
            N: Database size
            target_index: Target element index
            target_position: Position of target ('middle' or 'end')
        """
        self.N = N
        self.target_index = target_index
        self.target_position = target_position
        self.database = list(range(N))
        self.query_count = 0

    def search(self) -> int:
        """
        Perform linear search with query counting.

        Returns:
            Index of target (-1 if not found)
        """
        self.query_count = 0

        for i in range(self.N):
            self.query_count += 1
            if self.database[i] == self.target_index:
                return i

        return -1

    def get_query_count(self) -> int:
        """Return the number of queries made."""
        return self.query_count

    def get_expected_queries(self) -> float:
        """Return expected number of queries for this position."""
        if self.target_position == 'middle':
            return self.N / 2
        elif self.target_position == 'end':
            return self.N
        return self.N / 2

    def print_stats(self):
        """Print search statistics."""
        print(f"Classical Linear Search:")
        print(f"  Database Size: {self.N}")
        print(f"  Target: {self.target_index}")
        print(f"  Position: {self.target_position}")
        print(f"  Queries: {self.query_count}")
        print(f"  Expected: {self.get_expected_queries():.1f}")
        print(f"  Complexity: O(N) = {self.N}")
