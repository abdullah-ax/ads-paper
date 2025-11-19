"""
Classical Linear Search Implementation
Created by: abdullah-ax
Date: 2025-11-19
"""

class ClassicalSearch:
    """Classical linear search with query counting"""
    
    def __init__(self, N, target):
        """
        Initialize classical search
        
        Args:
            N (int): Database size
            target (int): Target value to find
        """
        self.database = list(range(N))  # Database with values 0 to N-1
        self.target = target
        self.query_count = 0
    
    def search(self):
        """
        Perform linear search and count queries
        
        Returns:
            int: Index of target (-1 if not found)
        """
        self.query_count = 0
        
        # Linear search with query counting
        for i in range(len(self.database)):
            self.query_count += 1  # Each comparison is a query
            
            if self.database[i] == self.target:
                return i  # Target found
        
        return -1  # Target not found
    
    def get_query_count(self):
        """Return the number of queries made"""
        return self.query_count
    
    def print_stats(self):
        """Print search statistics"""
        print(f"Classical Linear Search Statistics:")
        print(f"  Database Size: {len(self.database)}")
        print(f"  Target Value: {self.target}")
        print(f"  Total Queries: {self.query_count}")
        print(f"  Complexity: O(N) = {len(self.database)}")


if __name__ == "__main__":
    # Test the classical search
    searcher = ClassicalSearch(16, 8)
    result = searcher.search()
    searcher.print_stats()
    print(f"  Target found at index: {result}")