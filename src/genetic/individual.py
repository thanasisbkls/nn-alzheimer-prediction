"""
Individual representation for genetic algorithm

This module contains the individual representation for the genetic algorithm.
"""

import numpy as np
import random
from typing import List, Optional


class Individual:
    """Represents a binary chromosome for feature selection"""
    
    def __init__(self, num_features: int, chromosome: Optional[np.ndarray] = None):
        """
        Initialize individual with binary chromosome
        
        Args:
            num_features: Number of features in the dataset (default 35 for Alzheimer's dataset)
            chromosome: Pre-defined chromosome. If None, generates random one
        """
        self.num_features = num_features
        self.chromosome = chromosome if chromosome is not None else self._generate_random_chromosome()
        self.fitness: Optional[float] = None
    
    def _generate_random_chromosome(self) -> np.ndarray:
        """Generate a random binary chromosome with k features selected (k uniformly sampled 1-35)"""
        k = random.randint(1, self.num_features)
        chromosome = np.zeros(self.num_features, dtype=int)
        selected_indices = random.sample(range(self.num_features), k)
        chromosome[selected_indices] = 1
        return chromosome
    
    @property
    def num_selected_features(self) -> int:
        """Get the number of selected features (1s in chromosome)"""
        return int(np.sum(self.chromosome))
    
    def get_selected_features(self) -> List[int]:
        """
        Get indices of selected features for neural network training
        
        Returns:
            List of feature indices where chromosome value is 1
        """
        return list(np.where(self.chromosome == 1)[0])
    
    def copy(self) -> 'Individual':
        """
        Create a deep copy of the individual
        
        Returns:
            New Individual instance with copied chromosome
        """
        new_individual = Individual(self.num_features, self.chromosome.copy())
        new_individual.fitness = self.fitness
        return new_individual
    
    def __str__(self) -> str:
        num_selected = self.num_selected_features
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual(features={num_selected}/{self.num_features}, fitness={fitness_str})" 