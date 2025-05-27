"""
Population management for genetic algorithm

This module contains the population management for the genetic algorithm.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .individual import Individual


class Population:
    """Manages a collection of individuals in the genetic algorithm"""
    
    def __init__(self, size: int, num_features: int):
        """
        Initialize population with random individuals
        
        Args:
            size: Number of individuals in population
            num_features: Number of features for each individual
        """
        self.size = size
        self.num_features = num_features
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
        # Initialize with random individuals
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Create initial population with random individuals"""
        self.individuals = [Individual(self.num_features) for _ in range(self.size)]
    
    def update_best_individual(self) -> None:
        """Update the best individual in the population (for elitism)"""
        if not self.individuals:
            return
        
        # Find individual with best (lowest) fitness
        evaluated_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not evaluated_individuals:
            return
        
        current_best = min(evaluated_individuals, key=lambda ind: ind.fitness)
        
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate population statistics
        
        Returns:
            Dictionary with population statistics
        """
        evaluated_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        
        if not evaluated_individuals:
            return {
                'best_fitness': None,
                'worst_fitness': None,
                'avg_fitness': None,
                'std_fitness': None,
                'diversity': 0.0,
                'evaluated_count': 0
            }
        
        fitness_values = [ind.fitness for ind in evaluated_individuals]
        
        # Calculate diversity as average Hamming distance between chromosomes
        diversity = self._calculate_diversity()
        
        return {
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'avg_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'diversity': diversity,
            'evaluated_count': len(evaluated_individuals)
        }
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity as average Hamming distance"""
        if len(self.individuals) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                # Calculate Hamming distance between chromosomes
                distance = np.sum(self.individuals[i].chromosome != self.individuals[j].chromosome)
                total_distance += distance / self.num_features  # Normalize by chromosome length
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def replace_individuals(self, new_individuals: List[Individual]) -> None:
        """
        Replace current population with new individuals
        
        Args:
            new_individuals: List of new individuals for next generation
        """
        self.individuals = new_individuals
        self.generation += 1
    
    def __len__(self) -> int:
        """Return population size"""
        return len(self.individuals)
    
    def __iter__(self):
        """Make population iterable"""
        return iter(self.individuals) 