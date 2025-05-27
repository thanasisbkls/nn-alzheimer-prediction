"""
Genetic operators for feature selection

This module contains the genetic operators for the genetic algorithm.
"""

import random
import numpy as np
from typing import List, Tuple
from .individual import Individual


class TournamentSelection:
    """Tournament selection with tournament size k (default 3)"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection"""
        # Handle edge case where tournament size > population size
        actual_tournament_size = min(self.tournament_size, len(population))
        
        # Select tournament participants
        tournament = random.sample(population, actual_tournament_size)
        
        # Return best individual (lowest fitness)
        return min(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('inf'))


class UniformCrossover:
    """Uniform crossover with 50% probability per gene"""
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform uniform crossover between two parents"""
        # Create child chromosomes
        child1_chromosome = np.zeros_like(parent1.chromosome)
        child2_chromosome = np.zeros_like(parent2.chromosome)
        
        # Perform uniform crossover (50% probability per gene)
        for i in range(parent1.num_features):
            if random.random() < 0.5:
                # Swap genes
                child1_chromosome[i] = parent2.chromosome[i]
                child2_chromosome[i] = parent1.chromosome[i]
            else:
                # Keep original genes
                child1_chromosome[i] = parent1.chromosome[i]
                child2_chromosome[i] = parent2.chromosome[i]
        
        # Create children
        child1 = Individual(parent1.num_features, child1_chromosome)
        child2 = Individual(parent2.num_features, child2_chromosome)
        
        return child1, child2


class BitFlipMutation:
    """Bit-flip mutation operator"""
    
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply bit-flip mutation to an individual"""
        mutated = individual.copy()
        
        # Apply mutation to each bit
        for i in range(mutated.num_features):
            if random.random() < self.mutation_rate:
                # Flip bit
                mutated.chromosome[i] = 1 - mutated.chromosome[i]
        
        # Ensure at least one feature is selected
        if np.sum(mutated.chromosome) == 0:
            # Randomly select one feature
            random_index = random.randint(0, mutated.num_features - 1)
            mutated.chromosome[random_index] = 1
        
        return mutated 