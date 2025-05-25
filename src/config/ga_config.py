"""
Genetic Algorithm Configuration

This module contains the configuration parameters for the Genetic Algorithm.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class GAConfig:
    """Configuration parameters for the Genetic Algorithm"""
    
    # Population parameters
    population_size: int = 20
    max_generations: int = 1000
    early_stopping_patience: int = 50  # B2 specification: 50 generations
    min_improvement_threshold: float = 0.01  # Minimum improvement percentage (1%)
    
    # Genetic operator parameters
    crossover_rate: float = 0.6
    mutation_rate: float = 0.01
    tournament_size: int = 3
    
    # Fitness function parameters
    alpha: float = 0.3  # Feature count penalty weight
    n_folds: int = 5
    
    # Neural network parameters
    neural_net_patience: int = 5
    neural_net_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    dropout_rate: float = 0.0
    
    # Neural network architecture
    activation: str = 'relu'
    output_size: int = 1  # Binary classification
    hidden_sizes: List[int] = None
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Set default hidden layer sizes"""
        if self.hidden_sizes is None:
            self.hidden_sizes = [16, 8]
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Population parameters
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.max_generations <= 0:
            raise ValueError("Max generations must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")
        if not 0 <= self.min_improvement_threshold <= 1:
            raise ValueError("Minimum improvement threshold must be between 0 and 1")
        
        # Genetic operator parameters
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        
        # Fitness function parameters
        if self.alpha < 0:
            raise ValueError("Alpha must be non-negative")
        if self.n_folds <= 1:
            raise ValueError("Number of folds must be greater than 1")
        
        # Neural network training parameters
        if self.neural_net_patience <= 0:
            raise ValueError("Neural network patience must be positive")
        if self.neural_net_epochs <= 0:
            raise ValueError("Neural network epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.momentum < 0:
            raise ValueError("Momentum must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1 (exclusive)")
        
        # Neural network architecture parameters
        if self.activation not in ['relu', 'tanh', 'silu']:
            raise ValueError("Activation must be one of: 'relu', 'tanh', 'silu'")
        if self.output_size <= 0:
            raise ValueError("Output size must be positive")
        if not self.hidden_sizes or not isinstance(self.hidden_sizes, list):
            raise ValueError("Hidden sizes must be a non-empty list")
        if not all(isinstance(size, int) and size > 0 for size in self.hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers")
        
        # Reproducibility parameters
        if not isinstance(self.random_seed, int):
            raise ValueError("Random seed must be an integer")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement_threshold': self.min_improvement_threshold,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'tournament_size': self.tournament_size,
            'alpha': self.alpha,
            'n_folds': self.n_folds,
            'neural_net_patience': self.neural_net_patience,
            'neural_net_epochs': self.neural_net_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'output_size': self.output_size,
            'hidden_sizes': self.hidden_sizes,
            'random_seed': self.random_seed
        } 