"""
Fitness evaluator for genetic algorithm
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional
import gc

from ..config import GAConfig
from ..data import DataPreprocessor
from ..models import Net, Trainer
from .individual import Individual


class FitnessEvaluator:
    """Evaluates fitness of individuals using neural network training"""
    
    def __init__(self, 
                 config: GAConfig,
                 X_data: pd.DataFrame, 
                 y_data: pd.Series,
                 categorical_features: List[str], 
                 numerical_features: List[str]):
        """
        Initialize fitness evaluator
        
        Args:
            config: GA configuration
            X_data: Feature data
            y_data: Target data
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        """
        self.config = config
        self.X_data = X_data
        self.y_data = y_data
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        # Components
        self.preprocessor = DataPreprocessor(batch_size=config.batch_size)
        self.trainer = Trainer(
            max_epochs=config.neural_net_epochs,
            patience=config.neural_net_patience,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Caching for performance
        self.fitness_cache: Dict[tuple, float] = {}
        self.reference_loss: Optional[float] = None
        
        # Calculate reference loss using all features (L_full)
        self._calculate_reference_loss()
    
    def _calculate_reference_loss(self) -> None:
        """Calculate reference loss using all features for normalization (L_full)"""
        all_features_individual = Individual(len(self.X_data.columns))
        all_features_individual.chromosome = np.ones(len(self.X_data.columns), dtype=int)
        self.reference_loss = self._evaluate_without_cache(all_features_individual)
    
    def evaluate_individual(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual with caching
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value (lower is better)
        """
        # Check cache first
        chromosome_key = tuple(individual.chromosome)
        if chromosome_key in self.fitness_cache:
            individual.fitness = self.fitness_cache[chromosome_key]
            return individual.fitness
        
        # Calculate fitness
        fitness = self._evaluate_without_cache(individual)
        
        # Cache and store result
        self.fitness_cache[chromosome_key] = fitness
        individual.fitness = fitness
        
        return fitness
    
    def _evaluate_without_cache(self, individual: Individual) -> float:
        """
        Internal method to evaluate individual fitness using cross-validation
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value
        """
        # Get selected features
        selected_features = individual.get_selected_features()
        if not selected_features:  # No features selected
            return float('inf')
            
        X_selected = self.X_data.iloc[:, selected_features]
        
        # Filter feature lists for selected features
        selected_categorical = [f for f in self.categorical_features if f in X_selected.columns]
        selected_numerical = [f for f in self.numerical_features if f in X_selected.columns]
        
        # Perform 5-fold cross-validation
        cv_losses = self._cross_validate(X_selected, selected_categorical, selected_numerical)
        
        # Calculate final fitness: F(b) = L_test(b) + λ × (|b|/N)
        avg_cv_loss = np.mean(cv_losses)
        num_selected_features = len(selected_features)
        fitness = self._calculate_regularized_fitness(avg_cv_loss, num_selected_features)
        
        return fitness
    
    def _cross_validate(self, 
                       X_selected: pd.DataFrame,
                       selected_categorical: List[str], 
                       selected_numerical: List[str]) -> List[float]:
        """
        Perform 5-fold cross-validation
        
        Args:
            X_selected: Selected features data
            selected_categorical: Selected categorical features
            selected_numerical: Selected numerical features
            
        Returns:
            List of validation losses for each fold
        """
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds, 
            shuffle=True, 
            random_state=self.config.random_seed
        )
        
        fold_losses = []
        
        for train_idx, val_idx in kfold.split(X_selected, self.y_data):
            # Split data for this fold
            X_train_fold = X_selected.iloc[train_idx]
            X_val_fold = X_selected.iloc[val_idx]
            y_train_fold = self.y_data.iloc[train_idx]
            y_val_fold = self.y_data.iloc[val_idx]
            
            # Reset preprocessor for each fold to avoid data leakage
            self.preprocessor.reset_preprocessors()
            
            # Preprocess data
            train_loader, val_loader, input_size = self.preprocessor.preprocess_fold_data(
                X_train_fold, X_val_fold, y_train_fold, y_val_fold,
                selected_categorical, selected_numerical
            )
            
            # Create and train model
            model = Net(
                input_size=input_size,
                hidden_sizes=self.config.hidden_sizes,
                output_size=self.config.output_size,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate
            )
            
            # Train model with early stopping (5 epochs patience)
            trained_model, training_result = self.trainer.train_model(model, train_loader, val_loader, verbose=0)
            best_val_loss = training_result['best_val_loss']
            fold_losses.append(best_val_loss)
            
            # Clean up
            del trained_model, model
            gc.collect()
        
        return fold_losses
    
    def _calculate_regularized_fitness(self, cv_loss: float, num_selected_features: int) -> float:
        """
        Calculate final fitness: F(b) = L_test(b) + λ × (|b|/N) where λ = α × L_full
        
        Args:
            cv_loss: Cross-validation loss (L_test)
            num_selected_features: Number of selected features (|b|)
            
        Returns:
            Regularized fitness value
        """
        # Feature ratio: |b|/N
        feature_ratio = num_selected_features / len(self.X_data.columns)
        
        # Penalty: λ × (|b|/N) where λ = α × L_full
        if self.reference_loss is not None and self.reference_loss > 0:
            penalty = self.config.alpha * self.reference_loss * feature_ratio
        else:
            penalty = self.config.alpha * feature_ratio
        
        return cv_loss + penalty
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the fitness cache"""
        return {
            'cache_size': len(self.fitness_cache),
            'cache_hits': len(self.fitness_cache),  # Approximation
            'total_evaluations': len(self.fitness_cache)
        }
    
    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get distribution statistics of cached fitness values"""
        if not self.fitness_cache:
            return {
                'min_fitness': 0.0,
                'max_fitness': 0.0,
                'mean_fitness': 0.0,
                'std_fitness': 0.0
            }
        
        fitness_values = list(self.fitness_cache.values())
        return {
            'min_fitness': float(np.min(fitness_values)),
            'max_fitness': float(np.max(fitness_values)),
            'mean_fitness': float(np.mean(fitness_values)),
            'std_fitness': float(np.std(fitness_values))
        } 