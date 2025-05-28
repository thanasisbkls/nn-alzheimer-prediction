"""
Fitness evaluator for genetic algorithm individuals

This module contains the fitness evaluation logic for the Alzheimer's disease
feature selection genetic algorithm using the fixed weights approach.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split

from ..config import GAConfig
from ..data import DataPreprocessor
from ..models import Net, Trainer
from ..utils import setup_logger
from .individual import Individual


class FitnessEvaluator:
    """
    Evaluates fitness of individuals using fixed neural network weights approach
    """
    
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
        
        # Fitness caching
        self.fitness_cache: Dict[str, float] = {}
        
        # Reference model and preprocessing
        self.reference_model: Optional[Net] = None
        self.reference_loss: Optional[float] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        
        # Logging
        self.logger = setup_logger(__name__)
        
        # Initialize reference model and preprocessing
        self._initialize_reference_model()
    
    def _initialize_reference_model(self) -> None:
        """Initialize reference model with all features using fixed weights approach"""
        self.logger.info("Initializing reference model with fixed weights approach...")
        
        # Split data into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_data, self.y_data, 
            test_size=0.2, 
            random_state=self.config.random_seed,
            stratify=self.y_data
        )
        
        # Further split training data into train/validation (80/20 of training data)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.config.random_seed,
            stratify=y_train
        )
        
        # Initialize and fit preprocessor on training data only
        self.preprocessor = DataPreprocessor(batch_size=self.config.batch_size)
        self.preprocessor.fit_preprocessors(X_train_split, self.categorical_features, self.numerical_features)
        
        # Transform training and validation data
        X_train_processed = self.preprocessor.transform_data(X_train_split, self.categorical_features, self.numerical_features)
        X_val_processed = self.preprocessor.transform_data(X_val_split, self.categorical_features, self.numerical_features)
        
        # Create reference model
        self.reference_model = Net(
            input_size=X_train_processed.shape[1],
            hidden_sizes=self.config.hidden_sizes,
            output_size=self.config.output_size,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate
        )
        
        # Create trainer
        trainer = Trainer(
            max_epochs=self.config.neural_net_epochs,
            patience=self.config.neural_net_patience,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Create data loaders for training with proper train/validation split
        train_loader = self.preprocessor.create_data_loader(X_train_processed, y_train_split, shuffle=True)
        val_loader = self.preprocessor.create_data_loader(X_val_processed, y_val_split, shuffle=True)
        
        # Train the model
        self.logger.info("Training reference model...")
        trained_model, _ = trainer.train_model(
            model=self.reference_model,
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=1
        )
        
        self.reference_model = trained_model
        
        # Evaluate on test set to get reference loss
        X_test_processed = self.preprocessor.transform_data(X_test, self.categorical_features, self.numerical_features)
        test_loader = self.preprocessor.create_data_loader(X_test_processed, y_test, shuffle=True)
        
        test_metrics = trainer.evaluate_model(self.reference_model, test_loader)
        self.reference_loss = test_metrics['loss']
        
        self.logger.info(f"Reference model trained. Test loss (L_full): {self.reference_loss:.4f}")
        
        # Store test data for individual evaluation
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate_individual(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value (lower is better)
        """
        # Check cache first
        chromosome_key = individual.get_chromosome_string()
        if chromosome_key in self.fitness_cache:
            individual.fitness = self.fitness_cache[chromosome_key]
            return individual.fitness
        
        # Get selected features
        selected_feature_indices = individual.get_selected_features()
        
        # Convert selected feature indices to feature names
        selected_feature_names = [self.X_data.columns[i] for i in selected_feature_indices]
        
        # Set context for weight copying (store selected features for sophisticated mapping)
        self._current_selected_features = selected_feature_names
        
        # Transform test data with selected features
        X_test_processed = self.preprocessor.transform_data(
            self.X_test, self.categorical_features, self.numerical_features, 
            selected_features=selected_feature_names
        )
        
        # Create test data loader
        test_loader = self.preprocessor.create_data_loader(X_test_processed, self.y_test, shuffle=True)
        
        # Create model with adjusted input size for selected features
        model = Net(
            input_size=X_test_processed.shape[1],
            hidden_sizes=self.config.hidden_sizes,
            output_size=self.config.output_size,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate
        )
        
        # Convert to double precision to match data type
        model.double()
        
        # Copy compatible weights from reference model using sophisticated mapping
        self._copy_compatible_weights(model, self.reference_model)
        
        # Clean up context
        delattr(self, '_current_selected_features')
        
        # Create trainer for evaluation
        trainer = Trainer(
            max_epochs=self.config.neural_net_epochs,
            patience=self.config.neural_net_patience,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(model, test_loader)
        test_loss = test_metrics['loss']
        
        # Calculate fitness using the specified formula
        # F(b) = L_test(b) + α × L_full × (|b|/N)
        num_selected = len(selected_feature_indices)
        total_features = len(self.X_data.columns)
        
        feature_penalty = self.config.alpha * self.reference_loss * (num_selected / total_features)
        fitness = test_loss + feature_penalty
        
        # Store in cache and individual
        individual.fitness = fitness
        self.fitness_cache[chromosome_key] = fitness
        
        self.logger.debug(f"Evaluated individual: {num_selected} features, "
                         f"test_loss={test_loss:.4f}, penalty={feature_penalty:.4f}, "
                         f"fitness={fitness:.4f}")
        
        return fitness
    
    def _copy_compatible_weights(self, target_model: Net, source_model: Net) -> None:
        """
        Copy compatible weights from source model to target model using Dynamic Input Layer Creation approach.

        This method implements sophisticated weight mapping between the reference model (trained on all features)
        and the target model (using selected features). It creates a proper mapping between original and 
        processed features, accounting for one-hot encoding expansion.
        
        Args:
            target_model: Model to copy weights to (for selected features)
            source_model: Reference model to copy weights from (trained on all features)
        """
        # Get the feature mapping between original and processed features
        feature_mapping = self._create_feature_mapping()
        
        target_state = target_model.state_dict()
        source_state = source_model.state_dict()
        
        # Get the selected features for this individual
        if not hasattr(self, '_current_selected_features'):
            raise ValueError("Weight copying requires selected features context. "
                           "This method should only be called from evaluate_individual.")
            
        selected_feature_names = self._current_selected_features
        
        # Create mapping from selected original features to their processed feature indices
        selected_processed_indices = []
        for original_feature in selected_feature_names:
            if original_feature in feature_mapping:
                selected_processed_indices.extend(feature_mapping[original_feature])
        
        # Sort indices to maintain consistent ordering
        selected_processed_indices = sorted(selected_processed_indices)
        
        self.logger.debug(f"Mapping {len(selected_feature_names)} original features to "
                         f"{len(selected_processed_indices)} processed features")
        
        # Copy weights layer by layer
        for name, target_param in target_state.items():
            if name in source_state:
                source_param = source_state[name]
                
                if name.endswith('.0.weight'):  # First layer weight matrix
                    # Extract relevant weights for selected features
                    if len(source_param.shape) == 2:  # Weight matrix [output_size, input_size]
                        # Extract columns corresponding to selected processed features
                        selected_weights = source_param[:, selected_processed_indices]
                        
                        # Ensure dimensions match
                        if selected_weights.shape == target_param.shape:
                            target_state[name] = selected_weights.clone()
                            self.logger.debug(f"Successfully copied {name} with exact match: {selected_weights.shape}")
                        else:
                            self.logger.warning(f"Dimension mismatch for {name}: "
                                              f"expected {target_param.shape}, got {selected_weights.shape}")
                            # Fallback: copy what we can
                            min_rows = min(target_param.shape[0], selected_weights.shape[0])
                            min_cols = min(target_param.shape[1], selected_weights.shape[1])
                            target_state[name][:min_rows, :min_cols] = selected_weights[:min_rows, :min_cols].clone()
                
                elif name.endswith('.0.bias'):  # First layer bias
                    # First layer bias should be copied directly (same hidden layer size)
                    if source_param.shape == target_param.shape:
                        target_state[name] = source_param.clone()
                    else:
                        min_size = min(target_param.shape[0], source_param.shape[0])
                        target_state[name][:min_size] = source_param[:min_size].clone()
                
                else:  # All other layers (hidden layers, output layer)
                    # Copy directly as they should have the same dimensions
                    if source_param.shape == target_param.shape:
                        target_state[name] = source_param.clone()
                    elif 'weight' in name and len(source_param.shape) == 2:
                        # Handle potential dimension mismatches in hidden layers
                        min_rows = min(target_param.shape[0], source_param.shape[0])
                        min_cols = min(target_param.shape[1], source_param.shape[1])
                        target_state[name][:min_rows, :min_cols] = source_param[:min_rows, :min_cols].clone()
                    elif 'bias' in name and len(source_param.shape) == 1:
                        min_size = min(target_param.shape[0], source_param.shape[0])
                        target_state[name][:min_size] = source_param[:min_size].clone()
        
        target_model.load_state_dict(target_state)
        
    def _create_feature_mapping(self) -> Dict[str, List[int]]:
        """
        Create mapping from original features to their corresponding processed feature indices.
        
        Returns:
            Dictionary mapping original feature names to lists of processed feature indices
        """
        if not self.preprocessor or not self.preprocessor._fitted:
            raise ValueError("Preprocessor must be fitted before creating feature mapping")
        
        # Get all processed features by transforming the full dataset
        X_all_processed = self.preprocessor.transform_data(
            self.X_data, self.categorical_features, self.numerical_features
        )
        processed_feature_names = list(X_all_processed.columns)
        
        feature_mapping = {}
        
        # Map categorical features to their one-hot encoded columns
        for original_feature in self.categorical_features:
            if original_feature in self.preprocessor._encoders:
                encoder = self.preprocessor._encoders[original_feature]
                categories = list(encoder.categories_[0])
                # Skip first category due to drop='first'
                encoded_names = [f"{original_feature}_{cat}" for cat in categories[1:]]
                
                # Find indices of these encoded features in the processed dataset
                indices = []
                for encoded_name in encoded_names:
                    if encoded_name in processed_feature_names:
                        indices.append(processed_feature_names.index(encoded_name))
                
                feature_mapping[original_feature] = indices
        
        # Map numerical features (they keep their original names)
        for original_feature in self.numerical_features:
            if original_feature in processed_feature_names:
                feature_mapping[original_feature] = [processed_feature_names.index(original_feature)]
        
        return feature_mapping
    

    def get_cache_size(self) -> int:
        """Get the current size of the fitness cache"""
        return len(self.fitness_cache)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.fitness_cache),
            'cache_hits': 0,  # Could be implemented with counters if needed
            'cache_misses': 0  # Could be implemented with counters if needed
        }
    
    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get fitness distribution statistics"""
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
    
    def clear_cache(self) -> None:
        """Clear the fitness cache"""
        self.fitness_cache.clear()
        self.logger.info("Fitness cache cleared") 