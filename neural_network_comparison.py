#!/usr/bin/env python3
"""
Neural Network Comparison Module

Compare the performance of the following models:
1. Neural network trained with GA-selected features (optimal from experiments)
2. Neural network trained with all features (reference model)

Analyzes:
- Generalizability of the two networks
- Effect of feature selection on NN performance  
- Possibility of overfitting to test data
- Performance after retraining with entire training set
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import GAConfig
from src.data import AlzheimerDataLoader, DataPreprocessor
from src.models import Net, Trainer
from src.utils import setup_logger, set_seed

# Import new visualization and reporting modules
from src.visualization.nn_visualizer import NNVisualizer
from src.reporting.nn_reporter import NNReporter


class NeuralNetworkComparison:
    """
    Comprehensive comparison between GA-optimized and reference neural networks
    """
    
    def __init__(self, data_file: str = "alzheimers_disease_data.csv"):
        """Initialize the comparison module"""
        self.data_file = data_file
        self.logger = setup_logger(__name__)
        
        # Create output directories
        self.results_dir = Path("comp_analysis_results")
        self.plots_dir = Path("comp_analysis_plots")
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Initialize visualization and reporting modules
        self.visualizer = NNVisualizer(self.plots_dir)
        self.reporter = NNReporter(self.results_dir, self.X_data)
        
        # Results storage
        self.comparison_results = {}
        
    def _load_data(self):
        """Load and prepare the Alzheimer's dataset"""
        self.logger.info(f"Loading data from {self.data_file}")
        
        data_loader = AlzheimerDataLoader()
        self.X_data, self.y_data, self.categorical_features, self.numerical_features = \
            data_loader.load_data(self.data_file)
        
        self.logger.info(f"Data loaded: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} features")
        self.logger.info(f"Categorical features: {len(self.categorical_features)}")
        self.logger.info(f"Numerical features: {len(self.numerical_features)}")
        
    def load_best_ga_results(self, results_file: str) -> Dict[str, Any]:
        """Load the best GA experiment results"""        
        self.logger.info(f"Loading GA results from {results_file}")
        
        with open(results_file, 'r') as f:
            ga_results = json.load(f)
        
        # Find the best performing experiment
        best_experiment = None
        best_fitness = float('inf')
        
        for exp_name, exp_data in ga_results['experiments'].items():
            if 'statistics' in exp_data:
                mean_fitness = exp_data['statistics']['best_fitness']['mean']
                if mean_fitness < best_fitness:
                    best_fitness = mean_fitness
                    best_experiment = exp_data
        
        if best_experiment is None:
            raise ValueError("No valid experiment results found")
        
        self.logger.info(f"Best experiment: {best_experiment['experiment_info']['name']}")
        self.logger.info(f"Best mean fitness: {best_fitness:.4f}")
        
        return best_experiment
    
    def extract_best_individual_features(self, best_experiment: Dict[str, Any]) -> List[str]:
        """Extract the feature names selected by the best individual"""
        
        # Find the best individual across all runs
        best_individual = None
        best_fitness = float('inf')
        
        for run_data in best_experiment['runs']:
            if run_data.get('success', False):
                individual = run_data.get('best_individual', {})
                fitness = individual.get('fitness', float('inf'))
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual
        
        if best_individual is None:
            raise ValueError("No successful runs found in best experiment")
        
        # Extract selected feature indices
        selected_indices = best_individual.get('selected_features', [])
        
        # Map indices to feature names
        feature_names = list(self.X_data.columns)
        selected_features = [feature_names[i] for i in selected_indices]
        
        self.logger.info(f"Best individual selected {len(selected_features)} features:")
        self.logger.info(f"Features: {selected_features}")
        
        return selected_features
    
    def create_train_test_splits(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Create train/test splits for comparison"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_data, self.y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_data
        )
        
        self.logger.info(f"Train set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_neural_network(self, config: GAConfig, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           selected_features: Optional[List[str]] = None,
                           model_name: str = "model") -> Dict[str, Any]:
        """Train a neural network with specified features"""        
        set_seed(config.random_seed)
        
        # Split training data into train/validation (80/20)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=config.random_seed,
            stratify=y_train
        )
        
        # Initialize preprocessor and fit on training split only
        preprocessor = DataPreprocessor(batch_size=config.batch_size)
        preprocessor.fit_preprocessors(X_train_split, self.categorical_features, self.numerical_features)
        
        # Transform data
        if selected_features is not None:
            X_train_processed = preprocessor.transform_data(
                X_train_split, self.categorical_features, self.numerical_features, 
                selected_features=selected_features
            )
            X_val_processed = preprocessor.transform_data(
                X_val_split, self.categorical_features, self.numerical_features,
                selected_features=selected_features
            )
            X_test_processed = preprocessor.transform_data(
                X_test, self.categorical_features, self.numerical_features,
                selected_features=selected_features
            )
            self.logger.info(f"{model_name}: Using {len(selected_features)} selected features")
        else:
            X_train_processed = preprocessor.transform_data(
                X_train_split, self.categorical_features, self.numerical_features
            )
            X_val_processed = preprocessor.transform_data(
                X_val_split, self.categorical_features, self.numerical_features
            )
            X_test_processed = preprocessor.transform_data(
                X_test, self.categorical_features, self.numerical_features
            )
            self.logger.info(f"{model_name}: Using all {X_train_processed.shape[1]} features")
        
        # Create data loaders with proper train/validation split
        train_loader = preprocessor.create_data_loader(X_train_processed, y_train_split, shuffle=True)
        val_loader = preprocessor.create_data_loader(X_val_processed, y_val_split, shuffle=True)
        test_loader = preprocessor.create_data_loader(X_test_processed, y_test, shuffle=True)
        
        # Create model
        model = Net(
            input_size=X_train_processed.shape[1],
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size,
            activation=config.activation,
            dropout_rate=config.dropout_rate
        )
        
        # Create trainer
        trainer = Trainer(
            max_epochs=config.neural_net_epochs,
            patience=config.neural_net_patience,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Train model
        self.logger.info(f"Training {model_name}...")
        trained_model, training_metrics = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=1
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(trained_model, test_loader)
        
        # Get detailed predictions for analysis
        predictions, probabilities = self._get_detailed_predictions(trained_model, test_loader)
        
        return {
            'model': trained_model,
            'preprocessor': preprocessor,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'processed_features': list(X_train_processed.columns),
            'input_size': X_train_processed.shape[1]
        }
    
    def _get_detailed_predictions(self, model: torch.nn.Module, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get detailed predictions and probabilities from model
        
        Provides individual-level predictions and probabilities for each sample in the test set.
        """
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
        
        return np.array(all_predictions), np.array(all_probabilities) 
    
    def compare_models(self, ga_results: Dict[str, Any], reference_results: Dict[str, Any],
                      y_test: pd.Series) -> Dict[str, Any]:
        """Compare GA-optimized model with reference model"""
        
        # Validate input data consistency
        if len(ga_results['predictions']) != len(y_test):
            raise ValueError(f"GA predictions length ({len(ga_results['predictions'])}) doesn't match test set length ({len(y_test)})")
        
        if len(reference_results['predictions']) != len(y_test):
            raise ValueError(f"Reference predictions length ({len(reference_results['predictions'])}) doesn't match test set length ({len(y_test)})")
        
        if len(ga_results['predictions']) != len(reference_results['predictions']):
            raise ValueError(f"GA and reference predictions have different lengths: {len(ga_results['predictions'])} vs {len(reference_results['predictions'])}")
        
        comparison = {
            'ga_model': {
                'test_accuracy': ga_results['test_metrics']['accuracy'],
                'test_loss': ga_results['test_metrics']['loss'],
                'test_mse': ga_results['test_metrics']['mse'],
                'num_features': ga_results['input_size'],
                'feature_names': ga_results['processed_features']
            },
            'reference_model': {
                'test_accuracy': reference_results['test_metrics']['accuracy'],
                'test_loss': reference_results['test_metrics']['loss'],
                'test_mse': reference_results['test_metrics']['mse'],
                'num_features': reference_results['input_size'],
                'feature_names': reference_results['processed_features']
            }
        }
        
        # Calculate performance differences
        comparison['performance_difference'] = {
            'accuracy_diff': ga_results['test_metrics']['accuracy'] - reference_results['test_metrics']['accuracy'],
            'loss_diff': ga_results['test_metrics']['loss'] - reference_results['test_metrics']['loss'],
            'mse_diff': ga_results['test_metrics']['mse'] - reference_results['test_metrics']['mse'],
            'feature_reduction': 1 - (ga_results['input_size'] / reference_results['input_size']),
            'feature_reduction_count': reference_results['input_size'] - ga_results['input_size']
        }
        
        # Detailed classification analysis
        y_test_array = y_test.values
        
        try:
            # GA model classification report
            ga_report = classification_report(
                y_test_array, ga_results['predictions'], 
                target_names=['No Alzheimer', 'Alzheimer'], 
                output_dict=True,
                zero_division=0
            )
            
            # Reference model classification report
            ref_report = classification_report(
                y_test_array, reference_results['predictions'],
                target_names=['No Alzheimer', 'Alzheimer'],
                output_dict=True,
                zero_division=0
            )
            
            comparison['classification_reports'] = {
                'ga_model': ga_report,
                'reference_model': ref_report
            }
            
            # Confusion matrices
            comparison['confusion_matrices'] = {
                'ga_model': confusion_matrix(y_test_array, ga_results['predictions']).tolist(),
                'reference_model': confusion_matrix(y_test_array, reference_results['predictions']).tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating classification reports: {e}")
            comparison['classification_reports'] = None
            comparison['confusion_matrices'] = None
        
        return comparison
    
    def analyze_generalizability(self, ga_results: Dict[str, Any], reference_results: Dict[str, Any], 
                               config: GAConfig) -> Dict[str, Any]:
        """Analyze generalizability of both models"""
        
        analysis = {}
        
        # Enhanced overfitting indicators - need to find best epochs first
        ga_train_losses = ga_results['training_metrics']['train_losses']
        ga_val_losses = ga_results['training_metrics']['val_losses']
        ga_train_accuracies = ga_results['training_metrics']['train_accuracies']
        ga_val_accuracies = ga_results['training_metrics']['val_accuracies']
        
        ref_train_losses = reference_results['training_metrics']['train_losses']
        ref_val_losses = reference_results['training_metrics']['val_losses']
        ref_train_accuracies = reference_results['training_metrics']['train_accuracies']
        ref_val_accuracies = reference_results['training_metrics']['val_accuracies']
        
        # Find the epoch with minimum validation loss (best model) - this is the saved model
        ga_best_val_epoch = ga_val_losses.index(min(ga_val_losses))
        ref_best_val_epoch = ref_val_losses.index(min(ref_val_losses))
        
        # Training vs Test performance gap
        # Use the training and validation accuracies from the epoch with best validation loss
        # This corresponds to the actual saved model that was used for testing
        ga_train_acc_at_best = ga_train_accuracies[ga_best_val_epoch]
        ga_val_acc_at_best = ga_val_accuracies[ga_best_val_epoch]
        ga_test_acc = ga_results['test_metrics']['accuracy']
        
        ref_train_acc_at_best = ref_train_accuracies[ref_best_val_epoch]
        ref_val_acc_at_best = ref_val_accuracies[ref_best_val_epoch]
        ref_test_acc = reference_results['test_metrics']['accuracy']
        
        # Calculate generalization gaps using the correct epoch data
        ga_generalization_gap = ga_val_acc_at_best - ga_test_acc  # Val-test gap for saved model
        ga_train_val_gap = ga_train_acc_at_best - ga_val_acc_at_best  # Train-val gap at best epoch
        
        ref_generalization_gap = ref_val_acc_at_best - ref_test_acc
        ref_train_val_gap = ref_train_acc_at_best - ref_val_acc_at_best
        
        # Also track final epoch metrics for comparison
        ga_final_train_acc = ga_train_accuracies[-1]
        ga_final_val_acc = ga_val_accuracies[-1]
        ref_final_train_acc = ref_train_accuracies[-1]
        ref_final_val_acc = ref_val_accuracies[-1]
        
        analysis['generalization_gaps'] = {
            'ga_model': {
                'train_accuracy_at_best': ga_train_acc_at_best,
                'val_accuracy_at_best': ga_val_acc_at_best,
                'test_accuracy': ga_test_acc,
                'val_test_gap': ga_generalization_gap,
                'train_val_gap_at_best': ga_train_val_gap,
                'overall_gap_at_best': ga_train_acc_at_best - ga_test_acc,  # Train-test gap for saved model
                'best_epoch': ga_best_val_epoch,
                # Final epoch metrics for reference
                'final_train_accuracy': ga_final_train_acc,
                'final_val_accuracy': ga_final_val_acc,
                'final_train_val_gap': ga_final_train_acc - ga_final_val_acc
            },
            'reference_model': {
                'train_accuracy_at_best': ref_train_acc_at_best,
                'val_accuracy_at_best': ref_val_acc_at_best,
                'test_accuracy': ref_test_acc,
                'val_test_gap': ref_generalization_gap,
                'train_val_gap_at_best': ref_train_val_gap,
                'overall_gap_at_best': ref_train_acc_at_best - ref_test_acc,
                'best_epoch': ref_best_val_epoch,
                # Final epoch metrics for reference
                'final_train_accuracy': ref_final_train_acc,
                'final_val_accuracy': ref_final_val_acc,
                'final_train_val_gap': ref_final_train_acc - ref_final_val_acc
            }
        }
        
        # Learning curve analysis
        analysis['learning_curves'] = {
            'ga_model': {
                'train_losses': ga_train_losses,
                'val_losses': ga_val_losses,
                'train_accuracies': ga_train_accuracies,
                'val_accuracies': ga_val_accuracies
            },
            'reference_model': {
                'train_losses': ref_train_losses,
                'val_losses': ref_val_losses,
                'train_accuracies': ref_train_accuracies,
                'val_accuracies': ref_val_accuracies
            }
        }
        
        analysis['overfitting_indicators'] = {
            'ga_model': {
                'final_train_val_loss_diff': ga_train_losses[-1] - ga_val_losses[-1],
                'final_train_val_acc_diff': ga_final_train_acc - ga_final_val_acc,
                'best_val_epoch': ga_best_val_epoch,
                'total_epochs': len(ga_train_losses),
                'early_stopped': len(ga_train_losses) < config.neural_net_epochs,
                'val_loss_at_best': ga_val_losses[ga_best_val_epoch],
                'train_loss_at_best': ga_train_losses[ga_best_val_epoch],
                'loss_divergence_at_best': ga_train_losses[ga_best_val_epoch] - ga_val_losses[ga_best_val_epoch],
                'acc_divergence_at_best': ga_train_acc_at_best - ga_val_acc_at_best,
                'max_val_loss_increase': max(ga_val_losses) - min(ga_val_losses),
                'val_loss_trend': 'increasing' if ga_val_losses[-1] > ga_val_losses[ga_best_val_epoch] else 'stable',
                'epochs_after_best': len(ga_train_losses) - 1 - ga_best_val_epoch
            },
            'reference_model': {
                'final_train_val_loss_diff': ref_train_losses[-1] - ref_val_losses[-1],
                'final_train_val_acc_diff': ref_final_train_acc - ref_final_val_acc,
                'best_val_epoch': ref_best_val_epoch,
                'total_epochs': len(ref_train_losses),
                'early_stopped': len(ref_train_losses) < config.neural_net_epochs,
                'val_loss_at_best': ref_val_losses[ref_best_val_epoch],
                'train_loss_at_best': ref_train_losses[ref_best_val_epoch],
                'loss_divergence_at_best': ref_train_losses[ref_best_val_epoch] - ref_val_losses[ref_best_val_epoch],
                'acc_divergence_at_best': ref_train_acc_at_best - ref_val_acc_at_best,
                'max_val_loss_increase': max(ref_val_losses) - min(ref_val_losses),
                'val_loss_trend': 'increasing' if ref_val_losses[-1] > ref_val_losses[ref_best_val_epoch] else 'stable',
                'epochs_after_best': len(ref_train_losses) - 1 - ref_best_val_epoch
            }
        }
        
        return analysis
    
    def retrain_with_full_dataset(self, ga_results: Dict[str, Any], selected_features: List[str], 
                                config: GAConfig, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Retrain the best GA model using the original dataset (all features).
        
        This method:
        1. Takes the trained GA model (trained on selected features)
        2. Creates a new model for the full feature set
        3. Transfers weights from GA model for selected features
        4. Randomly initializes weights for additional features
        5. Retrains on the full dataset
        
        Args:
            ga_results: Results from GA model training (contains trained model)
            selected_features: Features that were selected by GA
            config: GA configuration
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            
        Returns:
            Dictionary containing retrained model results and comparison
        """
        
        self.logger.info("Retraining GA model with full dataset (weight transfer approach)...")
        
        # Get the trained GA model
        ga_model = ga_results['model']
        ga_preprocessor = ga_results['preprocessor']
        
        # Create preprocessor for full dataset
        full_preprocessor = DataPreprocessor(batch_size=config.batch_size)
        full_preprocessor.fit_preprocessors(X_train, self.categorical_features, self.numerical_features)
        
        # Transform data with all features
        X_train_full_processed = full_preprocessor.transform_data(
            X_train, self.categorical_features, self.numerical_features
        )
        X_test_full_processed = full_preprocessor.transform_data(
            X_test, self.categorical_features, self.numerical_features
        )
        
        # Transform data with selected features (for weight mapping)
        X_train_selected_processed = ga_preprocessor.transform_data(
            X_train, self.categorical_features, self.numerical_features, 
            selected_features=selected_features
        )
        
        self.logger.info(f"Full dataset features: {X_train_full_processed.shape[1]}")
        self.logger.info(f"GA selected features: {X_train_selected_processed.shape[1]}")
        
        # Create new model for full dataset
        full_model = Net(
            input_size=X_train_full_processed.shape[1],
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size,
            activation=config.activation,
            dropout_rate=config.dropout_rate
        )
        
        # Transfer weights from GA model to full model
        self._transfer_weights(ga_model, full_model, X_train_selected_processed.columns, 
                             X_train_full_processed.columns)
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=config.random_seed,
            stratify=y_train
        )
        
        # Transform split data
        X_train_split_processed = full_preprocessor.transform_data(
            X_train_split, self.categorical_features, self.numerical_features
        )
        X_val_split_processed = full_preprocessor.transform_data(
            X_val_split, self.categorical_features, self.numerical_features
        )
        
        # Create data loaders
        train_loader = full_preprocessor.create_data_loader(X_train_split_processed, y_train_split, shuffle=True)
        val_loader = full_preprocessor.create_data_loader(X_val_split_processed, y_val_split, shuffle=True)
        test_loader = full_preprocessor.create_data_loader(X_test_full_processed, y_test, shuffle=True)
        
        # Create trainer
        trainer = Trainer(
            max_epochs=config.neural_net_epochs,
            patience=config.neural_net_patience,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Train the expanded model
        self.logger.info("Training expanded GA model with transferred weights...")
        trained_full_model, training_metrics = trainer.train_model(
            model=full_model,
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=1
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(trained_full_model, test_loader)
        
        # Get detailed predictions
        predictions, probabilities = self._get_detailed_predictions(trained_full_model, test_loader)
        
        # Train reference model for comparison (if not already done)
        reference_results = self.train_neural_network(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selected_features=None,
            model_name="Reference_Full"
        )
        
        # Package results
        ga_full_results = {
            'model': trained_full_model,
            'preprocessor': full_preprocessor,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'processed_features': list(X_train_full_processed.columns),
            'input_size': X_train_full_processed.shape[1],
            'weight_transfer_info': {
                'original_features': len(X_train_selected_processed.columns),
                'expanded_features': len(X_train_full_processed.columns),
                'transferred_weights': True
            }
        }
        
        # Compare expanded GA model with reference model
        full_comparison = self.compare_models(ga_full_results, reference_results, y_test)
        
        # Analyze generalizability of expanded model
        full_generalizability = self.analyze_generalizability(ga_full_results, reference_results, config)
        
        return {
            'ga_expanded_results': ga_full_results,
            'reference_full_results': reference_results,
            'full_comparison': full_comparison,
            'full_generalizability': full_generalizability,
            'weight_transfer_summary': {
                'selected_features': selected_features,
                'selected_feature_count': len(selected_features),
                'total_feature_count': len(self.X_data.columns),
                'processed_selected_features': len(X_train_selected_processed.columns),
                'processed_total_features': len(X_train_full_processed.columns),
                'weights_transferred': True
            }
        }
    
    def _transfer_weights(self, source_model: torch.nn.Module, target_model: torch.nn.Module,
                         source_features: List[str], target_features: List[str]):
        """
        Transfer weights from source model (GA) to target model (full dataset).
        
        Args:
            source_model: Trained GA model with selected features
            target_model: New model for full dataset
            source_features: Feature names from GA model
            target_features: Feature names for full model
        """
        
        self.logger.info("Transferring weights from GA model to expanded model...")
        
        # Convert to lists if they are pandas Index objects
        if hasattr(source_features, 'tolist'):
            source_features = source_features.tolist()
        if hasattr(target_features, 'tolist'):
            target_features = target_features.tolist()
        
        # Create mapping from source features to target features
        feature_mapping = {}
        for i, source_feat in enumerate(source_features):
            if source_feat in target_features:
                target_idx = target_features.index(source_feat)
                feature_mapping[i] = target_idx
        
        self.logger.info(f"Mapped {len(feature_mapping)} features from source to target model")
        
        # Transfer weights layer by layer
        source_layers = [module for module in source_model.modules() if isinstance(module, torch.nn.Linear)]
        target_layers = [module for module in target_model.modules() if isinstance(module, torch.nn.Linear)]
        
        if len(source_layers) != len(target_layers):
            raise ValueError(f"Model architectures don't match: {len(source_layers)} vs {len(target_layers)} layers")
        
        with torch.no_grad():
            for layer_idx, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers)):
                
                if layer_idx == 0:  # First layer (input layer)
                    # Transfer weights for mapped features
                    for source_idx, target_idx in feature_mapping.items():
                        target_layer.weight[:, target_idx] = source_layer.weight[:, source_idx]
                    
                    # Copy bias (same for all neurons)
                    target_layer.bias.copy_(source_layer.bias)
                    
                    self.logger.info(f"Layer {layer_idx}: Transferred weights for {len(feature_mapping)} input features")
                    
                else:  # Hidden and output layers
                    # For hidden/output layers, copy all weights since the hidden layer size should be the same
                    if source_layer.weight.shape == target_layer.weight.shape:
                        target_layer.weight.copy_(source_layer.weight)
                        target_layer.bias.copy_(source_layer.bias)
                        self.logger.info(f"Layer {layer_idx}: Copied all weights (shape: {source_layer.weight.shape})")
                    else:
                        self.logger.warning(f"Layer {layer_idx}: Shape mismatch, keeping random initialization")
        
        self.logger.info("Weight transfer completed")
    
    def generate_visualizations(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                              full_retrain_results: Dict[str, Any] = None):
        """Generate comprehensive visualizations"""
        
        self.logger.info("Generating visualizations...")
        
        # Use the new visualizer module
        self.visualizer.generate_all_plots(comparison, generalizability, full_retrain_results)
        
        self.logger.info(f"All plots saved to {self.plots_dir}")
    
    def generate_comprehensive_report(self, comparison: Dict[str, Any], 
                                    generalizability: Dict[str, Any],
                                    selected_features: List[str],
                                    full_retrain_results: Dict[str, Any] = None):
        """Generate comprehensive analysis report"""
        
        # Use the new reporter module
        self.reporter.generate_report(comparison, generalizability, selected_features, full_retrain_results)
    
    def save_results(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                    selected_features: List[str], full_retrain_results: Dict[str, Any] = None):
        """Save all results to JSON files"""
        
        # Clean the data by removing non-serializable objects (models, preprocessors)
        clean_comparison = self._clean_results_for_serialization(comparison)
        clean_generalizability = self._clean_results_for_serialization(generalizability)
        clean_full_retrain_results = None
        if full_retrain_results:
            clean_full_retrain_results = self._clean_results_for_serialization(full_retrain_results)
        
        # Use the new reporter module with cleaned data
        self.reporter.save_results(clean_comparison, clean_generalizability, selected_features, clean_full_retrain_results)
    
    def _clean_results_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-serializable objects from results data"""
        import copy
        
        # Deep copy to avoid modifying original data
        clean_data = copy.deepcopy(data)
        
        def clean_dict(d):
            if isinstance(d, dict):
                # Remove known non-serializable keys
                keys_to_remove = ['model', 'preprocessor']
                for key in keys_to_remove:
                    if key in d:
                        del d[key]
                
                # Recursively clean nested dictionaries
                for key, value in d.items():
                    if isinstance(value, dict):
                        clean_dict(value)
                    elif isinstance(value, list):
                        clean_list(value)
            return d
        
        def clean_list(lst):
            for item in lst:
                if isinstance(item, dict):
                    clean_dict(item)
                elif isinstance(item, list):
                    clean_list(item)
        
        return clean_dict(clean_data)
    
    def run_complete_analysis(self, results_file: str) -> Dict[str, Any]:
        """Run the complete neural network comparison analysis"""
        
        self.logger.info("="*80)
        self.logger.info("STARTING NEURAL NETWORK COMPARISON ANALYSIS")
        self.logger.info("="*80)
        
        try:
            # 1. Load best GA results
            self.logger.info("Step 1: Loading best GA experiment results...")
            best_experiment = self.load_best_ga_results(results_file)
            
            # 2. Extract selected features
            self.logger.info("Step 2: Extracting selected features from best individual...")
            selected_features = self.extract_best_individual_features(best_experiment)
            
            # 3. Create train/test splits
            self.logger.info("Step 3: Creating train/test splits...")
            X_train, X_test, y_train, y_test = self.create_train_test_splits()
            
            # 4. Get GA configuration from best experiment
            config_dict = best_experiment['experiment_info']['config']
            config = GAConfig(**config_dict)
            
            # 5. Train GA model with selected features
            self.logger.info("Step 4: Training GA model with selected features...")
            ga_results = self.train_neural_network(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=selected_features,
                model_name="GA_Model"
            )
            
            # 6. Train reference model with all features
            self.logger.info("Step 5: Training reference model with all features...")
            reference_results = self.train_neural_network(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=None,
                model_name="Reference_Model"
            )
            
            # 7. Compare models
            self.logger.info("Step 6: Comparing model performance...")
            comparison = self.compare_models(ga_results, reference_results, y_test)
            
            # 8. Analyze generalizability
            self.logger.info("Step 7: Analyzing generalizability...")
            generalizability = self.analyze_generalizability(ga_results, reference_results, config)
            
            # 9. Retrain with full dataset
            self.logger.info("Step 8: Retraining with full dataset...")
            full_retrain_results = self.retrain_with_full_dataset(ga_results, selected_features, config, X_train, y_train, X_test, y_test)
            
            # 10. Generate visualizations
            self.logger.info("Step 9: Generating visualizations...")
            self.generate_visualizations(comparison, generalizability, full_retrain_results)
            
            # 11. Generate comprehensive report
            self.logger.info("Step 10: Generating comprehensive report...")
            self.generate_comprehensive_report(comparison, generalizability, selected_features, full_retrain_results)
            
            # 12. Save results
            self.logger.info("Step 11: Saving results...")
            self.save_results(comparison, generalizability, selected_features, full_retrain_results)
            
            # 13. Print summary
            self._print_analysis_summary(comparison, generalizability, selected_features, full_retrain_results)
            
            self.logger.info("="*80)
            self.logger.info("NEURAL NETWORK COMPARISON ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return {
                'comparison': comparison,
                'generalizability': generalizability,
                'selected_features': selected_features,
                'full_retrain_results': full_retrain_results
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
    
    def _print_analysis_summary(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                              selected_features: List[str], full_retrain_results: Dict[str, Any]):
        """Print comprehensive analysis summary"""
        
        # Use the new reporter module
        self.reporter.print_analysis_summary(comparison, generalizability, selected_features, full_retrain_results)


def main(results_file: str):
    """Main function to run the neural network comparison analysis"""
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    from datetime import datetime
    log_file = f"logs/nn_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(__name__, log_file)
    
    try:
        # Create and run analysis
        analyzer = NeuralNetworkComparison("alzheimers_disease_data.csv")
        results = analyzer.run_complete_analysis(results_file)
        
        logger.info("Neural network comparison analysis completed successfully!")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Results saved to: {analyzer.results_dir}")
        logger.info(f"Plots saved to: {analyzer.plots_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Neural network comparison analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python neural_network_comparison.py <results_file>")
        print("Example: python neural_network_comparison.py /experiment_results/comprehensive_results_20250605_005118.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    main(results_file) 