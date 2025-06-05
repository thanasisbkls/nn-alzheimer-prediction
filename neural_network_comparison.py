#!/usr/bin/env python3
"""
Neural Network Comparison Module

Compare the performance of two models in two tasks:
1. GA-selected features model vs Reference model (all features)
2. GA model retrained with all features vs Reference model

Focus areas:
- Generalizability of the networks
- Effect of feature reduction on NN performance  
- Possibility of overfitting to test data
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import GAConfig
from src.data import AlzheimerDataLoader, DataPreprocessor
from src.models import Net, Trainer
from src.utils import setup_logger, set_seed


class SimpleNeuralNetworkComparison:
    """
    Simple comparison between GA-optimized and reference neural networks
    """
    
    def __init__(self, data_file: str = "alzheimers_disease_data.csv"):
        """Initialize the comparison module"""
        self.data_file = data_file
        self.logger = setup_logger(__name__)
        
        # Create output directory
        self.plots_dir = Path("simple_comparison_plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare the Alzheimer's dataset"""
        self.logger.info(f"Loading data from {self.data_file}")
        
        data_loader = AlzheimerDataLoader()
        self.X_data, self.y_data, self.categorical_features, self.numerical_features = \
            data_loader.load_data(self.data_file)
        
        self.logger.info(f"Data loaded: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} features")
        
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
        """Get detailed predictions and probabilities from model"""
        
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
    
    def retrain_ga_with_all_features(self, ga_results: Dict[str, Any], selected_features: List[str], 
                                   config: GAConfig, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Retrain GA model with all features using weight transfer"""
        
        self.logger.info("Retraining GA model with all features using weight transfer...")
        
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
        
        return {
            'model': trained_full_model,
            'preprocessor': full_preprocessor,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'processed_features': list(X_train_full_processed.columns),
            'input_size': X_train_full_processed.shape[1]
        }
    
    def _transfer_weights(self, source_model: torch.nn.Module, target_model: torch.nn.Module,
                         source_features: List[str], target_features: List[str]):
        """Transfer weights from source model (GA) to target model (full dataset)"""
        
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
    
    def compare_and_analyze(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any], 
                          model1_name: str, model2_name: str, y_test: pd.Series) -> Dict[str, Any]:
        """Compare two models and analyze generalizability, feature effects, and overfitting"""
        
        # Basic comparison
        comparison = {
            'model1': {
                'name': model1_name,
                'test_accuracy': model1_results['test_metrics']['accuracy'],
                'test_loss': model1_results['test_metrics']['loss'],
                'num_features': model1_results['input_size']
            },
            'model2': {
                'name': model2_name,
                'test_accuracy': model2_results['test_metrics']['accuracy'],
                'test_loss': model2_results['test_metrics']['loss'],
                'num_features': model2_results['input_size']
            }
        }
        
        # Performance differences
        comparison['differences'] = {
            'accuracy_diff': model1_results['test_metrics']['accuracy'] - model2_results['test_metrics']['accuracy'],
            'loss_diff': model1_results['test_metrics']['loss'] - model2_results['test_metrics']['loss'],
            'feature_reduction': 1 - (model1_results['input_size'] / model2_results['input_size']) if model2_results['input_size'] > 0 else 0
        }
        
        # Generalizability analysis
        generalizability = self._analyze_generalizability(model1_results, model2_results, model1_name, model2_name)
        
        # Overfitting analysis
        overfitting = self._analyze_overfitting(model1_results, model2_results, model1_name, model2_name)
        
        return {
            'comparison': comparison,
            'generalizability': generalizability,
            'overfitting': overfitting
        }
    
    def _analyze_generalizability(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any],
                                model1_name: str, model2_name: str) -> Dict[str, Any]:
        """Analyze generalizability by comparing train/val/test performance"""
        
        # Extract training curves
        m1_train_losses = model1_results['training_metrics']['train_losses']
        m1_val_losses = model1_results['training_metrics']['val_losses']
        m1_train_accs = model1_results['training_metrics']['train_accuracies']
        m1_val_accs = model1_results['training_metrics']['val_accuracies']
        
        m2_train_losses = model2_results['training_metrics']['train_losses']
        m2_val_losses = model2_results['training_metrics']['val_losses']
        m2_train_accs = model2_results['training_metrics']['train_accuracies']
        m2_val_accs = model2_results['training_metrics']['val_accuracies']
        
        # Find best epochs (minimum validation loss)
        m1_best_epoch = m1_val_losses.index(min(m1_val_losses))
        m2_best_epoch = m2_val_losses.index(min(m2_val_losses))
        
        # Calculate generalization gaps
        m1_train_val_gap = m1_train_accs[m1_best_epoch] - m1_val_accs[m1_best_epoch]
        m1_val_test_gap = m1_val_accs[m1_best_epoch] - model1_results['test_metrics']['accuracy']
        
        m2_train_val_gap = m2_train_accs[m2_best_epoch] - m2_val_accs[m2_best_epoch]
        m2_val_test_gap = m2_val_accs[m2_best_epoch] - model2_results['test_metrics']['accuracy']
        
        return {
            model1_name: {
                'train_val_gap': m1_train_val_gap,
                'val_test_gap': m1_val_test_gap,
                'total_gap': m1_train_accs[m1_best_epoch] - model1_results['test_metrics']['accuracy'],
                'best_epoch': m1_best_epoch,
                'training_curves': {
                    'train_losses': m1_train_losses,
                    'val_losses': m1_val_losses,
                    'train_accuracies': m1_train_accs,
                    'val_accuracies': m1_val_accs
                }
            },
            model2_name: {
                'train_val_gap': m2_train_val_gap,
                'val_test_gap': m2_val_test_gap,
                'total_gap': m2_train_accs[m2_best_epoch] - model2_results['test_metrics']['accuracy'],
                'best_epoch': m2_best_epoch,
                'training_curves': {
                    'train_losses': m2_train_losses,
                    'val_losses': m2_val_losses,
                    'train_accuracies': m2_train_accs,
                    'val_accuracies': m2_val_accs
                }
            }
        }
    
    def _analyze_overfitting(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any],
                           model1_name: str, model2_name: str) -> Dict[str, Any]:
        """Analyze overfitting indicators"""
        
        # Training curves
        m1_train_losses = model1_results['training_metrics']['train_losses']
        m1_val_losses = model1_results['training_metrics']['val_losses']
        m2_train_losses = model2_results['training_metrics']['train_losses']
        m2_val_losses = model2_results['training_metrics']['val_losses']
        
        # Find best epochs and analyze overfitting signs
        m1_best_epoch = m1_val_losses.index(min(m1_val_losses))
        m2_best_epoch = m2_val_losses.index(min(m2_val_losses))
        
        # Check if validation loss increased after best epoch (overfitting sign)
        m1_val_increase_after_best = max(m1_val_losses[m1_best_epoch:]) - m1_val_losses[m1_best_epoch] if m1_best_epoch < len(m1_val_losses) - 1 else 0
        m2_val_increase_after_best = max(m2_val_losses[m2_best_epoch:]) - m2_val_losses[m2_best_epoch] if m2_best_epoch < len(m2_val_losses) - 1 else 0
        
        return {
            model1_name: {
                'early_stopped': len(m1_train_losses) < 100,  # Assuming max epochs is 100
                'best_epoch': m1_best_epoch,
                'epochs_after_best': len(m1_train_losses) - 1 - m1_best_epoch,
                'val_loss_increase_after_best': m1_val_increase_after_best,
                'final_train_val_loss_diff': m1_train_losses[-1] - m1_val_losses[-1],
                'overfitting_score': m1_val_increase_after_best + max(0, m1_train_losses[-1] - m1_val_losses[-1])
            },
            model2_name: {
                'early_stopped': len(m2_train_losses) < 100,
                'best_epoch': m2_best_epoch,
                'epochs_after_best': len(m2_train_losses) - 1 - m2_best_epoch,
                'val_loss_increase_after_best': m2_val_increase_after_best,
                'final_train_val_loss_diff': m2_train_losses[-1] - m2_val_losses[-1],
                'overfitting_score': m2_val_increase_after_best + max(0, m2_train_losses[-1] - m2_val_losses[-1])
            }
        }
    
    def create_simple_visualizations(self, task1_analysis: Dict[str, Any], task2_analysis: Dict[str, Any],
                                   selected_features: List[str]):
        """Create separate focused visualizations for each type of analysis"""
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl", 8)
        
        self.logger.info("Creating focused visualizations...")
        
        # 1. Performance Comparison Plot
        self._create_performance_comparison_plot(task1_analysis, task2_analysis)
        
        # 2. Feature Analysis Plot
        self._create_feature_analysis_plot(task1_analysis, selected_features)
        
        # 3. Generalizability Analysis Plot
        self._create_generalizability_plot(task1_analysis, task2_analysis)
        
        # 4. Learning Curves Plot
        self._create_learning_curves_plot(task1_analysis, task2_analysis)
        
        self.logger.info(f"All focused plots saved to {self.plots_dir}")
    
    def _create_performance_comparison_plot(self, task1_analysis: Dict[str, Any], task2_analysis: Dict[str, Any]):
        """Create performance comparison plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Comparison: GA vs Reference Models', fontsize=16, fontweight='bold')
        
        # Task 1: Accuracy comparison
        models_t1 = ['GA Selected\nFeatures', 'Reference\nAll Features']
        accuracies_t1 = [task1_analysis['comparison']['model1']['test_accuracy'], 
                        task1_analysis['comparison']['model2']['test_accuracy']]
        bars1 = ax1.bar(models_t1, accuracies_t1, color=['lightblue', 'lightcoral'], alpha=0.8)
        ax1.set_title('Task 1: Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, max(accuracies_t1) * 1.15)  # Add space for text
        for bar, acc in zip(bars1, accuracies_t1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accuracies_t1) * 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Task 2: Accuracy comparison
        models_t2 = ['GA Retrained\nAll Features', 'Reference\nAll Features']
        accuracies_t2 = [task2_analysis['comparison']['model1']['test_accuracy'], 
                        task2_analysis['comparison']['model2']['test_accuracy']]
        bars2 = ax2.bar(models_t2, accuracies_t2, color=['lightgreen', 'lightcoral'], alpha=0.8)
        ax2.set_title('Task 2: Test Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, max(accuracies_t2) * 1.15)  # Add space for text
        for bar, acc in zip(bars2, accuracies_t2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accuracies_t2) * 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Task 1: Loss comparison
        losses_t1 = [task1_analysis['comparison']['model1']['test_loss'], 
                     task1_analysis['comparison']['model2']['test_loss']]
        bars3 = ax3.bar(models_t1, losses_t1, color=['lightblue', 'lightcoral'], alpha=0.8)
        ax3.set_title('Task 1: Test Loss Comparison')
        ax3.set_ylabel('Loss')
        ax3.set_ylim(0, max(losses_t1) * 1.15)  # Add space for text
        for bar, loss in zip(bars3, losses_t1):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(losses_t1) * 0.02, 
                    f'{loss:.3f}', ha='center', va='bottom')
        
        # Task 2: Loss comparison
        losses_t2 = [task2_analysis['comparison']['model1']['test_loss'], 
                     task2_analysis['comparison']['model2']['test_loss']]
        bars4 = ax4.bar(models_t2, losses_t2, color=['lightgreen', 'lightcoral'], alpha=0.8)
        ax4.set_title('Task 2: Test Loss Comparison')
        ax4.set_ylabel('Loss')
        ax4.set_ylim(0, max(losses_t2) * 1.15)  # Add space for text
        for bar, loss in zip(bars4, losses_t2):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(losses_t2) * 0.02, 
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '01_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_analysis_plot(self, task1_analysis: Dict[str, Any], selected_features: List[str]):
        """Create feature analysis plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Feature Selection Analysis', fontsize=16, fontweight='bold')
        
        # Feature count comparison
        models = ['GA Selected\nFeatures', 'Reference\nAll Features']
        feature_counts = [task1_analysis['comparison']['model1']['num_features'],
                         task1_analysis['comparison']['model2']['num_features']]
        bars1 = ax1.bar(models, feature_counts, color=['lightgreen', 'lightsalmon'], alpha=0.8)
        ax1.set_title('Feature Count Comparison')
        ax1.set_ylabel('Number of Features')
        for bar, count in zip(bars1, feature_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count}', ha='center', va='bottom')
        
        # Feature reduction pie chart
        total_features = len(self.X_data.columns)
        selected_features_count = len(selected_features)
        removed_features_count = total_features - selected_features_count
        
        sizes = [selected_features_count, removed_features_count]
        labels = [f'Selected\n({selected_features_count})', f'Removed\n({removed_features_count})']
        colors = ['lightgreen', 'lightcoral']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Selection Distribution')
        
        # Feature reduction vs accuracy scatter
        feature_reduction_pct = (1 - selected_features_count/total_features) * 100
        accuracy_with_reduction = task1_analysis['comparison']['model1']['test_accuracy']
        accuracy_without_reduction = task1_analysis['comparison']['model2']['test_accuracy']
        
        ax3.scatter([feature_reduction_pct], [accuracy_with_reduction], 
                   s=150, color='blue', label='GA Selected Features', alpha=0.8)
        ax3.scatter([0], [accuracy_without_reduction], 
                   s=150, color='red', label='All Features', alpha=0.8)
        ax3.set_title('Feature Reduction vs Accuracy')
        ax3.set_xlabel('Feature Reduction (%)')
        ax3.set_ylabel('Test Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Accuracy improvement bar
        accuracy_diff = task1_analysis['comparison']['differences']['accuracy_diff']
        color = 'green' if accuracy_diff > 0 else 'red'
        ax4.bar(['Accuracy\nImprovement'], [accuracy_diff], color=color, alpha=0.7)
        ax4.set_title('Feature Selection Impact')
        ax4.set_ylabel('Accuracy Difference')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.text(0, accuracy_diff + 0.005 if accuracy_diff > 0 else accuracy_diff - 0.015, 
                f'{accuracy_diff:+.3f}', ha='center', va='bottom' if accuracy_diff > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '02_feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_generalizability_plot(self, task1_analysis: Dict[str, Any], task2_analysis: Dict[str, Any]):
        """Create generalizability analysis plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Generalizability Analysis', fontsize=16, fontweight='bold')
        
        # Task 1: Generalization gaps
        model1_name_t1 = task1_analysis['comparison']['model1']['name']
        model2_name_t1 = task1_analysis['comparison']['model2']['name']
        gap_types = ['Train-Val Gap', 'Val-Test Gap']
        
        m1_gaps_t1 = [task1_analysis['generalizability'][model1_name_t1]['train_val_gap'],
                      task1_analysis['generalizability'][model1_name_t1]['val_test_gap']]
        m2_gaps_t1 = [task1_analysis['generalizability'][model2_name_t1]['train_val_gap'],
                      task1_analysis['generalizability'][model2_name_t1]['val_test_gap']]
        
        x = np.arange(len(gap_types))
        width = 0.35
        ax1.bar(x - width/2, m1_gaps_t1, width, label=model1_name_t1, alpha=0.8)
        ax1.bar(x + width/2, m2_gaps_t1, width, label=model2_name_t1, alpha=0.8)
        ax1.set_title('Task 1: Generalization Gaps')
        ax1.set_ylabel('Accuracy Gap')
        ax1.set_xticks(x)
        ax1.set_xticklabels(gap_types)
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Task 2: Generalization gaps
        model1_name_t2 = task2_analysis['comparison']['model1']['name']
        model2_name_t2 = task2_analysis['comparison']['model2']['name']
        
        m1_gaps_t2 = [task2_analysis['generalizability'][model1_name_t2]['train_val_gap'],
                      task2_analysis['generalizability'][model1_name_t2]['val_test_gap']]
        m2_gaps_t2 = [task2_analysis['generalizability'][model2_name_t2]['train_val_gap'],
                      task2_analysis['generalizability'][model2_name_t2]['val_test_gap']]
        
        ax2.bar(x - width/2, m1_gaps_t2, width, label=model1_name_t2, alpha=0.8)
        ax2.bar(x + width/2, m2_gaps_t2, width, label=model2_name_t2, alpha=0.8)
        ax2.set_title('Task 2: Generalization Gaps')
        ax2.set_ylabel('Accuracy Gap')
        ax2.set_xticks(x)
        ax2.set_xticklabels(gap_types)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Combined train-val gaps comparison
        models_combined = ['Task 1\nGA Selected', 'Task 1\nReference', 'Task 2\nGA Retrained', 'Task 2\nReference']
        train_val_gaps = [
            task1_analysis['generalizability'][model1_name_t1]['train_val_gap'],
            task1_analysis['generalizability'][model2_name_t1]['train_val_gap'],
            task2_analysis['generalizability'][model1_name_t2]['train_val_gap'],
            task2_analysis['generalizability'][model2_name_t2]['train_val_gap']
        ]
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightcoral']
        bars = ax3.bar(models_combined, train_val_gaps, color=colors, alpha=0.8)
        ax3.set_title('Train-Validation Gaps Comparison')
        ax3.set_ylabel('Train-Val Gap')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Combined val-test gaps comparison
        val_test_gaps = [
            task1_analysis['generalizability'][model1_name_t1]['val_test_gap'],
            task1_analysis['generalizability'][model2_name_t1]['val_test_gap'],
            task2_analysis['generalizability'][model1_name_t2]['val_test_gap'],
            task2_analysis['generalizability'][model2_name_t2]['val_test_gap']
        ]
        bars = ax4.bar(models_combined, val_test_gaps, color=colors, alpha=0.8)
        ax4.set_title('Validation-Test Gaps Comparison')
        ax4.set_ylabel('Val-Test Gap')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '03_generalizability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_learning_curves_plot(self, task1_analysis: Dict[str, Any], task2_analysis: Dict[str, Any]):
        """Create learning curves plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
        
        # Task 1: GA Selected vs Reference
        model1_name_t1 = task1_analysis['comparison']['model1']['name']
        model2_name_t1 = task1_analysis['comparison']['model2']['name']
        
        m1_curves_t1 = task1_analysis['generalizability'][model1_name_t1]['training_curves']
        epochs1_t1 = range(len(m1_curves_t1['train_accuracies']))
        
        ax1.plot(epochs1_t1, m1_curves_t1['train_accuracies'], 'b-', label='Train', alpha=0.8, linewidth=2)
        ax1.plot(epochs1_t1, m1_curves_t1['val_accuracies'], 'b--', label='Validation', alpha=0.8, linewidth=2)
        ax1.set_title(f'Task 1: {model1_name_t1} Learning Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        m2_curves_t1 = task1_analysis['generalizability'][model2_name_t1]['training_curves']
        epochs2_t1 = range(len(m2_curves_t1['train_accuracies']))
        
        ax2.plot(epochs2_t1, m2_curves_t1['train_accuracies'], 'r-', label='Train', alpha=0.8, linewidth=2)
        ax2.plot(epochs2_t1, m2_curves_t1['val_accuracies'], 'r--', label='Validation', alpha=0.8, linewidth=2)
        ax2.set_title(f'Task 1: {model2_name_t1} Learning Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Task 2: GA Retrained vs Reference
        model1_name_t2 = task2_analysis['comparison']['model1']['name']
        model2_name_t2 = task2_analysis['comparison']['model2']['name']
        
        m1_curves_t2 = task2_analysis['generalizability'][model1_name_t2]['training_curves']
        epochs1_t2 = range(len(m1_curves_t2['train_accuracies']))
        
        ax3.plot(epochs1_t2, m1_curves_t2['train_accuracies'], 'g-', label='Train', alpha=0.8, linewidth=2)
        ax3.plot(epochs1_t2, m1_curves_t2['val_accuracies'], 'g--', label='Validation', alpha=0.8, linewidth=2)
        ax3.set_title(f'Task 2: {model1_name_t2} Learning Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        m2_curves_t2 = task2_analysis['generalizability'][model2_name_t2]['training_curves']
        epochs2_t2 = range(len(m2_curves_t2['train_accuracies']))
        
        ax4.plot(epochs2_t2, m2_curves_t2['train_accuracies'], 'r-', label='Train', alpha=0.8, linewidth=2)
        ax4.plot(epochs2_t2, m2_curves_t2['val_accuracies'], 'r--', label='Validation', alpha=0.8, linewidth=2)
        ax4.set_title(f'Task 2: {model2_name_t2} Learning Curves')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '04_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comparison_analysis(self, results_file: str) -> Dict[str, Any]:
        """Run the complete two-task comparison analysis"""
        
        self.logger.info("="*80)
        self.logger.info("STARTING SIMPLE NEURAL NETWORK COMPARISON ANALYSIS")
        self.logger.info("="*80)
        
        try:
            # 1. Load best GA results and extract features
            self.logger.info("Step 1: Loading GA results and extracting features...")
            best_experiment = self.load_best_ga_results(results_file)
            selected_features = self.extract_best_individual_features(best_experiment)
            
            # 2. Create train/test splits
            self.logger.info("Step 2: Creating train/test splits...")
            X_train, X_test, y_train, y_test = self.create_train_test_splits()
            
            # 3. Get GA configuration
            config_dict = best_experiment['experiment_info']['config']
            config = GAConfig(**config_dict)
            
            # TASK 1: GA Selected Features vs Reference
            self.logger.info("="*50)
            self.logger.info("TASK 1: GA Selected Features vs Reference Model")
            self.logger.info("="*50)
            
            # Train GA model with selected features
            self.logger.info("Training GA model with selected features...")
            ga_selected_results = self.train_neural_network(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=selected_features,
                model_name="GA_Selected"
            )
            
            # Train reference model with all features
            self.logger.info("Training reference model with all features...")
            reference_results = self.train_neural_network(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=None,
                model_name="Reference"
            )
            
            # Compare and analyze Task 1
            task1_analysis = self.compare_and_analyze(
                ga_selected_results, reference_results,
                "GA_Selected", "Reference", y_test
            )
            
            # TASK 2: GA Retrained (All Features) vs Reference
            self.logger.info("="*50)
            self.logger.info("TASK 2: GA Retrained with All Features vs Reference Model")
            self.logger.info("="*50)
            
            # Retrain GA architecture with all features
            self.logger.info("Retraining GA architecture with all features...")
            ga_all_features_results = self.retrain_ga_with_all_features(
                ga_selected_results, selected_features, config, X_train, y_train, X_test, y_test
            )
            
            # Compare and analyze Task 2
            task2_analysis = self.compare_and_analyze(
                ga_all_features_results, reference_results,
                "GA_All_Features", "Reference", y_test
            )
            
            # 4. Generate simple visualizations
            self.logger.info("Step 3: Generating visualizations...")
            self.create_simple_visualizations(task1_analysis, task2_analysis, selected_features)
            
            self.logger.info("="*80)
            self.logger.info("SIMPLE NEURAL NETWORK COMPARISON ANALYSIS COMPLETED")
            self.logger.info("="*80)
            
            return {
                'task1_analysis': task1_analysis,
                'task2_analysis': task2_analysis,
                'selected_features': selected_features
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise


def main(results_file: str):
    """Main function to run the simple neural network comparison analysis"""
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    from datetime import datetime
    log_file = f"logs/simple_nn_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(__name__, log_file)
    
    try:
        # Create and run analysis
        analyzer = SimpleNeuralNetworkComparison("alzheimers_disease_data.csv")
        results = analyzer.run_comparison_analysis(results_file)
        
        logger.info("Simple neural network comparison analysis completed successfully!")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Plots saved to: {analyzer.plots_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simple neural network comparison analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python neural_network_comparison.py <results_file>")
        print("Example: python neural_network_comparison.py experiment_results/comprehensive_results_20250605_005118.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    main(results_file) 