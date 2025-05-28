#!/usr/bin/env python3
"""
Neural Network Visualizer Module

Visualization functions for Neural Network comparison results.
Extracts all plotting functionality from the neural network comparison module.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from .base_visualizer import BaseVisualizer


class NNVisualizer(BaseVisualizer):
    """
    Visualizer for Neural Network comparison results
    """
    
    def __init__(self, plots_dir: Path):
        """
        Initialize NN visualizer
        
        Args:
            plots_dir: Directory to save plots
        """
        super().__init__(plots_dir)
    
    def generate_all_plots(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                          full_retrain_results: Optional[Dict[str, Any]] = None):
        """
        Generate all neural network visualizations
        
        Args:
            comparison: Model comparison results
            generalizability: Generalizability analysis results
            full_retrain_results: Optional full retrain results
        """
        # Generate all plots
        self.plot_performance_comparison(comparison)
        self.plot_learning_curves(generalizability)
        self.plot_feature_analysis(comparison)
        self.plot_confusion_matrices(comparison)
        
        # Generate full retrain comparison if available
        if full_retrain_results:
            self.plot_full_retrain_comparison(comparison, full_retrain_results)
    
    def plot_performance_comparison(self, comparison: Dict[str, Any]):
        """Plot performance comparison between models"""
        
        fig, axes = self.create_subplot_grid(2, 2, figsize=(15, 12),
                                           suptitle='Neural Network Performance Comparison: GA vs Reference')
        
        # Accuracy comparison
        models = ['GA Model', 'Reference Model']
        accuracies = [comparison['ga_model']['test_accuracy'], comparison['reference_model']['test_accuracy']]
        
        bars = axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels
        self.add_value_labels_to_bars(axes[0, 0], bars)
        
        # Loss comparison
        losses = [comparison['ga_model']['test_loss'], comparison['reference_model']['test_loss']]
        bars = axes[0, 1].bar(models, losses, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('Test Loss Comparison')
        axes[0, 1].set_ylabel('Loss')
        
        self.add_value_labels_to_bars(axes[0, 1], bars)
        
        # Feature count comparison
        feature_counts = [comparison['ga_model']['num_features'], comparison['reference_model']['num_features']]
        bars = axes[1, 0].bar(models, feature_counts, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Number of Features Used')
        axes[1, 0].set_ylabel('Feature Count')
        
        self.add_value_labels_to_bars(axes[1, 0], bars, format_str='{:.0f}', offset=1)
        
        # Performance difference
        metrics = ['Accuracy', 'Loss', 'MSE', 'Feature Reduction']
        differences = [
            comparison['performance_difference']['accuracy_diff'],
            -comparison['performance_difference']['loss_diff'],  # Negative because lower loss is better
            -comparison['performance_difference']['mse_diff'],   # Negative because lower MSE is better
            comparison['performance_difference']['feature_reduction']
        ]
        
        colors = ['green' if d > 0 else 'red' for d in differences]
        axes[1, 1].bar(metrics, differences, color=colors, alpha=0.7)
        axes[1, 1].set_title('GA Model Improvement over Reference')
        axes[1, 1].set_ylabel('Improvement (positive = better)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('performance_comparison.png')
    
    def plot_learning_curves(self, generalizability: Dict[str, Any]):
        """Plot learning curves for both models"""
        
        fig, axes = self.create_subplot_grid(2, 2, figsize=(15, 10),
                                           suptitle='Learning Curves: Training vs Validation')
        
        # GA model learning curves
        ga_curves = generalizability['learning_curves']['ga_model']
        epochs_ga = range(len(ga_curves['train_losses']))
        
        axes[0, 0].plot(epochs_ga, ga_curves['train_losses'], label='Training Loss', color='blue')
        axes[0, 0].plot(epochs_ga, ga_curves['val_losses'], label='Validation Loss', color='red')
        axes[0, 0].set_title('GA Model - Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        self.setup_grid(axes[0, 0])
        
        axes[0, 1].plot(epochs_ga, ga_curves['train_accuracies'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(epochs_ga, ga_curves['val_accuracies'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('GA Model - Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        self.setup_grid(axes[0, 1])
        
        # Reference model learning curves
        ref_curves = generalizability['learning_curves']['reference_model']
        epochs_ref = range(len(ref_curves['train_losses']))
        
        axes[1, 0].plot(epochs_ref, ref_curves['train_losses'], label='Training Loss', color='blue')
        axes[1, 0].plot(epochs_ref, ref_curves['val_losses'], label='Validation Loss', color='red')
        axes[1, 0].set_title('Reference Model - Loss Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        self.setup_grid(axes[1, 0])
        
        axes[1, 1].plot(epochs_ref, ref_curves['train_accuracies'], label='Training Accuracy', color='blue')
        axes[1, 1].plot(epochs_ref, ref_curves['val_accuracies'], label='Validation Accuracy', color='red')
        axes[1, 1].set_title('Reference Model - Accuracy Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        self.setup_grid(axes[1, 1])
        
        plt.tight_layout()
        self.save_plot('learning_curves.png')
    
    def plot_feature_analysis(self, comparison: Dict[str, Any]):
        """Plot feature analysis"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature count comparison
        models = ['GA Model\n(Selected Features)', 'Reference Model\n(All Features)']
        feature_counts = [comparison['ga_model']['num_features'], comparison['reference_model']['num_features']]
        
        bars = ax1.bar(models, feature_counts, color=['lightgreen', 'lightblue'], alpha=0.8)
        ax1.set_title('Feature Usage Comparison')
        ax1.set_ylabel('Number of Features')
        
        # Add value labels
        for bar, count in zip(bars, feature_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count}', ha='center', va='bottom')
        
        # Feature reduction visualization
        total_features = comparison['reference_model']['num_features']
        selected_features = comparison['ga_model']['num_features']
        reduction_pct = (1 - selected_features/total_features) * 100
        
        sizes = [selected_features, total_features - selected_features]
        labels = [f'Selected\n({selected_features})', f'Removed\n({total_features - selected_features})']
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Feature Selection Effect\n({reduction_pct:.1f}% reduction)')
        
        plt.tight_layout()
        self.save_plot('feature_analysis.png')
    
    def plot_confusion_matrices(self, comparison: Dict[str, Any]):
        """Plot confusion matrices for both models"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # GA model confusion matrix
        ga_cm = np.array(comparison['confusion_matrices']['ga_model'])
        sns.heatmap(ga_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Alzheimer', 'Alzheimer'],
                   yticklabels=['No Alzheimer', 'Alzheimer'], ax=ax1)
        ax1.set_title('GA Model Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Reference model confusion matrix
        ref_cm = np.array(comparison['confusion_matrices']['reference_model'])
        sns.heatmap(ref_cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=['No Alzheimer', 'Alzheimer'],
                   yticklabels=['No Alzheimer', 'Alzheimer'], ax=ax2)
        ax2.set_title('Reference Model Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        self.save_plot('confusion_matrices.png')
    
    def plot_full_retrain_comparison(self, original_comparison: Dict[str, Any], 
                                    full_retrain_results: Dict[str, Any]):
        """Plot comparison between original and weight-transferred retrain results"""
        
        fig, axes = self.create_subplot_grid(2, 2, figsize=(15, 10),
                                           suptitle='Original vs Weight-Transferred Full Dataset Training')
        
        # Accuracy comparison
        scenarios = ['Original\n(Selected Features)', 'Weight Transfer\n(All Features)']
        
        ga_accuracies = [
            original_comparison['ga_model']['test_accuracy'],
            full_retrain_results['full_comparison']['ga_model']['test_accuracy']
        ]
        ref_accuracies = [
            original_comparison['reference_model']['test_accuracy'],
            full_retrain_results['full_comparison']['reference_model']['test_accuracy']
        ]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ga_accuracies, width, label='GA Model', color='skyblue')
        axes[0, 0].bar(x + width/2, ref_accuracies, width, label='Reference Model', color='lightcoral')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios)
        axes[0, 0].legend()
        
        # Loss comparison
        ga_losses = [
            original_comparison['ga_model']['test_loss'],
            full_retrain_results['full_comparison']['ga_model']['test_loss']
        ]
        ref_losses = [
            original_comparison['reference_model']['test_loss'],
            full_retrain_results['full_comparison']['reference_model']['test_loss']
        ]
        
        axes[0, 1].bar(x - width/2, ga_losses, width, label='GA Model', color='skyblue')
        axes[0, 1].bar(x + width/2, ref_losses, width, label='Reference Model', color='lightcoral')
        axes[0, 1].set_title('Test Loss Comparison')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios)
        axes[0, 1].legend()
        
        # Performance improvement
        original_improvement = original_comparison['performance_difference']['accuracy_diff']
        full_improvement = full_retrain_results['full_comparison']['performance_difference']['accuracy_diff']
        
        improvements = [original_improvement, full_improvement]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        axes[1, 0].bar(scenarios, improvements, color=colors, alpha=0.7)
        axes[1, 0].set_title('GA Model Accuracy Improvement over Reference')
        axes[1, 0].set_ylabel('Accuracy Difference')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Feature count comparison
        feature_counts = [
            original_comparison['ga_model']['num_features'],
            full_retrain_results['weight_transfer_summary']['processed_total_features']
        ]
        
        bars = axes[1, 1].bar(scenarios, feature_counts, color='lightgreen', alpha=0.8)
        axes[1, 1].set_title('Number of Features Used')
        axes[1, 1].set_ylabel('Feature Count')
        
        self.add_value_labels_to_bars(axes[1, 1], bars, format_str='{:.0f}', offset=1)
        
        plt.tight_layout()
        self.save_plot('full_retrain_comparison.png') 