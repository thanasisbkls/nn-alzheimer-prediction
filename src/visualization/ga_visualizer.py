#!/usr/bin/env python3
"""
GA Visualizer Module

Visualization functions for Genetic Algorithm experiment results.
Extracts all plotting functionality from the GA experiments runner.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .base_visualizer import BaseVisualizer


class GAVisualizer(BaseVisualizer):
    """
    Visualizer for Genetic Algorithm experiment results
    """
    
    def __init__(self, plots_dir: Path, X_data: pd.DataFrame):
        """
        Initialize GA visualizer
        
        Args:
            plots_dir: Directory to save plots
            X_data: Original dataset for feature analysis
        """
        super().__init__(plots_dir)
        self.X_data = X_data
    
    def generate_all_plots(self, all_results: Dict[str, Any], summary_statistics: Dict[str, Any]):
        """
        Generate all GA visualizations
        
        Args:
            all_results: Complete experiment results
            summary_statistics: Summary statistics across experiments
        """
        # Generate all plots
        self.plot_evolution_curves(all_results)
        self.plot_parameter_effects(summary_statistics)
        self.plot_convergence_analysis(summary_statistics)
        self.plot_feature_analysis(summary_statistics)
    
    def plot_evolution_curves(self, all_results: Dict[str, Any]):
        """Plot evolution curves for each experiment configuration"""
        
        # Create subplots for all experiments
        fig, axes = self.create_subplot_grid(2, 5, figsize=(20, 10), 
                                           suptitle='Evolution Curves: Mean Best Fitness vs Generation')
        axes = axes.flatten()
        
        for i, (exp_name, exp_result) in enumerate(all_results.items()):
            if 'mean_fitness_history' in exp_result and exp_result['mean_fitness_history']:
                ax = axes[i]
                
                fitness_history = exp_result['mean_fitness_history']
                generations = list(range(len(fitness_history)))
                
                ax.plot(generations, fitness_history, linewidth=2, label='Mean Best Fitness')
                ax.set_title(f"Exp {exp_result['experiment_info']['id']}: "
                           f"Pop={exp_result['experiment_info']['parameters']['population_size']}, "
                           f"Cx={exp_result['experiment_info']['parameters']['crossover_rate']}, "
                           f"Mut={exp_result['experiment_info']['parameters']['mutation_rate']:.2f}")
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness')
                self.setup_grid(ax)
                ax.legend()
        
        plt.tight_layout()
        self.save_plot('evolution_curves_all.png')
        
        # Individual plots for each experiment
        for exp_name, exp_result in all_results.items():
            if 'mean_fitness_history' in exp_result and exp_result['mean_fitness_history']:
                plt.figure(figsize=(10, 6))
                
                fitness_history = exp_result['mean_fitness_history']
                generations = list(range(len(fitness_history)))
                
                plt.plot(generations, fitness_history, linewidth=2, color='blue', label='Mean Best Fitness')
                plt.title(f"Evolution Curve - {exp_name}")
                plt.xlabel('Generation')
                plt.ylabel('Fitness (Lower is Better)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                self.save_plot(f'evolution_curve_{exp_name}.png')
    
    def plot_parameter_effects(self, summary_statistics: Dict[str, Any]):
        """Plot the effects of different parameters"""
        
        if not summary_statistics:
            return
        
        # Create DataFrame for analysis
        data = []
        for exp_name, stats in summary_statistics.items():
            data.append({
                'experiment': exp_name,
                'population_size': stats['parameters']['population_size'],
                'crossover_rate': stats['parameters']['crossover_rate'],
                'mutation_rate': stats['parameters']['mutation_rate'],
                'mean_fitness': stats['mean_best_fitness'],
                'mean_features': stats['mean_num_features'],
                'mean_generations': stats['mean_generations'],
                'convergence_rate': stats['convergence_rate']
            })
        
        df = pd.DataFrame(data)
        
        # Create parameter effects plot
        fig, axes = self.create_subplot_grid(2, 2, figsize=(15, 12), 
                                           suptitle='Parameter Effects Analysis')
        
        # Population size vs fitness
        pop_groups = df.groupby('population_size')['mean_fitness'].agg(['mean', 'std']).reset_index()
        bars = axes[0, 0].bar(pop_groups['population_size'].astype(str), pop_groups['mean'], 
                      yerr=pop_groups['std'], capsize=5)
        axes[0, 0].set_title('Population Size Effect on Fitness')
        axes[0, 0].set_xlabel('Population Size')
        axes[0, 0].set_ylabel('Mean Best Fitness')
        
        # Crossover rate vs fitness
        cx_groups = df.groupby('crossover_rate')['mean_fitness'].agg(['mean', 'std']).reset_index()
        bars = axes[0, 1].bar(cx_groups['crossover_rate'].astype(str), cx_groups['mean'], 
                      yerr=cx_groups['std'], capsize=5)
        axes[0, 1].set_title('Crossover Rate Effect on Fitness')
        axes[0, 1].set_xlabel('Crossover Rate')
        axes[0, 1].set_ylabel('Mean Best Fitness')
        
        # Mutation rate vs fitness
        mut_groups = df.groupby('mutation_rate')['mean_fitness'].agg(['mean', 'std']).reset_index()
        bars = axes[1, 0].bar(mut_groups['mutation_rate'].astype(str), mut_groups['mean'], 
                      yerr=mut_groups['std'], capsize=5)
        axes[1, 0].set_title('Mutation Rate Effect on Fitness')
        axes[1, 0].set_xlabel('Mutation Rate')
        axes[1, 0].set_ylabel('Mean Best Fitness')
        
        # Convergence rate comparison
        bars = axes[1, 1].bar(range(len(df)), df['convergence_rate'], color='green', alpha=0.7)
        axes[1, 1].set_title('Convergence Rate by Experiment')
        axes[1, 1].set_xlabel('Experiment ID')
        axes[1, 1].set_ylabel('Convergence Rate')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels([f"Exp{i+1}" for i in range(len(df))], rotation=45)
        
        plt.tight_layout()
        self.save_plot('parameter_effects.png')
    
    def plot_convergence_analysis(self, summary_statistics: Dict[str, Any]):
        """Plot convergence analysis"""
        
        if not summary_statistics:
            return
        
        # Create convergence comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        exp_names = list(summary_statistics.keys())
        convergence_rates = [stats['convergence_rate'] for stats in summary_statistics.values()]
        mean_generations = [stats['mean_generations'] for stats in summary_statistics.values()]
        
        # Convergence rate by experiment
        bars1 = ax1.bar(range(len(exp_names)), convergence_rates, color='skyblue', alpha=0.8)
        ax1.set_title('Convergence Rate by Experiment')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Convergence Rate')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        self.add_value_labels_to_bars(ax1, bars1, format_str='{:.2f}')
        
        # Mean generations to convergence
        bars2 = ax2.bar(range(len(exp_names)), mean_generations, color='lightcoral', alpha=0.8)
        ax2.set_title('Mean Generations to Convergence')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Mean Generations')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        
        # Add value labels on bars
        self.add_value_labels_to_bars(ax2, bars2, format_str='{:.0f}', offset=5)
        
        plt.tight_layout()
        self.save_plot('convergence_analysis.png')
    
    def plot_feature_analysis(self, summary_statistics: Dict[str, Any]):
        """Plot feature selection analysis"""
        
        if not summary_statistics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        exp_names = list(summary_statistics.keys())
        mean_features = [stats['mean_num_features'] for stats in summary_statistics.values()]
        std_features = [stats['std_num_features'] for stats in summary_statistics.values()]
        
        # Mean number of selected features
        bars = ax1.bar(range(len(exp_names)), mean_features, yerr=std_features, 
                      capsize=5, color='lightgreen', alpha=0.8)
        ax1.set_title('Mean Number of Selected Features')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Number of Features')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        
        # Feature reduction percentage
        total_features = len(self.X_data.columns)
        reduction_pct = [(total_features - feat) / total_features * 100 for feat in mean_features]
        
        bars2 = ax2.bar(range(len(exp_names)), reduction_pct, color='orange', alpha=0.8)
        ax2.set_title('Feature Reduction Percentage')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Reduction (%)')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        
        # Add value labels
        self.add_value_labels_to_bars(ax2, bars2, format_str='{:.1f}%', offset=1)
        
        plt.tight_layout()
        self.save_plot('feature_analysis.png') 