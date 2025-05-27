#!/usr/bin/env python3
"""
Comprehensive Genetic Algorithm Experiments Runner

Runs experiments for all configurations with statistical analysis and visualization.
Implements requirements:
- 10 runs per configuration for statistical significance
- Mean performance calculation across runs
- Evolution curve plotting for each configuration
- Comprehensive analysis of parameter effects on convergence
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import GAConfig
from src.data import AlzheimerDataLoader
from src.genetic import (
    TournamentSelection, 
    UniformCrossover, BitFlipMutation, FitnessEvaluator
)
from src.algorithms import GeneticAlgorithm
from src.utils import setup_logger, set_seed


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ComprehensiveExperimentRunner:
    """
    Comprehensive experiment runner with statistical analysis and visualization
    """
    
    def __init__(self, data_file: str = "alzheimers_disease_data.csv"):
        """Initialize experiment runner"""
        self.data_file = data_file
        self.logger = setup_logger(__name__)
        
        # Create output directories
        self.results_dir = Path("experiment_results")
        self.plots_dir = Path("plots")
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Results storage
        self.all_results = {}
        self.summary_statistics = {}
        
    def _load_data(self):
        """Load and prepare the Alzheimer's dataset"""
        self.logger.info(f"Loading data from {self.data_file}")
        
        data_loader = AlzheimerDataLoader()
        self.X_data, self.y_data, self.categorical_features, self.numerical_features = \
            data_loader.load_data(self.data_file)
        
        self.logger.info(f"Data loaded: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} features")
        self.logger.info(f"Categorical features: {len(self.categorical_features)}")
        self.logger.info(f"Numerical features: {len(self.numerical_features)}")
    
    def create_experiment_configurations(self) -> List[Dict[str, Any]]:
        """Create all experiment configurations from the provided table"""
        
        # Base configuration
        base_config = {
            'max_generations': 1000,
            'early_stopping_patience': 50,
            'min_improvement_threshold': 0.01,
            'tournament_size': 5,
            'alpha': 0.3,
            'random_seed': 42
        }
        
        # Experiment configurations from the table
        experiments = [
            {'id': 1, 'population_size': 20, 'crossover_rate': 0.6, 'mutation_rate': 0.00},
            {'id': 2, 'population_size': 20, 'crossover_rate': 0.6, 'mutation_rate': 0.01},
            {'id': 3, 'population_size': 20, 'crossover_rate': 0.6, 'mutation_rate': 0.10},
            {'id': 4, 'population_size': 20, 'crossover_rate': 0.9, 'mutation_rate': 0.01},
            {'id': 5, 'population_size': 20, 'crossover_rate': 0.1, 'mutation_rate': 0.01},
            {'id': 6, 'population_size': 200, 'crossover_rate': 0.6, 'mutation_rate': 0.00},
            {'id': 7, 'population_size': 200, 'crossover_rate': 0.6, 'mutation_rate': 0.01},
            {'id': 8, 'population_size': 200, 'crossover_rate': 0.6, 'mutation_rate': 0.10},
            {'id': 9, 'population_size': 200, 'crossover_rate': 0.9, 'mutation_rate': 0.01},
            {'id': 10, 'population_size': 200, 'crossover_rate': 0.1, 'mutation_rate': 0.01},
        ]
        
        # Create full configurations
        full_experiments = []
        for exp in experiments:
            # Combine base config with experiment-specific parameters (excluding 'id')
            config_dict = base_config.copy()
            exp_params = {k: v for k, v in exp.items() if k != 'id'}
            config_dict.update(exp_params)
            
            full_experiments.append({
                'id': exp['id'],
                'name': f"Exp{exp['id']}_Pop{exp['population_size']}_Cx{exp['crossover_rate']}_Mut{exp['mutation_rate']:.2f}",
                'config': GAConfig(**config_dict),
                'parameters': exp_params
            })
        
        return full_experiments
    
    def run_single_ga_run(self, config: GAConfig, experiment_id: int, run_id: int) -> Dict[str, Any]:
        """Run a single GA execution"""
        
        # Set unique random seed for this run across ALL experiments and runs
        # Formula: base_seed + (experiment_id - 1) * 10000 + run_id * 1000
        run_config = GAConfig(**config.to_dict())   # Copy the config
        unique_seed = config.random_seed + (experiment_id - 1) * 10000 + run_id * 1000
        run_config.random_seed = unique_seed
        set_seed(unique_seed)
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(
            config=run_config,
            X_data=self.X_data,
            y_data=self.y_data,
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features
        )
        
        # Create genetic operators
        selection_operator = TournamentSelection(tournament_size=run_config.tournament_size)
        crossover_operator = UniformCrossover()
        mutation_operator = BitFlipMutation(mutation_rate=run_config.mutation_rate)
        
        # Create and run GA
        ga = GeneticAlgorithm(
            config=run_config,
            fitness_evaluator=fitness_evaluator,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator
        )
        
        # Run evolution
        result = ga.run(num_features=len(self.X_data.columns))
        
        return result
    
    def run_experiment_configuration(self, experiment: Dict[str, Any], num_runs: int = 10) -> Dict[str, Any]:
        """Run a single experiment configuration with multiple runs"""
        
        exp_name = experiment['name']
        exp_config = experiment['config']
        experiment_id = experiment['id']
        
        self.logger.info(f"Starting experiment: {exp_name}")
        self.logger.info(f"Parameters: {experiment['parameters']}")
        
        run_results = []
        successful_runs = 0
        
        # Store fitness histories for averaging
        all_fitness_histories = []
        all_diversity_histories = []
        
        for run_id in range(num_runs):
            self.logger.info(f"  Run {run_id + 1}/{num_runs}")
            
            try:
                result = self.run_single_ga_run(exp_config, experiment_id, run_id)
                result['run_id'] = run_id
                result['success'] = True
                
                run_results.append(result)
                successful_runs += 1
                
                # Store histories for averaging
                if 'fitness_history' in result:
                    all_fitness_histories.append(result['fitness_history'])
                if 'diversity_history' in result:
                    all_diversity_histories.append(result['diversity_history'])
                
                # Log run result
                best_individual = result.get('best_individual', {})
                best_fitness = best_individual.get('fitness', float('inf'))
                best_features = best_individual.get('num_selected_features', 0)
                generations = result.get('generations_run', 0)
                
                self.logger.info(f"    Run {run_id + 1}: "
                               f"Fitness={best_fitness:.4f}, Features={best_features}, Generations={generations}")
                
            except Exception as e:
                self.logger.error(f"    Run {run_id + 1} failed: {str(e)}")
                run_results.append({
                    'run_id': run_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate mean evolution curves
        mean_fitness_history, mean_diversity_history = self._calculate_mean_histories(
            all_fitness_histories, all_diversity_histories
        )
        
        # Compile experiment results
        experiment_result = {
            'experiment_info': {
                'id': experiment['id'],
                'name': exp_name,
                'parameters': experiment['parameters'],
                'config': exp_config.to_dict(),
                'num_runs': num_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / num_runs
            },
            'runs': run_results,
            'mean_fitness_history': mean_fitness_history,
            'mean_diversity_history': mean_diversity_history
        }
        
        # Calculate statistics for successful runs
        successful_results = [r for r in run_results if r.get('success', False)]
        if successful_results:
            experiment_result['statistics'] = self._calculate_experiment_statistics(successful_results)
        
        return experiment_result
    
    def _calculate_mean_histories(self, fitness_histories: List[List[float]], 
                                diversity_histories: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Create averaged evolution curves from multiple GA runs with possibly different convergence times"""
        
        if not fitness_histories:
            return [], []
        
        # Find the maximum length to pad shorter histories
        max_length = max(len(history) for history in fitness_histories)
        
        # Pad histories with their last value
        padded_fitness = []
        for history in fitness_histories:
            padded = history + [history[-1]] * (max_length - len(history))
            padded_fitness.append(padded)
        
        padded_diversity = []
        for history in diversity_histories:
            padded = history + [history[-1]] * (max_length - len(history))
            padded_diversity.append(padded)
        
        # Calculate means
        mean_fitness = np.mean(padded_fitness, axis=0).tolist()
        mean_diversity = np.mean(padded_diversity, axis=0).tolist()
        
        return mean_fitness, mean_diversity
    
    def _calculate_experiment_statistics(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for an experiment"""
        
        # Extract metrics
        fitness_values = [r.get('best_individual', {}).get('fitness', float('inf')) for r in successful_results]
        feature_counts = [r.get('best_individual', {}).get('num_selected_features', 0) for r in successful_results]
        generations = [r.get('generations_run', 0) for r in successful_results]
        convergence_flags = [r.get('converged', False) for r in successful_results]
        
        # Calculate statistics
        stats = {
            'best_fitness': {
                'mean': float(np.mean(fitness_values)),
                'std': float(np.std(fitness_values)),
                'min': float(np.min(fitness_values)),
                'max': float(np.max(fitness_values)),
                'median': float(np.median(fitness_values))
            },
            'num_features': {
                'mean': float(np.mean(feature_counts)),
                'std': float(np.std(feature_counts)),
                'min': int(np.min(feature_counts)),
                'max': int(np.max(feature_counts)),
                'median': float(np.median(feature_counts))
            },
            'generations': {
                'mean': float(np.mean(generations)),
                'std': float(np.std(generations)),
                'min': int(np.min(generations)),
                'max': int(np.max(generations)),
                'median': float(np.median(generations)) 
            },
            'convergence': {
                'rate': sum(convergence_flags) / len(convergence_flags),
                'converged_runs': sum(convergence_flags),
                'total_runs': len(convergence_flags)
            }
        }
        
        return stats
    
    def run_all_experiments(self, num_runs_per_experiment: int = 10) -> Dict[str, Any]:
        """Run all experiment configurations"""
        
        self.logger.info("="*80)
        self.logger.info("STARTING COMPREHENSIVE GA EXPERIMENTS")
        self.logger.info("="*80)
        
        # Create experiments
        experiments = self.create_experiment_configurations()
        
        self.logger.info(f"Created {len(experiments)} experiment configurations:")
        for exp in experiments:
            self.logger.info(f"  {exp['id']}. {exp['name']}: {exp['parameters']}")
        
        # Run all experiments
        for experiment in experiments:
            try:
                result = self.run_experiment_configuration(experiment, num_runs_per_experiment)
                self.all_results[experiment['name']] = result
                
                # Log experiment summary
                if 'statistics' in result:
                    stats = result['statistics']
                    self.logger.info(f"Experiment {experiment['name']} completed:")
                    self.logger.info(f"  Mean best fitness: {stats['best_fitness']['mean']:.4f} ± {stats['best_fitness']['std']:.4f}")
                    self.logger.info(f"  Mean features: {stats['num_features']['mean']:.1f} ± {stats['num_features']['std']:.1f}")
                    self.logger.info(f"  Success rate: {result['experiment_info']['success_rate']:.1%}")
                
            except Exception as e:
                self.logger.error(f"Experiment {experiment['name']} failed: {e}")
                self.all_results[experiment['name']] = {
                    'experiment_info': {
                        'name': experiment['name'],
                        'success_rate': 0.0,
                        'error': str(e)
                    }
                }
        
        # Compile session results
        session_results = {
            'session_info': {
                'total_experiments': len(experiments),
                'num_runs_per_experiment': num_runs_per_experiment
            },
            'experiments': self.all_results
        }
        
        # Calculate summary statistics
        self._calculate_summary_statistics()
        
        # Save results
        self._save_results(session_results)
        
        # Generate visualizations
        self._generate_all_visualizations()
        
        # Generate analysis report
        self._generate_analysis_report()
        
        # Print summary
        self._print_comprehensive_summary()
        
        return session_results
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics across all experiments"""
        
        summary = {}
        
        for exp_name, exp_result in self.all_results.items():
            if 'statistics' in exp_result:
                exp_info = exp_result['experiment_info']
                stats = exp_result['statistics']
                
                summary[exp_name] = {
                    'id': exp_info['id'],
                    'parameters': exp_info['parameters'],
                    'mean_best_fitness': stats['best_fitness']['mean'],
                    'std_best_fitness': stats['best_fitness']['std'],
                    'mean_num_features': stats['num_features']['mean'],
                    'std_num_features': stats['num_features']['std'],
                    'mean_generations': stats['generations']['mean'],
                    'convergence_rate': stats['convergence']['rate'],
                    'success_rate': exp_info['success_rate']
                }
        
        self.summary_statistics = summary
    
    def _generate_all_visualizations(self):
        """Generate all required visualizations"""
        
        self.logger.info("Generating visualizations...")
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Evolution curves for each experiment
        self._plot_evolution_curves()
        
        # 2. Parameter effect analysis
        self._plot_parameter_effects()
        
        # 3. Convergence analysis
        self._plot_convergence_analysis()
        
        # 4. Feature selection analysis
        self._plot_feature_analysis()
        
        self.logger.info(f"All plots saved to {self.plots_dir}")
    
    def _plot_evolution_curves(self):
        """Plot evolution curves for each experiment configuration"""
        
        # Create subplots for all experiments
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.suptitle('Evolution Curves: Mean Best Fitness vs Generation', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (exp_name, exp_result) in enumerate(self.all_results.items()):
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
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'evolution_curves_all.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual plots for each experiment
        for exp_name, exp_result in self.all_results.items():
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
                plt.savefig(self.plots_dir / f'evolution_curve_{exp_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_parameter_effects(self):
        """Plot the effects of different parameters"""
        
        if not self.summary_statistics:
            return
        
        # Create DataFrame for analysis
        data = []
        for exp_name, stats in self.summary_statistics.items():
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
        
        # Population size effect
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Effects Analysis', fontsize=16)
        
        # Population size vs fitness
        pop_groups = df.groupby('population_size')['mean_fitness'].agg(['mean', 'std']).reset_index()
        axes[0, 0].bar(pop_groups['population_size'].astype(str), pop_groups['mean'], 
                      yerr=pop_groups['std'], capsize=5)
        axes[0, 0].set_title('Population Size Effect on Fitness')
        axes[0, 0].set_xlabel('Population Size')
        axes[0, 0].set_ylabel('Mean Best Fitness')
        
        # Crossover rate vs fitness
        cx_groups = df.groupby('crossover_rate')['mean_fitness'].agg(['mean', 'std']).reset_index()
        axes[0, 1].bar(cx_groups['crossover_rate'].astype(str), cx_groups['mean'], 
                      yerr=cx_groups['std'], capsize=5)
        axes[0, 1].set_title('Crossover Rate Effect on Fitness')
        axes[0, 1].set_xlabel('Crossover Rate')
        axes[0, 1].set_ylabel('Mean Best Fitness')
        
        # Mutation rate vs fitness
        mut_groups = df.groupby('mutation_rate')['mean_fitness'].agg(['mean', 'std']).reset_index()
        axes[1, 0].bar(mut_groups['mutation_rate'].astype(str), mut_groups['mean'], 
                      yerr=mut_groups['std'], capsize=5)
        axes[1, 0].set_title('Mutation Rate Effect on Fitness')
        axes[1, 0].set_xlabel('Mutation Rate')
        axes[1, 0].set_ylabel('Mean Best Fitness')
        
        # Convergence rate comparison
        axes[1, 1].bar(range(len(df)), df['convergence_rate'], color='green', alpha=0.7)
        axes[1, 1].set_title('Convergence Rate by Experiment')
        axes[1, 1].set_xlabel('Experiment ID')
        axes[1, 1].set_ylabel('Convergence Rate')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels([f"Exp{i+1}" for i in range(len(df))], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self):
        """Plot convergence analysis"""
        
        if not self.summary_statistics:
            return
        
        # Create convergence comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        exp_names = list(self.summary_statistics.keys())
        convergence_rates = [stats['convergence_rate'] for stats in self.summary_statistics.values()]
        mean_generations = [stats['mean_generations'] for stats in self.summary_statistics.values()]
        
        # Convergence rate by experiment
        bars1 = ax1.bar(range(len(exp_names)), convergence_rates, color='skyblue', alpha=0.8)
        ax1.set_title('Convergence Rate by Experiment')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Convergence Rate')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, convergence_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Mean generations to convergence
        bars2 = ax2.bar(range(len(exp_names)), mean_generations, color='lightcoral', alpha=0.8)
        ax2.set_title('Mean Generations to Convergence')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Mean Generations')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels([f"Exp{i+1}" for i in range(len(exp_names))], rotation=45)
        
        # Add value labels on bars
        for bar, gens in zip(bars2, mean_generations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{gens:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self):
        """Plot feature selection analysis"""
        
        if not self.summary_statistics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        exp_names = list(self.summary_statistics.keys())
        mean_features = [stats['mean_num_features'] for stats in self.summary_statistics.values()]
        std_features = [stats['std_num_features'] for stats in self.summary_statistics.values()]
        
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
        for bar, pct in zip(bars2, reduction_pct):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        
        report_path = self.results_dir / 'analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE GENETIC ALGORITHM ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total experiments conducted: {len(self.all_results)}\n")
            f.write(f"Runs per experiment: 10\n")
            f.write(f"Dataset: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} features\n\n")
            
            f.write("PARAMETER EFFECTS ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Population size analysis
            f.write("1. POPULATION SIZE EFFECT:\n")
            pop20_exps = [stats for stats in self.summary_statistics.values() 
                         if stats['parameters']['population_size'] == 20]
            pop200_exps = [stats for stats in self.summary_statistics.values() 
                          if stats['parameters']['population_size'] == 200]
            
            if pop20_exps and pop200_exps:
                pop20_fitness = np.mean([exp['mean_best_fitness'] for exp in pop20_exps])
                pop200_fitness = np.mean([exp['mean_best_fitness'] for exp in pop200_exps])
                
                f.write(f"   - Population 20: Mean fitness = {pop20_fitness:.4f}\n")
                f.write(f"   - Population 200: Mean fitness = {pop200_fitness:.4f}\n")
                f.write(f"   - Improvement with larger population: {((pop20_fitness - pop200_fitness) / pop20_fitness * 100):.2f}%\n\n")
            
            # Crossover rate analysis
            f.write("2. CROSSOVER RATE EFFECT:\n")
            for cx_rate in [0.1, 0.6, 0.9]:
                cx_exps = [stats for stats in self.summary_statistics.values() 
                          if stats['parameters']['crossover_rate'] == cx_rate]
                if cx_exps:
                    cx_fitness = np.mean([exp['mean_best_fitness'] for exp in cx_exps])
                    f.write(f"   - Crossover {cx_rate}: Mean fitness = {cx_fitness:.4f}\n")
            f.write("\n")
            
            # Mutation rate analysis
            f.write("3. MUTATION RATE EFFECT:\n")
            for mut_rate in [0.00, 0.01, 0.10]:
                mut_exps = [stats for stats in self.summary_statistics.values() 
                           if abs(stats['parameters']['mutation_rate'] - mut_rate) < 0.001]
                if mut_exps:
                    mut_fitness = np.mean([exp['mean_best_fitness'] for exp in mut_exps])
                    f.write(f"   - Mutation {mut_rate:.2f}: Mean fitness = {mut_fitness:.4f}\n")
            f.write("\n")
            
            f.write("CONVERGENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Best performing experiments
            sorted_exps = sorted(self.summary_statistics.items(), 
                               key=lambda x: x[1]['mean_best_fitness'])
            
            f.write("TOP 3 BEST PERFORMING CONFIGURATIONS:\n")
            for i, (exp_name, stats) in enumerate(sorted_exps[:3]):
                f.write(f"{i+1}. {exp_name}:\n")
                f.write(f"   - Mean fitness: {stats['mean_best_fitness']:.4f} ± {stats['std_best_fitness']:.4f}\n")
                f.write(f"   - Mean features: {stats['mean_num_features']:.1f}\n")
                f.write(f"   - Convergence rate: {stats['convergence_rate']:.2%}\n")
                f.write(f"   - Parameters: Pop={stats['parameters']['population_size']}, "
                       f"Cx={stats['parameters']['crossover_rate']}, "
                       f"Mut={stats['parameters']['mutation_rate']:.2f}\n\n")
            
            f.write("CONCLUSIONS AND RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Based on the experimental results:\n\n")
            
            # Add specific conclusions based on results
            best_exp = sorted_exps[0][1]
            f.write(f"1. Best configuration uses population size {best_exp['parameters']['population_size']}, "
                   f"crossover rate {best_exp['parameters']['crossover_rate']}, "
                   f"and mutation rate {best_exp['parameters']['mutation_rate']:.2f}\n\n")
            
            f.write("2. Parameter recommendations:\n")
            f.write("   - Population size: Larger populations generally provide better exploration\n")
            f.write("   - Crossover rate: Moderate rates (0.6) often perform well\n")
            f.write("   - Mutation rate: Low rates (0.01) provide good balance between exploration and exploitation\n\n")
            
            f.write("3. Feature selection effectiveness:\n")
            avg_reduction = np.mean([((len(self.X_data.columns) - stats['mean_num_features']) / 
                                    len(self.X_data.columns) * 100) 
                                   for stats in self.summary_statistics.values()])
            f.write(f"   - Average feature reduction: {avg_reduction:.1f}%\n")
            f.write(f"   - Original features: {len(self.X_data.columns)}\n")
            f.write(f"   - Typical selected features: {np.mean([stats['mean_num_features'] for stats in self.summary_statistics.values()]):.1f}\n")
        
        self.logger.info(f"Analysis report saved to {report_path}")
    
    def _save_results(self, session_results: Dict[str, Any]):
        """Save comprehensive results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"comprehensive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(session_results, f, indent=2, cls=NumpyEncoder)
        
        # Save summary statistics
        summary_file = self.results_dir / f"summary_statistics_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_statistics, f, indent=2, cls=NumpyEncoder)
        
        # Save CSV summary for easy analysis
        if self.summary_statistics:
            df_summary = pd.DataFrame.from_dict(self.summary_statistics, orient='index')
            csv_file = self.results_dir / f"summary_table_{timestamp}.csv"
            df_summary.to_csv(csv_file)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def _print_comprehensive_summary(self):
        """Print comprehensive summary"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        if not self.summary_statistics:
            print("No results to display.")
            return
        
        # Create summary table
        print(f"\n{'ID':<3} {'Population':<10} {'Crossover':<9} {'Mutation':<8} {'Mean Fitness':<12} {'±Std':<8} {'Features':<8} {'Conv.Rate':<9}")
        print("-" * 80)
        
        for exp_name, stats in self.summary_statistics.items():
            print(f"{stats['id']:<3} "
                  f"{stats['parameters']['population_size']:<10} "
                  f"{stats['parameters']['crossover_rate']:<9.1f} "
                  f"{stats['parameters']['mutation_rate']:<8.2f} "
                  f"{stats['mean_best_fitness']:<12.4f} "
                  f"{stats['std_best_fitness']:<8.4f} "
                  f"{stats['mean_num_features']:<8.1f} "
                  f"{stats['convergence_rate']:<9.2%}")
        
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY:")
        print("- Evolution curves plotted for each configuration")
        print("- Parameter effects analyzed and visualized")
        print("- Convergence patterns documented")
        print("- Feature selection effectiveness measured")
        print("- Comprehensive analysis report generated")
        print("="*80)


def main():
    """Main function to run comprehensive experiments"""
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/comprehensive_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(__name__, log_file)
    
    try:
        # Create experiment runner
        runner = ComprehensiveExperimentRunner("alzheimers_disease_data.csv")
        
        # Run all experiments with 10 runs each
        results = runner.run_all_experiments(num_runs_per_experiment=10)
        
        logger.info("Comprehensive experiments completed successfully!")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Results saved to: {runner.results_dir}")
        logger.info(f"Plots saved to: {runner.plots_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive experiments failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 