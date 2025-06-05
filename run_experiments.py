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
import numpy as np
from pathlib import Path

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

# Import new visualization and reporting modules
from src.visualization.ga_visualizer import GAVisualizer
from src.reporting.ga_reporter import GAReporter


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
        
        # Initialize visualization and reporting modules
        self.visualizer = GAVisualizer(self.plots_dir, self.X_data)
        self.reporter = GAReporter(self.results_dir, self.X_data)
        
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
        
        # Use the new visualizer module
        self.visualizer.generate_all_plots(self.all_results, self.summary_statistics)
        
        self.logger.info(f"All plots saved to {self.plots_dir}")
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        
        # Use the new reporter module
        self.reporter.generate_report(self.all_results, self.summary_statistics)
    
    def _save_results(self, session_results: Dict[str, Any]):
        """Save comprehensive results"""
        
        # Use the new reporter module
        self.reporter.save_results(session_results, self.summary_statistics)
    
    def _print_comprehensive_summary(self):
        """Print comprehensive summary"""
        
        # Use the new reporter module
        self.reporter.print_comprehensive_summary(self.summary_statistics)


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