"""
Main Genetic Algorithm controller
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

from ..config import GAConfig
from ..utils import set_seed
from ..genetic import Population, Individual, TournamentSelection, UniformCrossover, BitFlipMutation
from ..genetic.fitness_evaluator import FitnessEvaluator


class GeneticAlgorithm:
    """
    Main controller for the Genetic Algorithm evolution process
    """
    
    def __init__(self, 
                 config: GAConfig,
                 fitness_evaluator: FitnessEvaluator,
                 selection_operator: TournamentSelection,
                 crossover_operator: UniformCrossover,
                 mutation_operator: BitFlipMutation):
        """
        Initialize genetic algorithm
        
        Args:
            config: GA configuration parameters
            fitness_evaluator: Fitness evaluation strategy
            selection_operator: Selection strategy
            crossover_operator: Crossover strategy
            mutation_operator: Mutation strategy
        """
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        
        # Evolution state
        self.population: Optional[Population] = None
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.no_improvement_counter = 0  # Criterion i: no improvement at all
        self.low_improvement_counter = 0  # Criterion ii: improvement below threshold
        self.termination_reason: Optional[str] = None
        
        # Timing
        self.start_time: Optional[float] = None
        self.generation_times: List[float] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Validation
        self.config.validate()
    
    def run(self, num_features: int) -> Dict:
        """
        Run the complete genetic algorithm evolution
        
        Args:
            num_features: Number of features in the dataset
            
        Returns:
            Dictionary with evolution results
        """
        self.logger.info("Starting Genetic Algorithm evolution")
        self.logger.info(f"Configuration: Population={self.config.population_size}, "
                        f"Generations={self.config.max_generations}, "
                        f"Features={num_features}")
        
        # Set random seed for reproducibility
        set_seed(self.config.random_seed)
        
        # Initialize population
        self._initialize_population(num_features)
        self.start_time = time.time()
        
        try:
            # Evolution loop
            for self.generation in range(self.config.max_generations):
                generation_start = time.time()
                
                # Evaluate population
                self._evaluate_population()
                
                # Update statistics
                self._update_statistics()
                
                # Check termination conditions
                if self._should_terminate():
                    self.logger.info(f"Early termination at generation {self.generation}")
                    break
                
                # Create next generation
                self._evolve_population()
                
                # Log progress
                generation_time = time.time() - generation_start
                self.generation_times.append(generation_time)
                self._log_generation_progress(generation_time)
            
            # Check if terminated due to max generations
            if self.termination_reason is None:
                self.termination_reason = f"Maximum generations reached ({self.config.max_generations})"
                self.logger.info(self.termination_reason)
            
            # Final evaluation and results
            self._evaluate_population()
            self._update_statistics()
            
            return self._compile_results()
            
        except Exception as e:
            self.logger.error(f"Error during evolution: {e}")
            raise
        finally:
            total_time = time.time() - self.start_time if self.start_time else 0
            self.logger.info(f"Evolution completed in {total_time:.2f} seconds")
    
    def _initialize_population(self, num_features: int) -> None:
        """Initialize the population with random individuals"""
        self.logger.info("Initializing population...")
        self.population = Population(self.config.population_size, num_features)
        self.generation = 0
        self.no_improvement_counter = 0
        self.low_improvement_counter = 0
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population"""
        if not self.population:
            raise RuntimeError("Population not initialized")
        
        # Evaluate all individuals (fitness evaluator handles caching)
        for individual in self.population.individuals:
            self.fitness_evaluator.evaluate_individual(individual)
        
        # Update best individual
        self.population.update_best_individual()
        
        self.logger.debug(f"Population evaluation completed (cache size: {len(self.fitness_evaluator.fitness_cache)})")
    
    def _update_statistics(self) -> None:
        """Update evolution statistics"""
        if not self.population:
            return
        
        stats = self.population.get_statistics()
        
        # Update fitness history
        if 'best_fitness' in stats:
            self.best_fitness_history.append(stats['best_fitness'])
        
        # Update diversity history
        self.diversity_history.append(stats['diversity'])
        
        # Update termination counters based on improvement
        if len(self.best_fitness_history) >= 2:
            current_fitness = self.best_fitness_history[-1]
            previous_fitness = self.best_fitness_history[-2]
            
            # Check for any improvement (criterion i: no improvement at all)
            if current_fitness >= previous_fitness:  # No improvement (fitness should decrease)
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0
            
            # Check for significant improvement (criterion ii: improvement below threshold)
            if previous_fitness > 0:
                improvement_percentage = abs(previous_fitness - current_fitness) / previous_fitness
            else:
                improvement_percentage = float('inf') if current_fitness != previous_fitness else 0
            
            if improvement_percentage < self.config.min_improvement_threshold:
                self.low_improvement_counter += 1
            else:
                self.low_improvement_counter = 0
    
    def _should_terminate(self) -> bool:
        """Check if evolution should terminate early based on multiple criteria"""
        
        # Criterion i: No improvement at all for specified number of generations
        if self.no_improvement_counter >= self.config.early_stopping_patience:
            self.termination_reason = f"No improvement for {self.no_improvement_counter} generations"
            self.logger.info(self.termination_reason)
            return True
        
        # Criterion ii: Improvement below threshold (<1%) for specified number of generations
        if self.low_improvement_counter >= self.config.early_stopping_patience:
            self.termination_reason = f"Improvement below threshold for {self.low_improvement_counter} generations (threshold: {self.config.min_improvement_threshold:.1%})"
            self.logger.info(self.termination_reason)
            return True
        
        # Criterion iii: Maximum generations reached (handled in main loop)
        # This is already handled by the for loop in run() method
        
        return False
    
    def _evolve_population(self) -> None:
        """Create the next generation through selection, crossover, and mutation"""
        if not self.population:
            raise RuntimeError("Population not initialized")
        
        new_individuals = []
        
        # Elitism - keep best individual
        if self.population.best_individual:
            new_individuals.append(self.population.best_individual.copy())
        
        # Generate offspring to fill the rest of the population
        while len(new_individuals) < self.config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if len(new_individuals) < self.config.population_size - 1:
                # Create two children if we have space
                child1, child2 = self._create_offspring(parent1, parent2)
                new_individuals.extend([child1, child2])
            else:
                # Create one child if we only have space for one
                child1, _ = self._create_offspring(parent1, parent2)
                new_individuals.append(child1)
        
        # Trim to exact population size (in case we added one too many)
        new_individuals = new_individuals[:self.config.population_size]
        
        # Replace population
        self.population.replace_individuals(new_individuals)
    
    def _select_parent(self) -> Individual:
        """Select a parent using the configured selection strategy"""
        if not self.population:
            raise RuntimeError("Population not initialized")
        
        return self.selection_operator.select(self.population.individuals)
    
    def _create_offspring(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring through crossover and mutation"""
        # Crossover
        if self.config.crossover_rate > 0 and \
           len(self.population.individuals) > 1:  # Need at least 2 parents for crossover
            
            import random
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        # Mutation
        if self.config.mutation_rate > 0:
            child1 = self.mutation_operator.mutate(child1)
            child2 = self.mutation_operator.mutate(child2)
        
        return child1, child2
    
    def _log_generation_progress(self, generation_time: float) -> None:
        """Log progress for current generation"""
        if not self.population:
            return
        
        stats = self.population.get_statistics()
        
        if self.generation % 10 == 0 or self.generation < 10:  # Log every 10 generations or first 10
            self.logger.info(
                f"Generation {self.generation:3d}: "
                f"Best={stats.get('best_fitness', 'N/A'):.4f}, "
                f"Avg={stats.get('avg_fitness', 'N/A'):.4f}, "
                f"Diversity={stats.get('diversity', 'N/A'):.3f}, "
                f"Time={generation_time:.2f}s"
            )
    
    def _compile_results(self) -> Dict:
        """Compile final evolution results"""
        if not self.population:
            raise RuntimeError("Population not initialized")
        
        final_stats = self.population.get_statistics()
        total_time = time.time() - self.start_time if self.start_time else 0
        
        results = {
            # Best solution (serializable format)
            'best_individual': {
                'chromosome': self.population.best_individual.chromosome.tolist() if self.population.best_individual else None,
                'fitness': self.population.best_individual.fitness if self.population.best_individual else None,
                'selected_features': (self.population.best_individual.get_selected_features() 
                                     if self.population.best_individual else None),
                'num_selected_features': (self.population.best_individual.num_selected_features 
                                         if self.population.best_individual else None)
            },
            'best_fitness': self.population.best_individual.fitness if self.population.best_individual else None,
            'best_features': (self.population.best_individual.get_selected_features() 
                             if self.population.best_individual else None),
            'best_num_features': (self.population.best_individual.num_selected_features 
                                 if self.population.best_individual else None),
            
            # Evolution statistics
            'generations_completed': self.generation + 1,
            'generations_run': self.generation + 1,  # Alias for compatibility
            'converged': (self.no_improvement_counter >= self.config.early_stopping_patience or 
                         self.low_improvement_counter >= self.config.early_stopping_patience),
            'convergence_generation': None,  # Will be set below
            'termination_reason': self.termination_reason,
            'no_improvement_counter': self.no_improvement_counter,
            'low_improvement_counter': self.low_improvement_counter,
            
            # Performance metrics
            'total_time': total_time,
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            'fitness_evaluations': len(self.fitness_evaluator.fitness_cache),
            
            # Evolution history
            'fitness_history': self.best_fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
            
            # Final population statistics
            'final_population_stats': final_stats,
            
            # Configuration used
            'config': self.config.to_dict(),
            
            # Cache statistics
            'cache_stats': self.fitness_evaluator.get_cache_stats(),
            'fitness_distribution': self.fitness_evaluator.get_fitness_distribution()
        }
        
        # Set convergence generation based on which criterion was triggered
        if self.no_improvement_counter >= self.config.early_stopping_patience:
            results['convergence_generation'] = self.generation - self.no_improvement_counter + 1
        elif self.low_improvement_counter >= self.config.early_stopping_patience:
            results['convergence_generation'] = self.generation - self.low_improvement_counter + 1
        
        self.logger.info(f"Evolution results: {self.generation + 1} generations, "
                        f"Best fitness: {results['best_fitness']:.4f}, "
                        f"Best features: {results['best_num_features']}")
        
        return results 