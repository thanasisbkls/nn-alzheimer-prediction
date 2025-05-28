#!/usr/bin/env python3
"""
GA Reporter Module

Reporting functions for Genetic Algorithm experiment results.
Extracts all reporting functionality from the GA experiments runner.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .base_reporter import BaseReporter


class GAReporter(BaseReporter):
    """
    Reporter for Genetic Algorithm experiment results
    """
    
    def __init__(self, results_dir: Path, X_data: pd.DataFrame):
        """
        Initialize GA reporter
        
        Args:
            results_dir: Directory to save reports
            X_data: Original dataset for feature analysis
        """
        super().__init__(results_dir)
        self.X_data = X_data
    
    def generate_report(self, all_results: Dict[str, Any], summary_statistics: Dict[str, Any]) -> Path:
        """
        Generate comprehensive GA analysis report
        
        Args:
            all_results: Complete experiment results
            summary_statistics: Summary statistics across experiments
            
        Returns:
            Path to generated report
        """
        report_path = self.results_dir / 'analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Main header
            f.write("COMPREHENSIVE GENETIC ALGORITHM ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive summary
            self._write_executive_summary(f, all_results, summary_statistics)
            
            # Parameter effects analysis
            self._write_parameter_effects_analysis(f, summary_statistics)
            
            # Convergence analysis
            self._write_convergence_analysis(f, summary_statistics)
            
            # Conclusions and recommendations
            self._write_conclusions_and_recommendations(f, summary_statistics)
        
        return report_path
    
    def _write_executive_summary(self, f, all_results: Dict[str, Any], summary_statistics: Dict[str, Any]):
        """Write executive summary section"""
        self.write_section_header(f, "EXECUTIVE SUMMARY", level=2)
        
        f.write(f"Total experiments conducted: {len(all_results)}\n")
        f.write(f"Runs per experiment: 10\n")
        f.write(f"Dataset: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} features\n\n")
    
    def _write_parameter_effects_analysis(self, f, summary_statistics: Dict[str, Any]):
        """Write parameter effects analysis section"""
        self.write_section_header(f, "PARAMETER EFFECTS ANALYSIS", level=2)
        
        # Population size analysis
        f.write("1. POPULATION SIZE EFFECT:\n")
        pop20_exps = [stats for stats in summary_statistics.values() 
                     if stats['parameters']['population_size'] == 20]
        pop200_exps = [stats for stats in summary_statistics.values() 
                      if stats['parameters']['population_size'] == 200]
        
        if pop20_exps and pop200_exps:
            pop20_fitness = np.mean([exp['mean_best_fitness'] for exp in pop20_exps])
            pop200_fitness = np.mean([exp['mean_best_fitness'] for exp in pop200_exps])
            
            f.write(f"   - Population 20: Mean fitness = {self.format_number(pop20_fitness)}\n")
            f.write(f"   - Population 200: Mean fitness = {self.format_number(pop200_fitness)}\n")
            improvement = ((pop20_fitness - pop200_fitness) / pop20_fitness * 100)
            f.write(f"   - Improvement with larger population: {self.format_number(improvement, 2)}%\n\n")
        
        # Crossover rate analysis
        f.write("2. CROSSOVER RATE EFFECT:\n")
        for cx_rate in [0.1, 0.6, 0.9]:
            cx_exps = [stats for stats in summary_statistics.values() 
                      if stats['parameters']['crossover_rate'] == cx_rate]
            if cx_exps:
                cx_fitness = np.mean([exp['mean_best_fitness'] for exp in cx_exps])
                f.write(f"   - Crossover {cx_rate}: Mean fitness = {self.format_number(cx_fitness)}\n")
        f.write("\n")
        
        # Mutation rate analysis
        f.write("3. MUTATION RATE EFFECT:\n")
        for mut_rate in [0.00, 0.01, 0.10]:
            mut_exps = [stats for stats in summary_statistics.values() 
                       if abs(stats['parameters']['mutation_rate'] - mut_rate) < 0.001]
            if mut_exps:
                mut_fitness = np.mean([exp['mean_best_fitness'] for exp in mut_exps])
                f.write(f"   - Mutation {self.format_number(mut_rate, 2)}: Mean fitness = {self.format_number(mut_fitness)}\n")
        f.write("\n")
    
    def _write_convergence_analysis(self, f, summary_statistics: Dict[str, Any]):
        """Write convergence analysis section"""
        self.write_section_header(f, "CONVERGENCE ANALYSIS", level=2)
        
        # Best performing experiments
        sorted_exps = sorted(summary_statistics.items(), 
                           key=lambda x: x[1]['mean_best_fitness'])
        
        f.write("TOP 3 BEST PERFORMING CONFIGURATIONS:\n")
        for i, (exp_name, stats) in enumerate(sorted_exps[:3]):
            self.write_numbered_point(f, i+1, f"{exp_name}:")
            f.write(f"   - Mean fitness: {self.format_number(stats['mean_best_fitness'])} ± {self.format_number(stats['std_best_fitness'])}\n")
            f.write(f"   - Mean features: {self.format_number(stats['mean_num_features'], 1)}\n")
            f.write(f"   - Convergence rate: {self.format_percentage(stats['convergence_rate'])}\n")
            f.write(f"   - Parameters: Pop={stats['parameters']['population_size']}, "
                   f"Cx={stats['parameters']['crossover_rate']}, "
                   f"Mut={self.format_number(stats['parameters']['mutation_rate'], 2)}\n\n")
    
    def _write_conclusions_and_recommendations(self, f, summary_statistics: Dict[str, Any]):
        """Write conclusions and recommendations section"""
        self.write_section_header(f, "CONCLUSIONS AND RECOMMENDATIONS", level=2)
        f.write("Based on the experimental results:\n\n")
        
        # Add specific conclusions based on results
        sorted_exps = sorted(summary_statistics.items(), 
                           key=lambda x: x[1]['mean_best_fitness'])
        best_exp = sorted_exps[0][1]
        
        self.write_numbered_point(f, 1, 
            f"Best configuration uses population size {best_exp['parameters']['population_size']}, "
            f"crossover rate {best_exp['parameters']['crossover_rate']}, "
            f"and mutation rate {self.format_number(best_exp['parameters']['mutation_rate'], 2)}\n")
        
        self.write_numbered_point(f, 2, "Parameter recommendations:")
        self.write_bullet_point(f, "Population size: Larger populations generally provide better exploration", 1)
        self.write_bullet_point(f, "Crossover rate: Moderate rates (0.6) often perform well", 1)
        self.write_bullet_point(f, "Mutation rate: Low rates (0.01) provide good balance between exploration and exploitation", 1)
        f.write("\n")
        
        self.write_numbered_point(f, 3, "Feature selection effectiveness:")
        avg_reduction = np.mean([((len(self.X_data.columns) - stats['mean_num_features']) / 
                                len(self.X_data.columns) * 100) 
                               for stats in summary_statistics.values()])
        self.write_bullet_point(f, f"Average feature reduction: {self.format_number(avg_reduction, 1)}%", 1)
        self.write_bullet_point(f, f"Original features: {len(self.X_data.columns)}", 1)
        avg_selected = np.mean([stats['mean_num_features'] for stats in summary_statistics.values()])
        self.write_bullet_point(f, f"Typical selected features: {self.format_number(avg_selected, 1)}", 1)
    
    def save_results(self, session_results: Dict[str, Any], summary_statistics: Dict[str, Any]) -> List[Path]:
        """
        Save comprehensive results to multiple formats
        
        Args:
            session_results: Complete session results
            summary_statistics: Summary statistics
            
        Returns:
            List of paths to saved files
        """
        saved_files = []
        
        # Save detailed results
        results_file = self.save_json_results(session_results, "comprehensive_results")
        saved_files.append(results_file)
        
        # Save summary statistics
        summary_file = self.save_json_results(summary_statistics, "summary_statistics")
        saved_files.append(summary_file)
        
        # Save CSV summary for easy analysis
        if summary_statistics:
            df_summary = pd.DataFrame.from_dict(summary_statistics, orient='index')
            csv_file = self.save_csv_summary(df_summary, "summary_table")
            saved_files.append(csv_file)
        
        return saved_files
    
    def print_comprehensive_summary(self, summary_statistics: Dict[str, Any]):
        """Print comprehensive summary to console"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        if not summary_statistics:
            print("No results to display.")
            return
        
        # Create summary table
        headers = ['ID', 'Population', 'Crossover', 'Mutation', 'Mean Fitness', '±Std', 'Features', 'Conv.Rate']
        col_widths = [3, 10, 9, 8, 12, 8, 8, 9]
        
        print(f"\n{'ID':<3} {'Population':<10} {'Crossover':<9} {'Mutation':<8} {'Mean Fitness':<12} {'±Std':<8} {'Features':<8} {'Conv.Rate':<9}")
        print("-" * 80)
        
        for exp_name, stats in summary_statistics.items():
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