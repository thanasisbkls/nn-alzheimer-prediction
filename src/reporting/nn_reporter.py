#!/usr/bin/env python3
"""
Neural Network Reporter Module

Reporting functions for Neural Network comparison results.
Extracts all reporting functionality from the neural network comparison module.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_reporter import BaseReporter


class NNReporter(BaseReporter):
    """
    Reporter for Neural Network comparison results
    """
    
    def __init__(self, results_dir: Path, X_data: pd.DataFrame):
        """
        Initialize NN reporter
        
        Args:
            results_dir: Directory to save reports
            X_data: Original dataset for analysis context
        """
        super().__init__(results_dir)
        self.X_data = X_data
    
    def generate_report(self, comparison: Dict[str, Any], 
                       generalizability: Dict[str, Any],
                       selected_features: List[str],
                       full_retrain_results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Generate comprehensive neural network comparison report
        
        Args:
            comparison: Model comparison results
            generalizability: Generalizability analysis results
            selected_features: List of GA-selected features
            full_retrain_results: Optional full retrain results
            
        Returns:
            Path to generated report
        """
        report_path = self.results_dir / 'neural_network_comparison_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Main header
            f.write("NEURAL NETWORK COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive summary
            self._write_executive_summary(f, comparison, selected_features)
            
            # Selected features section
            self._write_selected_features(f, selected_features)
            
            # Performance comparison
            self._write_performance_comparison(f, comparison)
            
            # Analysis of requirements
            self._write_analysis_sections(f, comparison, generalizability, full_retrain_results)
            
            # Final recommendations
            self._write_final_recommendations(f, comparison, generalizability)
        
        return report_path
    
    def _write_executive_summary(self, f, comparison: Dict[str, Any], selected_features: List[str]):
        """Write executive summary section"""
        self.write_section_header(f, "EXECUTIVE SUMMARY", level=2)
        
        f.write(f"Dataset: {self.X_data.shape[0]} samples, {self.X_data.shape[1]} original features\n")
        f.write(f"GA-selected features: {len(selected_features)} features\n")
        f.write(f"Feature reduction: {self.format_percentage(comparison['performance_difference']['feature_reduction'])}\n\n")
    
    def _write_selected_features(self, f, selected_features: List[str]):
        """Write selected features section"""
        self.write_section_header(f, "SELECTED FEATURES BY GA", level=2)
        
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feature}\n")
        f.write("\n")
    
    def _write_performance_comparison(self, f, comparison: Dict[str, Any]):
        """Write performance comparison section"""
        self.write_section_header(f, "PERFORMANCE COMPARISON", level=2)
        
        f.write(f"GA Model (Selected Features):\n")
        f.write(f"  - Test Accuracy: {self.format_number(comparison['ga_model']['test_accuracy'])}\n")
        f.write(f"  - Test Loss: {self.format_number(comparison['ga_model']['test_loss'])}\n")
        f.write(f"  - Test MSE: {self.format_number(comparison['ga_model']['test_mse'])}\n")
        f.write(f"  - Features Used: {comparison['ga_model']['num_features']}\n\n")
        
        f.write(f"Reference Model (All Features):\n")
        f.write(f"  - Test Accuracy: {self.format_number(comparison['reference_model']['test_accuracy'])}\n")
        f.write(f"  - Test Loss: {self.format_number(comparison['reference_model']['test_loss'])}\n")
        f.write(f"  - Test MSE: {self.format_number(comparison['reference_model']['test_mse'])}\n")
        f.write(f"  - Features Used: {comparison['reference_model']['num_features']}\n\n")
        
        f.write(f"Performance Differences (GA - Reference):\n")
        f.write(f"  - Accuracy Difference: {comparison['performance_difference']['accuracy_diff']:+.4f}\n")
        f.write(f"  - Loss Difference: {comparison['performance_difference']['loss_diff']:+.4f}\n")
        f.write(f"  - MSE Difference: {comparison['performance_difference']['mse_diff']:+.4f}\n\n")
    
    def _write_analysis_sections(self, f, comparison: Dict[str, Any], 
                                generalizability: Dict[str, Any], 
                                full_retrain_results: Optional[Dict[str, Any]]):
        """Write detailed analysis sections"""
        self.write_section_header(f, "ANALYSIS OF REQUIREMENTS", level=1)
        
        # i. Generalizability analysis
        self._write_generalizability_analysis(f, generalizability)
        
        # ii. Effect of feature selection
        self._write_feature_selection_analysis(f, comparison)
        
        # iii. Overfitting analysis
        self._write_overfitting_analysis(f, generalizability)
        
        # Full dataset retraining analysis (if available)
        if full_retrain_results:
            self._write_weight_transfer_analysis(f, comparison, full_retrain_results)
    
    def _write_generalizability_analysis(self, f, generalizability: Dict[str, Any]):
        """Write generalizability comparison section"""
        self.write_section_header(f, "i. GENERALIZABILITY COMPARISON", level=2)
        
        ga_gap = generalizability['generalization_gaps']['ga_model']['val_test_gap']
        ref_gap = generalizability['generalization_gaps']['reference_model']['val_test_gap']
        
        f.write(f"Training-Test Performance Gaps:\n")
        f.write(f"  - GA Model: {self.format_number(ga_gap)} (Train: {self.format_number(generalizability['generalization_gaps']['ga_model']['train_accuracy_at_best'])}, Test: {self.format_number(generalizability['generalization_gaps']['ga_model']['test_accuracy'])})\n")
        f.write(f"  - Reference Model: {self.format_number(ref_gap)} (Train: {self.format_number(generalizability['generalization_gaps']['reference_model']['train_accuracy_at_best'])}, Test: {self.format_number(generalizability['generalization_gaps']['reference_model']['test_accuracy'])})\n\n")
        
        if abs(ga_gap) < abs(ref_gap):
            f.write("CONCLUSION: GA model shows BETTER generalizability (smaller train-test gap)\n")
        elif abs(ga_gap) > abs(ref_gap):
            f.write("CONCLUSION: Reference model shows BETTER generalizability (smaller train-test gap)\n")
        else:
            f.write("CONCLUSION: Both models show SIMILAR generalizability\n")
        
        f.write(f"Gap difference: {abs(ga_gap) - abs(ref_gap):+.4f}\n\n")
    
    def _write_feature_selection_analysis(self, f, comparison: Dict[str, Any]):
        """Write feature selection effect analysis"""
        self.write_section_header(f, "ii. EFFECT OF FEATURE SELECTION ON NN PERFORMANCE", level=2)
        
        feature_reduction_pct = comparison['performance_difference']['feature_reduction'] * 100
        accuracy_change = comparison['performance_difference']['accuracy_diff']
        
        f.write(f"Feature Reduction: {feature_reduction_pct:.1f}% ({comparison['reference_model']['num_features']} → {comparison['ga_model']['num_features']} features)\n")
        f.write(f"Accuracy Change: {accuracy_change:+.4f}\n")
        f.write(f"Loss Change: {comparison['performance_difference']['loss_diff']:+.4f}\n\n")
        
        if accuracy_change > 0:
            f.write("CONCLUSION: Feature selection IMPROVED neural network performance\n")
            f.write("This suggests that the removed features contained noise or were redundant.\n")
        elif accuracy_change < -0.01:  # Significant decrease
            f.write("CONCLUSION: Feature selection DECREASED neural network performance\n")
            f.write("This suggests that some important information was lost in feature selection.\n")
        else:
            f.write("CONCLUSION: Feature selection had MINIMAL IMPACT on neural network performance\n")
            f.write("This suggests effective feature selection without significant information loss.\n")
        
        f.write(f"Performance per feature ratio: GA={comparison['ga_model']['test_accuracy']/comparison['ga_model']['num_features']:.6f}, ")
        f.write(f"Reference={comparison['reference_model']['test_accuracy']/comparison['reference_model']['num_features']:.6f}\n\n")
    
    def _write_overfitting_analysis(self, f, generalizability: Dict[str, Any]):
        """Write overfitting analysis section"""
        self.write_section_header(f, "iii. OVERFITTING ANALYSIS", level=2)
        
        ga_overfitting = generalizability['overfitting_indicators']['ga_model']
        ref_overfitting = generalizability['overfitting_indicators']['reference_model']
        
        f.write(f"Overfitting Indicators:\n")
        f.write(f"GA Model:\n")
        f.write(f"  - Train-Val Loss Difference: {self.format_number(ga_overfitting['final_train_val_loss_diff'])}\n")
        f.write(f"  - Max Validation Loss Increase: {self.format_number(ga_overfitting['max_val_loss_increase'])}\n")
        f.write(f"  - Early Stopping Triggered: {ga_overfitting['early_stopped']}\n\n")
        
        f.write(f"Reference Model:\n")
        f.write(f"  - Train-Val Loss Difference: {self.format_number(ref_overfitting['final_train_val_loss_diff'])}\n")
        f.write(f"  - Max Validation Loss Increase: {self.format_number(ref_overfitting['max_val_loss_increase'])}\n")
        f.write(f"  - Early Stopping Triggered: {ref_overfitting['early_stopped']}\n\n")
        
        # Determine which model shows more overfitting
        ga_overfitting_score = abs(ga_overfitting['final_train_val_loss_diff']) + ga_overfitting['max_val_loss_increase']
        ref_overfitting_score = abs(ref_overfitting['final_train_val_loss_diff']) + ref_overfitting['max_val_loss_increase']
        
        if ga_overfitting_score < ref_overfitting_score:
            f.write("CONCLUSION: GA model shows LESS overfitting than reference model\n")
            f.write("Feature selection appears to have reduced overfitting by removing noisy features.\n")
        elif ga_overfitting_score > ref_overfitting_score:
            f.write("CONCLUSION: GA model shows MORE overfitting than reference model\n")
            f.write("This could indicate that the selected features are too specific to the training data.\n")
        else:
            f.write("CONCLUSION: Both models show SIMILAR levels of overfitting\n")
        
        f.write(f"Overfitting score difference: {ga_overfitting_score - ref_overfitting_score:+.4f}\n\n")
    
    def _write_weight_transfer_analysis(self, f, comparison: Dict[str, Any], 
                                      full_retrain_results: Dict[str, Any]):
        """Write weight transfer retraining analysis"""
        self.write_section_header(f, "WEIGHT TRANSFER RETRAINING ANALYSIS", level=2)
        
        original_ga_acc = comparison['ga_model']['test_accuracy']
        original_ref_acc = comparison['reference_model']['test_accuracy']
        expanded_ga_acc = full_retrain_results['full_comparison']['ga_model']['test_accuracy']
        full_ref_acc = full_retrain_results['full_comparison']['reference_model']['test_accuracy']
        
        weight_transfer_info = full_retrain_results['weight_transfer_summary']
        
        f.write(f"Weight Transfer Summary:\n")
        f.write(f"  - Original GA features: {weight_transfer_info['selected_feature_count']}\n")
        f.write(f"  - Total dataset features: {weight_transfer_info['total_feature_count']}\n")
        f.write(f"  - Processed selected features: {weight_transfer_info['processed_selected_features']}\n")
        f.write(f"  - Processed total features: {weight_transfer_info['processed_total_features']}\n")
        f.write(f"  - Weights transferred: {weight_transfer_info['weights_transferred']}\n\n")
        
        f.write(f"Performance with GA-selected features:\n")
        f.write(f"  - GA Model: {self.format_number(original_ga_acc)}\n")
        f.write(f"  - Reference Model: {self.format_number(original_ref_acc)}\n")
        f.write(f"  - Difference: {original_ga_acc - original_ref_acc:+.4f}\n\n")
        
        f.write(f"Performance with weight-transferred full features:\n")
        f.write(f"  - GA Expanded Model: {self.format_number(expanded_ga_acc)}\n")
        f.write(f"  - Reference Model: {self.format_number(full_ref_acc)}\n")
        f.write(f"  - Difference: {expanded_ga_acc - full_ref_acc:+.4f}\n\n")
        
        ga_change = expanded_ga_acc - original_ga_acc
        ref_change = full_ref_acc - original_ref_acc
        
        f.write(f"Effect of expanding to full feature set:\n")
        f.write(f"  - GA Model change: {ga_change:+.4f}\n")
        f.write(f"  - Reference Model change: {ref_change:+.4f}\n\n")
        
        # Analyze the effect of weight transfer vs random initialization
        original_advantage = original_ga_acc - original_ref_acc
        expanded_advantage = expanded_ga_acc - full_ref_acc
        
        f.write(f"GA Model Advantage Analysis:\n")
        f.write(f"  - Original advantage (selected features): {original_advantage:+.4f}\n")
        f.write(f"  - Expanded advantage (all features): {expanded_advantage:+.4f}\n")
        f.write(f"  - Advantage change: {expanded_advantage - original_advantage:+.4f}\n\n")
        
        if expanded_advantage > original_advantage:
            f.write("CONCLUSION: Weight transfer IMPROVED GA model's advantage\n")
            f.write("The transferred weights provided a good initialization for the full feature set.\n")
        elif expanded_advantage < original_advantage - 0.01:  # Significant decrease
            f.write("CONCLUSION: Weight transfer REDUCED GA model's advantage\n")
            f.write("The additional features may have introduced noise or the weight transfer was suboptimal.\n")
        else:
            f.write("CONCLUSION: Weight transfer MAINTAINED GA model's advantage\n")
            f.write("The expanded model performs similarly to the original, suggesting effective weight transfer.\n")
        
        # Generalizability analysis for expanded model
        if 'full_generalizability' in full_retrain_results:
            expanded_gen = full_retrain_results['full_generalizability']['generalization_gaps']['ga_model']
            ref_gen = full_retrain_results['full_generalizability']['generalization_gaps']['reference_model']
            
            f.write(f"\nExpanded Model Generalizability:\n")
            f.write(f"  - GA Expanded val-test gap: {self.format_number(expanded_gen['val_test_gap'])}\n")
            f.write(f"  - Reference val-test gap: {self.format_number(ref_gen['val_test_gap'])}\n")
            
            if abs(expanded_gen['val_test_gap']) < abs(ref_gen['val_test_gap']):
                f.write("  - GA expanded model shows better generalizability\n")
            else:
                f.write("  - Reference model shows better generalizability\n")
    
    def _write_final_recommendations(self, f, comparison: Dict[str, Any], 
                                   generalizability: Dict[str, Any]):
        """Write final recommendations section"""
        self.write_section_header(f, "FINAL RECOMMENDATIONS", level=2)
        
        if comparison['performance_difference']['accuracy_diff'] > 0:
            f.write("1. USE GA-SELECTED FEATURES: They provide better performance with fewer features\n")
        else:
            f.write("1. CONSIDER ALL FEATURES: Full feature set provides better performance\n")
        
        ga_gap = generalizability['generalization_gaps']['ga_model']['val_test_gap']
        ref_gap = generalizability['generalization_gaps']['reference_model']['val_test_gap']
        
        if abs(ga_gap) < abs(ref_gap):
            f.write("2. GA model shows better generalizability\n")
        else:
            f.write("2. Reference model shows better generalizability\n")
        
        feature_reduction_pct = comparison['performance_difference']['feature_reduction'] * 100
        f.write(f"3. Feature selection achieved {feature_reduction_pct:.1f}% dimensionality reduction\n")
        f.write(f"4. Consider the trade-off between model complexity and performance\n")
    
    def save_results(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                    selected_features: List[str], full_retrain_results: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Save all results to JSON and CSV files
        
        Args:
            comparison: Model comparison results
            generalizability: Generalizability analysis results
            selected_features: GA-selected features
            full_retrain_results: Optional full retrain results
            
        Returns:
            List of paths to saved files
        """
        saved_files = []
        
        # Compile all results
        all_results = {
            'dataset_info': {
                'total_samples': len(self.X_data),
                'total_features': len(self.X_data.columns),
                'selected_features': selected_features,
                'feature_reduction_pct': comparison['performance_difference']['feature_reduction'] * 100
            },
            'performance_comparison': comparison,
            'generalizability_analysis': generalizability,
            'full_retrain_results': full_retrain_results
        }
        
        # Save detailed results
        results_file = self.save_json_results(all_results, "nn_comparison_results")
        saved_files.append(results_file)
        
        # Save summary CSV
        summary_data = {
            'Model': ['GA_Selected_Features', 'All_Features'],
            'Test_Accuracy': [comparison['ga_model']['test_accuracy'], comparison['reference_model']['test_accuracy']],
            'Test_Loss': [comparison['ga_model']['test_loss'], comparison['reference_model']['test_loss']],
            'Num_Features': [comparison['ga_model']['num_features'], comparison['reference_model']['num_features']],
            'Generalization_Gap': [
                generalizability['generalization_gaps']['ga_model']['val_test_gap'],
                generalizability['generalization_gaps']['reference_model']['val_test_gap']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = self.save_csv_summary(summary_df, "nn_comparison_summary")
        saved_files.append(csv_file)
        
        return saved_files
    
    def print_analysis_summary(self, comparison: Dict[str, Any], generalizability: Dict[str, Any],
                              selected_features: List[str], full_retrain_results: Optional[Dict[str, Any]]):
        """Print comprehensive analysis summary to console"""
        
        print("\n" + "="*80)
        print("NEURAL NETWORK COMPARISON ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDATASET INFORMATION:")
        print(f"  Total samples: {len(self.X_data)}")
        print(f"  Original features: {len(self.X_data.columns)}")
        print(f"  GA-selected features: {len(selected_features)}")
        print(f"  Feature reduction: {self.format_percentage(comparison['performance_difference']['feature_reduction'])}")
        
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"  {'Model':<20} {'Accuracy':<10} {'Loss':<10} {'Features':<10}")
        print(f"  {'-'*50}")
        print(f"  {'GA Model':<20} {comparison['ga_model']['test_accuracy']:<10.4f} {comparison['ga_model']['test_loss']:<10.4f} {comparison['ga_model']['num_features']:<10}")
        print(f"  {'Reference Model':<20} {comparison['reference_model']['test_accuracy']:<10.4f} {comparison['reference_model']['test_loss']:<10.4f} {comparison['reference_model']['num_features']:<10}")
        
        print(f"\nKEY FINDINGS:")
        
        # Generalizability
        ga_gap = abs(generalizability['generalization_gaps']['ga_model']['val_test_gap'])
        ref_gap = abs(generalizability['generalization_gaps']['reference_model']['val_test_gap'])
        if ga_gap < ref_gap:
            print(f"  ✓ GA model shows better generalizability (gap: {ga_gap:.4f} vs {ref_gap:.4f})")
        else:
            print(f"  ✗ Reference model shows better generalizability (gap: {ref_gap:.4f} vs {ga_gap:.4f})")
        
        # Performance
        acc_diff = comparison['performance_difference']['accuracy_diff']
        if acc_diff > 0:
            print(f"  ✓ GA model achieves better accuracy (+{acc_diff:.4f})")
        else:
            print(f"  ✗ Reference model achieves better accuracy ({acc_diff:.4f})")
        
        # Feature efficiency
        ga_efficiency = comparison['ga_model']['test_accuracy'] / comparison['ga_model']['num_features']
        ref_efficiency = comparison['reference_model']['test_accuracy'] / comparison['reference_model']['num_features']
        if ga_efficiency > ref_efficiency:
            print(f"  ✓ GA model is more feature-efficient ({ga_efficiency:.6f} vs {ref_efficiency:.6f})")
        else:
            print(f"  ✗ Reference model is more feature-efficient ({ref_efficiency:.6f} vs {ga_efficiency:.6f})")
        
        # Full dataset training
        if full_retrain_results:
            original_diff = comparison['performance_difference']['accuracy_diff']
            full_diff = full_retrain_results['full_comparison']['performance_difference']['accuracy_diff']
            weight_transfer_info = full_retrain_results['weight_transfer_summary']
            
            print(f"  • Weight transfer experiment:")
            print(f"    - Original GA advantage: {original_diff:+.4f}")
            print(f"    - Expanded GA advantage: {full_diff:+.4f}")
            print(f"    - Features: {weight_transfer_info['processed_selected_features']} → {weight_transfer_info['processed_total_features']}")
            
            if full_diff > original_diff:
                print(f"    ✓ Weight transfer improved GA model performance")
            elif full_diff < original_diff - 0.01:
                print(f"    ✗ Weight transfer reduced GA model performance")
            else:
                print(f"    ≈ Weight transfer maintained GA model performance")
        
        print(f"\nOUTPUT FILES:")
        print(f"  • Comprehensive report: {self.results_dir}/neural_network_comparison_report.txt")
        print(f"  • Detailed results: {self.results_dir}/nn_comparison_results_*.json")
        print(f"  • Summary CSV: {self.results_dir}/nn_comparison_summary_*.csv")
        print("="*80) 