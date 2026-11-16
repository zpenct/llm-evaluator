"""
Modul untuk visualisasi metrik dan hasil evaluasi dengan human labels comparison
"""
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List

import config
from utils import load_json, logger, ensure_directories, get_timestamp


class MetricsVisualizer:
    """Visualizer untuk metrik evaluasi dengan human labels comparison"""
    
    def __init__(self, metrics_file: str):
        self.metrics = load_json(metrics_file)
        
        if not self.metrics:
            raise ValueError(f"Could not load metrics from {metrics_file}")
        
        self.evaluators = self.metrics.get("evaluators", [])
        self.has_human = self.metrics.get("has_human_labels", False)
        
        logger.info(f"Loaded metrics for {len(self.evaluators)} evaluators")
        if self.has_human:
            logger.info("✓ Human labels available for comparison")
        
        # Setup style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_asr_comparison(self, output_dir: str = None) -> str:
        """Plot ASR comparison bar chart"""
        logger.info("Creating ASR comparison plot...")
        
        asr_data = self.metrics["metrics"]["attack_success_rate"]
        
        # Prepare data
        evaluator_names = []
        normal_asr = []
        long_context_asr = []
        role_play_asr = []
        overall_asr = []
        
        for evaluator, data in asr_data.items():
            evaluator_names.append(evaluator.upper())
            overall_asr.append(data["overall"]["asr_percentage"])
            
            by_type = data["by_prompt_type"]
            normal_asr.append(by_type.get("normal", {}).get("asr_percentage", 0))
            long_context_asr.append(by_type.get("long_context", {}).get("asr_percentage", 0))
            role_play_asr.append(by_type.get("role_play", {}).get("asr_percentage", 0))
        
        # Create grouped bar chart
        x = np.arange(len(evaluator_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width*1.5, normal_asr, width, label='Normal', color='#3498db')
        bars2 = ax.bar(x - width*0.5, long_context_asr, width, label='Long Context', color='#e74c3c')
        bars3 = ax.bar(x + width*0.5, role_play_asr, width, label='Role Play', color='#f39c12')
        bars4 = ax.bar(x + width*1.5, overall_asr, width, label='Overall', color='#2ecc71')
        
        # Styling
        ax.set_xlabel('Evaluator', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Attack Success Rate Comparison by Evaluator and Prompt Type', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(evaluator_names)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "asr_comparison", output_dir)
        plt.close()
        
        return output_path
    
    def plot_accuracy_vs_human(self, output_dir: str = None) -> str:
        """Plot accuracy comparison dengan human labels"""
        if not self.has_human:
            logger.warning("No human labels available for accuracy plot")
            return None
        
        logger.info("Creating accuracy vs human plot...")
        
        acc_data = self.metrics["metrics"]["accuracy_vs_human"]
        
        # Prepare data
        evaluators = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for evaluator, data in acc_data.items():
            overall = data["overall"]
            evaluators.append(evaluator.upper())
            accuracy_scores.append(overall["accuracy"])
            precision_scores.append(overall["precision"])
            recall_scores.append(overall["recall"])
            f1_scores.append(overall["f1_score"])
        
        # Add majority voting if available
        if "majority_voting" in self.metrics["metrics"]:
            mv_data = self.metrics["metrics"]["majority_voting"]["vs_human"]
            evaluators.append("MAJORITY")
            accuracy_scores.append(mv_data["accuracy"])
            precision_scores.append(mv_data["precision"])
            recall_scores.append(mv_data["recall"])
            f1_scores.append(mv_data["f1_score"])
        
        # Create grouped bar chart
        x = np.arange(len(evaluators))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width*1.5, accuracy_scores, width, label='Accuracy', color='#3498db')
        bars2 = ax.bar(x - width*0.5, precision_scores, width, label='Precision', color='#e74c3c')
        bars3 = ax.bar(x + width*0.5, recall_scores, width, label='Recall', color='#f39c12')
        bars4 = ax.bar(x + width*1.5, f1_scores, width, label='F1-Score', color='#9b59b6')
        
        # Styling
        ax.set_xlabel('Evaluator', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('AI Judge Performance vs Human Labels', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(evaluators)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "accuracy_vs_human", output_dir)
        plt.close()
        
        return output_path
    
    def plot_confusion_matrices(self, output_dir: str = None) -> List[str]:
        """Plot confusion matrix heatmap untuk setiap evaluator vs human"""
        if not self.has_human:
            logger.warning("No human labels available for confusion matrices")
            return []
        
        logger.info("Creating confusion matrix heatmaps...")
        
        acc_data = self.metrics["metrics"]["accuracy_vs_human"]
        output_paths = []
        
        for evaluator, data in acc_data.items():
            cm = data["overall"]["confusion_matrix"]
            
            # Create confusion matrix array
            matrix = np.array([
                [cm["tn"], cm["fp"]],
                [cm["fn"], cm["tp"]]
            ])
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted SAFE', 'Predicted UNSAFE'],
                       yticklabels=['Human SAFE', 'Human UNSAFE'],
                       ax=ax, cbar_kws={'label': 'Count'})
            
            # Add metrics as text
            metrics = data["overall"]
            metrics_text = (
                f'Accuracy: {metrics["accuracy"]:.2f}%\n'
                f'Precision: {metrics["precision"]:.2f}%\n'
                f'Recall: {metrics["recall"]:.2f}%\n'
                f'F1-Score: {metrics["f1_score"]:.2f}\n'
                f'FPR: {metrics["false_positive_rate"]:.2f}%\n'
                f'FNR: {metrics["false_negative_rate"]:.2f}%'
            )
            
            ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'Confusion Matrix: {evaluator.upper()} vs Human',
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            output_path = self._save_plot(fig, f"confusion_matrix_{evaluator}", output_dir)
            output_paths.append(output_path)
            plt.close()
        
        return output_paths
    
    def plot_error_rates_by_prompt_type(self, output_dir: str = None) -> str:
        """Plot false positive & false negative rates by prompt type"""
        if not self.has_human:
            logger.warning("No human labels available for error rate plot")
            return None
        
        logger.info("Creating error rates by prompt type plot...")
        
        acc_data = self.metrics["metrics"]["accuracy_vs_human"]
        
        # Prepare data
        prompt_types = config.PROMPT_TYPES
        evaluators = list(acc_data.keys())
        
        # Calculate FPR and FNR for each evaluator and prompt type
        fpr_data = {eval: [] for eval in evaluators}
        fnr_data = {eval: [] for eval in evaluators}
        
        for evaluator in evaluators:
            for pt in prompt_types:
                pt_data = acc_data[evaluator]["by_prompt_type"].get(pt, {})
                cm = pt_data.get("confusion_matrix", {"fp": 0, "tn": 0, "fn": 0, "tp": 0})
                
                fp = cm.get("fp", 0)
                tn = cm.get("tn", 0)
                fn = cm.get("fn", 0)
                tp = cm.get("tp", 0)
                
                fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
                fnr = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0
                
                fpr_data[evaluator].append(fpr)
                fnr_data[evaluator].append(fnr)
        
        # Create grouped bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(prompt_types))
        width = 0.25
        
        # False Positive Rate
        for i, evaluator in enumerate(evaluators):
            offset = width * (i - len(evaluators)/2 + 0.5)
            ax1.bar(x + offset, fpr_data[evaluator], width, label=evaluator.upper())
        
        ax1.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('False Positive Rate by Prompt Type\n(Over-defensive)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # False Negative Rate
        for i, evaluator in enumerate(evaluators):
            offset = width * (i - len(evaluators)/2 + 0.5)
            ax2.bar(x + offset, fnr_data[evaluator], width, label=evaluator.upper())
        
        ax2.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('False Negative Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('False Negative Rate by Prompt Type\n(Missed Detection - Critical!)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "error_rates_by_prompt_type", output_dir)
        plt.close()
        
        return output_path
    
    def plot_category_performance_heatmap(self, output_dir: str = None) -> str:
        """Plot performance heatmap per category"""
        if not self.has_human:
            logger.warning("No human labels available for category performance")
            return None
        
        logger.info("Creating category performance heatmap...")
        
        acc_data = self.metrics["metrics"]["accuracy_vs_human"]
        
        # Get all categories
        all_categories = set()
        for data in acc_data.values():
            all_categories.update(data["by_label"].keys())
        
        categories = sorted(list(all_categories))
        evaluators = list(acc_data.keys())
        
        # Build matrix: evaluators x categories
        matrix = []
        for evaluator in evaluators:
            row = []
            for category in categories:
                cat_data = acc_data[evaluator]["by_label"].get(category, {})
                accuracy = cat_data.get("accuracy", 0) / 100  # Convert to 0-1 scale
                row.append(accuracy)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[c.replace('_', ' ').title() for c in categories],
                   yticklabels=[e.upper() for e in evaluators],
                   vmin=0, vmax=1, ax=ax,
                   cbar_kws={'label': 'Accuracy (0-1)'})
        
        ax.set_title('Accuracy by Category (vs Human Labels)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Evaluator', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "category_performance_heatmap", output_dir)
        plt.close()
        
        return output_path
    
    def plot_cohens_kappa_comparison(self, output_dir: str = None) -> str:
        """Plot Cohen's Kappa comparison dengan human"""
        if not self.has_human:
            logger.warning("No human labels available for Cohen's Kappa")
            return None
        
        logger.info("Creating Cohen's Kappa comparison plot...")
        
        kappa_data = self.metrics["metrics"].get("cohens_kappa_vs_human", {})
        
        if not kappa_data:
            return None
        
        evaluators = list(kappa_data.keys())
        kappa_values = list(kappa_data.values())
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if k > 0.8 else '#f39c12' if k > 0.6 else '#e74c3c' 
                 for k in kappa_values]
        
        bars = ax.bar([e.upper() for e in evaluators], kappa_values, color=colors, alpha=0.7)
        
        # Add interpretation lines
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, 
                  label='Almost Perfect (>0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, 
                  label='Substantial (>0.6)')
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, 
                  label='Moderate (>0.4)')
        
        ax.set_xlabel('Evaluator', fontsize=12, fontweight='bold')
        ax.set_ylabel("Cohen's Kappa", fontsize=12, fontweight='bold')
        ax.set_title("Inter-Rater Reliability: AI Judges vs Human",
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "cohens_kappa_vs_human", output_dir)
        plt.close()
        
        return output_path
    
    def plot_asr_by_category(self, output_dir: str = None) -> str:
        """Plot ASR by category"""
        logger.info("Creating ASR by category plot...")
        
        asr_data = self.metrics["metrics"]["attack_success_rate"]
        
        # Collect categories
        all_categories = set()
        for data in asr_data.values():
            all_categories.update(data["by_label"].keys())
        
        categories = sorted(list(all_categories))
        
        # Prepare data
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(categories))
        width = 0.8 / len(self.evaluators)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.evaluators)))
        
        for i, evaluator in enumerate(self.evaluators):
            evaluator_asr = []
            for category in categories:
                asr = asr_data[evaluator]["by_label"].get(category, {}).get("asr_percentage", 0)
                evaluator_asr.append(asr)
            
            ax.bar(x + i*width, evaluator_asr, width, 
                  label=evaluator.upper(), color=colors[i])
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Attack Success Rate by Category', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(self.evaluators) - 1) / 2)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self._save_plot(fig, "asr_by_category", output_dir)
        plt.close()
        
        return output_path
    
    def _save_plot(self, fig, name: str, output_dir: str = None) -> str:
        """Helper untuk save plot"""
        if output_dir is None:
            output_dir = config.REPORTS_DIR
        
        ensure_directories(output_dir)
        
        timestamp = get_timestamp()
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath}")
        
        return filepath
    
    def generate_all_plots(self, output_dir: str = None) -> Dict[str, Any]:
        """Generate semua visualisasi"""
        logger.info("Generating all visualizations...")
        
        results = {
            "timestamp": get_timestamp(),
            "plots": {}
        }
        
        # ASR Comparison
        results["plots"]["asr_comparison"] = self.plot_asr_comparison(output_dir)
        
        # ASR by Category
        results["plots"]["asr_by_category"] = self.plot_asr_by_category(output_dir)
        
        # If human labels available
        if self.has_human:
            # Accuracy vs Human
            results["plots"]["accuracy_vs_human"] = self.plot_accuracy_vs_human(output_dir)
            
            # Confusion Matrices
            confusion_plots = self.plot_confusion_matrices(output_dir)
            results["plots"]["confusion_matrices"] = confusion_plots
            
            # Error Rates
            error_plot = self.plot_error_rates_by_prompt_type(output_dir)
            if error_plot:
                results["plots"]["error_rates"] = error_plot
            
            # Category Performance
            cat_plot = self.plot_category_performance_heatmap(output_dir)
            if cat_plot:
                results["plots"]["category_performance"] = cat_plot
            
            # Cohen's Kappa
            kappa_plot = self.plot_cohens_kappa_comparison(output_dir)
            if kappa_plot:
                results["plots"]["cohens_kappa"] = kappa_plot
        
        logger.info("All visualizations generated successfully")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from metrics report"
    )
    parser.add_argument(
        "metrics_file",
        help="Path to metrics report JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory for plots (default: {config.REPORTS_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        visualizer = MetricsVisualizer(args.metrics_file)
        results = visualizer.generate_all_plots(args.output_dir)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Generated plots:")
        for plot_name, plot_path in results["plots"].items():
            if isinstance(plot_path, list):
                print(f"  {plot_name}: {len(plot_path)} files")
                for p in plot_path:
                    print(f"    - {p}")
            elif plot_path:
                print(f"  {plot_name}: {plot_path}")
        
        if visualizer.has_human:
            print("\n✓ Human labels comparison included!")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()