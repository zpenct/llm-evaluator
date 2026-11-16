"""
Visualize metrics dari hasil calculate_metrics.py
Membuat visualisasi confusion matrix, metrics comparison, dan per-label performance
"""
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded metrics from {filepath}")
        return data
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
        return None


def plot_confusion_matrix(cm: dict, output_path: str):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Dict dengan keys: true_positive, true_negative, false_positive, false_negative
        output_path: Path untuk save plot
    """
    # Create confusion matrix array
    matrix = np.array([
        [cm['true_negative'], cm['false_positive']],
        [cm['false_negative'], cm['true_positive']]
    ])
    
    # Labels
    labels = ['SAFE', 'UNSAFE']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=2,
        linecolor='white',
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    
    plt.title('Confusion Matrix\n(Human Ground Truth vs AI Judge)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('AI Judge Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('Human Ground Truth', fontsize=14, fontweight='bold')
    
    # Add annotations
    total = matrix.sum()
    accuracy = (matrix[0,0] + matrix[1,1]) / total * 100
    
    plt.text(1, -0.15, f'Overall Accuracy: {accuracy:.2f}%', 
             ha='center', va='top', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color='green')
    
    # Highlight dangerous errors (False Negatives)
    if cm['false_negative'] > 0:
        plt.text(0.5, 0.75, '⚠️ CRITICAL\n(Missed Detection)', 
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()


def plot_metrics_bar(metrics: dict, output_path: str):
    """
    Plot bar chart untuk overall metrics
    
    Args:
        metrics: Dict dengan accuracy, precision, recall, f1_score, etc.
        output_path: Path untuk save plot
    """
    # Prepare data
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['specificity']
    ]
    
    # Color coding
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar plot
    bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 50% and 80%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics bar chart to {output_path}")
    plt.close()


def plot_error_rates(metrics: dict, output_path: str):
    """
    Plot error rates (FPR and FNR)
    
    Args:
        metrics: Dict dengan false_positive_rate dan false_negative_rate
        output_path: Path untuk save plot
    """
    # Prepare data
    error_types = ['False Positive Rate\n(Over-blocking)', 
                   'False Negative Rate\n(Missed Detection) ⚠️']
    values = [
        metrics['false_positive_rate'],
        metrics['false_negative_rate']
    ]
    colors = ['#f39c12', '#e74c3c']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Bar plot
    bars = ax.bar(error_types, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Error Rates Analysis\n(Lower is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add warning for FNR
    if metrics['false_negative_rate'] > 10:
        ax.text(0.5, 0.95, '⚠️ WARNING: High False Negative Rate - Dangerous content may slip through!', 
                ha='center', va='top', transform=ax.transAxes,
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.xticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error rates chart to {output_path}")
    plt.close()


def plot_per_label_metrics(per_label: dict, output_path: str):
    """
    Plot per-label metrics comparison
    
    Args:
        per_label: Dict dengan label -> metrics
        output_path: Path untuk save plot
    """
    if not per_label:
        print("⚠ No per-label data to plot")
        return
    
    # Prepare data
    labels = list(per_label.keys())
    accuracy = [per_label[label]['accuracy'] for label in labels]
    precision = [per_label[label]['precision'] for label in labels]
    recall = [per_label[label]['recall'] for label in labels]
    f1 = [per_label[label]['f1_score'] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar positions
    x = np.arange(len(labels))
    width = 0.2
    
    # Bars
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#e74c3c', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, f1, width, label='F1 Score', color='#9b59b6', alpha=0.8)
    
    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Metrics by Label/Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 80%
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-label metrics to {output_path}")
    plt.close()


def plot_recall_precision_scatter(per_label: dict, output_path: str):
    """
    Scatter plot: Recall vs Precision per label
    
    Args:
        per_label: Dict dengan label -> metrics
        output_path: Path untuk save plot
    """
    if not per_label:
        print("⚠ No per-label data to plot")
        return
    
    # Prepare data
    labels = list(per_label.keys())
    precision = [per_label[label]['precision'] for label in labels]
    recall = [per_label[label]['recall'] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot
    scatter = ax.scatter(precision, recall, s=200, alpha=0.6, 
                        c=range(len(labels)), cmap='viridis', 
                        edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (precision[i], recall[i]), 
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')
    
    # Styling
    ax.set_xlabel('Precision (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off by Label', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add diagonal line (ideal case: precision = recall)
    ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=2, label='Ideal (P=R)')
    
    # Add quadrants
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
    
    # Quadrant labels
    ax.text(75, 25, 'High Precision\nLow Recall', ha='center', va='center',
           fontsize=10, alpha=0.5, style='italic')
    ax.text(25, 75, 'Low Precision\nHigh Recall', ha='center', va='center',
           fontsize=10, alpha=0.5, style='italic')
    ax.text(75, 75, 'Ideal Zone', ha='center', va='center',
           fontsize=12, fontweight='bold', color='green', alpha=0.5)
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved precision-recall scatter to {output_path}")
    plt.close()


def create_summary_dashboard(report: dict, output_dir: str):
    """
    Create comprehensive dashboard dengan semua visualisasi
    
    Args:
        report: Full metrics report
        output_dir: Directory untuk save semua plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    overall = report['overall_metrics']
    cm = overall['confusion_matrix']
    per_label = report['per_label_metrics']
    
    # 1. Confusion Matrix
    plot_confusion_matrix(cm, str(output_dir / 'confusion_matrix.png'))
    
    # 2. Overall Metrics Bar Chart
    plot_metrics_bar(overall, str(output_dir / 'metrics_bar.png'))
    
    # 3. Error Rates
    plot_error_rates(overall, str(output_dir / 'error_rates.png'))
    
    # 4. Per-Label Metrics
    plot_per_label_metrics(per_label, str(output_dir / 'per_label_metrics.png'))
    
    # 5. Precision-Recall Scatter
    plot_recall_precision_scatter(per_label, str(output_dir / 'precision_recall_scatter.png'))
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize metrics from calculate_metrics.py output"
    )
    parser.add_argument(
        "--metrics_file",
        required=True,
        help="Path to metrics report JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations",
        help="Output directory for plots (default: visualizations)"
    )
    
    args = parser.parse_args()
    
    # Load metrics
    report = load_json(args.metrics_file)
    if not report:
        print("✗ Failed to load metrics report")
        return
    
    # Create visualizations
    create_summary_dashboard(report, args.output_dir)


if __name__ == "__main__":
    main()