"""
Report generation module for creating model evaluation reports and visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay

from settings import FIGURE_SIZE_COMPARISON, FIGURE_SIZE_SINGLE, OUTPUT_DIR, PLOT_DPI


class ReportGenerator:
    """Generates comprehensive reports and visualizations for model evaluation."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory for saving reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            model_name: Name of the model
            labels: Class labels
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        if labels is None:
            labels = ["No Goal", "Goal"]

        fig, ax = plt.subplots(figsize=(8, 6))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")

        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot ROC curve for a single model.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: AUC score
            model_name: Name of the model
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

        ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve: {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{model_name}_roc_curve.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_roc_curves_comparison(
        self,
        models_data: Dict[str, Dict],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot ROC curves for multiple models on the same plot.

        Args:
            models_data: Dict mapping model names to dicts with 'fpr', 'tpr', 'auc'
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

        for model_name, data in models_data.items():
            fpr = data["fpr"]
            tpr = data["tpr"]
            auc = data["auc"]
            ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "roc_curves_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot calibration curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins for calibration
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=model_name)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(f"Calibration Curve: {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{model_name}_calibration_curve.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_logistic_curve(
        self,
        model: Any,
        feature_name: str,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot logistic regression curve showing the fitted model.

        Args:
            model: Trained model
            feature_name: Name of the feature being plotted
            X: Feature values (1D array)
            y: True labels
            model_name: Name of the model
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        import pandas as pd

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

        # Create range for smooth curve
        x_min, x_max = X.min(), X.max()
        x_range = np.linspace(x_min, x_max, 300)

        # Predict probabilities for the range
        X_range_df = pd.DataFrame({feature_name: x_range})
        y_pred = model.predict(X_range_df)

        # Plot the logistic curve
        ax.plot(x_range, y_pred, "b-", linewidth=2, label="Logistic Curve")

        # Calculate binned actual rates for overlay
        n_bins = 20
        bins = np.linspace(x_min, x_max, n_bins + 1)
        bin_indices = np.digitize(X, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_centers = []
        actual_rates = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 5:  # Only show bins with enough samples
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                actual_rates.append(y[mask].mean())

        # Plot actual rates as scatter points
        ax.scatter(
            bin_centers,
            actual_rates,
            color="red",
            s=50,
            alpha=0.6,
            label="Actual Goal Rate (binned)",
            zorder=5,
        )

        ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Goal Probability", fontsize=12)
        ax.set_title(
            f"Logistic Curve: {model_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{model_name}_logistic_curve.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_feature_vs_probability(
        self,
        model: Any,
        feature_name: str,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot feature values against predicted goal probabilities with actual outcomes.

        Args:
            model: Trained model
            feature_name: Name of the feature being plotted
            X: Feature values (1D array)
            y: True labels (0 or 1)
            model_name: Name of the model
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        import pandas as pd

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

        # Get predictions
        X_df = pd.DataFrame({feature_name: X})
        y_pred = model.predict(X_df)

        # Separate goals and non-goals
        goals_mask = y == 1
        non_goals_mask = y == 0

        # Plot scatter points with jitter for better visibility
        jitter = 0.02
        y_jitter_goals = y[goals_mask] + np.random.uniform(
            -jitter, jitter, goals_mask.sum()
        )
        y_jitter_non_goals = y[non_goals_mask] + np.random.uniform(
            -jitter, jitter, non_goals_mask.sum()
        )

        ax.scatter(
            X[goals_mask],
            y_jitter_goals,
            c="green",
            alpha=0.3,
            s=20,
            label="Goals",
            marker="^",
        )
        ax.scatter(
            X[non_goals_mask],
            y_jitter_non_goals,
            c="red",
            alpha=0.2,
            s=20,
            label="Non-Goals",
            marker="v",
        )

        # Plot predicted probabilities as a line
        sorted_idx = np.argsort(X)
        ax.plot(
            X[sorted_idx],
            y_pred[sorted_idx],
            "b-",
            linewidth=2,
            label="Predicted Probability",
            alpha=0.8,
        )

        ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Goal Probability / Outcome", fontsize=12)
        ax.set_title(
            f"{feature_name.replace('_', ' ').title()} vs Goal Probability: {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()

        if save_path is None:
            save_path = (
                self.output_dir / f"{model_name}_feature_vs_probability.png"
            )
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def plot_metrics_comparison(
        self,
        models_metrics: Dict[str, Dict],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot bar chart comparing metrics across models.

        Args:
            models_metrics: Dict mapping model names to metrics dicts
            save_path: Path to save the plot (if None, auto-generated)

        Returns:
            Path where plot was saved
        """
        # Prepare data for plotting
        metrics_names = list(next(iter(models_metrics.values())).keys())
        model_names = list(models_metrics.keys())

        n_models = len(model_names)

        fig, axes = plt.subplots(
            2, 3, figsize=FIGURE_SIZE_COMPARISON
        )  # 2 rows, 3 columns
        axes = axes.flatten()

        for i, metric_name in enumerate(metrics_names):
            if i >= len(axes):
                break

            ax = axes[i]
            values = [models_metrics[model][metric_name] for model in model_names]

            bars = ax.bar(range(n_models), values, color="steelblue", alpha=0.7)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=10)
            ax.set_title(metric_name.replace("_", " ").title(), fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Hide unused subplots
        for i in range(len(metrics_names), len(axes)):
            axes[i].axis("off")

        plt.suptitle("Metrics Comparison Across Models", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "metrics_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        return str(save_path)

    def generate_single_model_report(
        self,
        model_name: str,
        evaluation_results: Dict,
        training_metadata: Optional[Dict] = None,
        save_path: Optional[str] = None,
        model: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a markdown report for a single model.

        Args:
            model_name: Name of the model
            evaluation_results: Evaluation results dictionary
            training_metadata: Training metadata dictionary
            save_path: Path to save the report (if None, auto-generated)
            model: Trained model (for generating plots)
            X: Feature data (for generating plots)
            y: Target data (for generating plots)
            features: List of feature names used by the model

        Returns:
            Path where report was saved
        """
        report_lines = []

        # Header
        report_lines.append(f"# Model Evaluation Report: {model_name}")
        report_lines.append(
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("\n---\n")

        # Training metadata
        if training_metadata:
            report_lines.append("## Training Information\n")
            report_lines.append(
                f"- **Model Class**: {training_metadata.get('model_class', 'N/A')}"
            )
            report_lines.append(
                f"- **Features**: {', '.join(training_metadata.get('features', []))}"
            )
            report_lines.append(
                f"- **Training Samples**: {training_metadata.get('training_samples', 'N/A')}"
            )
            report_lines.append(
                f"- **Training Time**: {training_metadata.get('training_time_seconds', 'N/A')} seconds"
            )
            report_lines.append(
                f"- **Trained At**: {training_metadata.get('trained_at', 'N/A')}"
            )

            if "hyperparams" in training_metadata:
                report_lines.append("\n### Hyperparameters\n")
                for key, value in training_metadata["hyperparams"].items():
                    report_lines.append(f"- **{key}**: {value}")

            report_lines.append("\n---\n")

        # Evaluation results
        report_lines.append("## Evaluation Results\n")

        if "cv_metrics_summary" in evaluation_results:
            # Cross-validation results
            report_lines.append(
                f"**Cross-Validation**: {evaluation_results['n_folds']} folds\n"
            )

            report_lines.append("### CV Metrics (mean ± std)\n")
            report_lines.append("| Metric | Mean | Std | Min | Max |")
            report_lines.append("|--------|------|-----|-----|-----|")

            for metric_name, values in evaluation_results["cv_metrics_summary"].items():
                report_lines.append(
                    f"| {metric_name.replace('_', ' ').title()} | "
                    f"{values['mean']:.4f} | {values['std']:.4f} | "
                    f"{values['min']:.4f} | {values['max']:.4f} |"
                )

            report_lines.append("\n### Aggregate Metrics (all folds combined)\n")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")

            for metric_name, value in evaluation_results["aggregate_metrics"].items():
                report_lines.append(
                    f"| {metric_name.replace('_', ' ').title()} | {value:.4f} |"
                )

        elif "metrics" in evaluation_results:
            # Single evaluation results
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")

            for metric_name, value in evaluation_results["metrics"].items():
                report_lines.append(
                    f"| {metric_name.replace('_', ' ').title()} | {value:.4f} |"
                )

            if "confusion_matrix" in evaluation_results:
                report_lines.append("\n### Confusion Matrix\n")
                cm = np.array(evaluation_results["confusion_matrix"])
                report_lines.append(
                    "|           | Predicted Negative | Predicted Positive |"
                )
                report_lines.append(
                    "|-----------|--------------------|--------------------|"
                )
                report_lines.append(
                    f"| **Actual Negative** | {cm[0, 0]} | {cm[0, 1]} |"
                )
                report_lines.append(
                    f"| **Actual Positive** | {cm[1, 0]} | {cm[1, 1]} |"
                )

        report_lines.append("\n---\n")

        # Generate plots if model and data are provided
        if model is not None and X is not None and y is not None and features:
            report_lines.append("## Visualizations\n")
            
            import pandas as pd
            
            # Get the feature data
            if len(features) == 1:
                feature_name = features[0]
                X_feature = X[feature_name].values
                y_vals = y.values
                
                # Generate logistic curve plot
                logistic_curve_path = self.plot_logistic_curve(
                    model=model,
                    feature_name=feature_name,
                    X=X_feature,
                    y=y_vals,
                    model_name=model_name,
                )
                report_lines.append(f"### Logistic Curve\n")
                report_lines.append(f"![Logistic Curve]({Path(logistic_curve_path).name})\n")
                
                # Generate feature vs probability plot
                feature_prob_path = self.plot_feature_vs_probability(
                    model=model,
                    feature_name=feature_name,
                    X=X_feature,
                    y=y_vals,
                    model_name=model_name,
                )
                report_lines.append(f"### {feature_name.replace('_', ' ').title()} vs Goal Probability\n")
                report_lines.append(f"![Feature vs Probability]({Path(feature_prob_path).name})\n")
            
            report_lines.append("\n---\n")

        # Save report
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_report.md"
        else:
            save_path = Path(save_path)

        with open(save_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Generated report: {save_path}")
        return str(save_path)

    def generate_comparison_report(
        self,
        models_evaluations: Dict[str, Dict],
        models_metadata: Optional[Dict[str, Dict]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a markdown comparison report for multiple models.

        Args:
            models_evaluations: Dict mapping model names to evaluation results
            models_metadata: Dict mapping model names to training metadata
            save_path: Path to save the report (if None, auto-generated)

        Returns:
            Path where report was saved
        """
        report_lines = []

        # Header
        report_lines.append("# Model Comparison Report")
        report_lines.append(
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"\n**Number of Models**: {len(models_evaluations)}")
        report_lines.append("\n---\n")

        # Models summary
        report_lines.append("## Models Summary\n")
        report_lines.append("| Model | Features | Training Samples |")
        report_lines.append("|-------|----------|------------------|")

        for model_name in models_evaluations.keys():
            if models_metadata and model_name in models_metadata:
                metadata = models_metadata[model_name]
                features = ", ".join(metadata.get("features", []))
                samples = metadata.get("training_samples", "N/A")
            else:
                features = "N/A"
                samples = "N/A"

            report_lines.append(f"| {model_name} | {features} | {samples} |")

        report_lines.append("\n---\n")

        # Metrics comparison
        report_lines.append("## Metrics Comparison\n")

        # Extract metrics - handle both CV and single evaluation
        all_metrics = {}
        for model_name, eval_results in models_evaluations.items():
            if "aggregate_metrics" in eval_results:
                all_metrics[model_name] = eval_results["aggregate_metrics"]
            elif "metrics" in eval_results:
                all_metrics[model_name] = eval_results["metrics"]

        if all_metrics:
            # Get all metric names
            metric_names = list(next(iter(all_metrics.values())).keys())

            # Create table header
            report_lines.append("| Metric | " + " | ".join(all_metrics.keys()) + " |")
            report_lines.append(
                "|--------|" + "|".join(["--------"] * len(all_metrics)) + "|"
            )

            # Add rows for each metric
            for metric_name in metric_names:
                row = [metric_name.replace("_", " ").title()]
                for model_name in all_metrics.keys():
                    value = all_metrics[model_name].get(metric_name, 0)
                    row.append(f"{value:.4f}")

                report_lines.append("| " + " | ".join(row) + " |")

        # Best model per metric
        report_lines.append("\n### Best Model per Metric\n")
        report_lines.append("| Metric | Best Model | Value |")
        report_lines.append("|--------|------------|-------|")

        if all_metrics:
            metric_names = list(next(iter(all_metrics.values())).keys())

            for metric_name in metric_names:
                best_model = max(
                    all_metrics.keys(),
                    key=lambda m: all_metrics[m].get(metric_name, 0),
                )
                best_value = all_metrics[best_model][metric_name]

                report_lines.append(
                    f"| {metric_name.replace('_', ' ').title()} | "
                    f"{best_model} | {best_value:.4f} |"
                )

        report_lines.append("\n---\n")

        # Save report
        if save_path is None:
            save_path = self.output_dir / "model_comparison_report.md"
        else:
            save_path = Path(save_path)

        with open(save_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Generated comparison report: {save_path}")
        return str(save_path)
