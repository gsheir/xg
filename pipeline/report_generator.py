"""
Report generation module for creating model evaluation reports and visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.plotting_tools import (
    plot_feature_vs_probability,
    plot_logistic_curve,
)
from pipeline.settings import (
    OUTPUT_DIR,
)


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

            # Get the feature data
            if len(features) == 1:
                feature_name = features[0]
                X_feature = X[feature_name].values
                y_vals = y.values

                # Generate logistic curve plot
                logistic_curve_path = plot_logistic_curve(
                    model=model,
                    feature_name=feature_name,
                    X=X_feature,
                    y=y_vals,
                    model_name=model_name,
                    output_dir=self.output_dir,
                )
                report_lines.append("### Logistic Curve\n")
                report_lines.append(
                    f"![Logistic Curve]({Path(logistic_curve_path).name})\n"
                )

                # Generate feature vs probability plot
                feature_prob_path = plot_feature_vs_probability(
                    model=model,
                    feature_name=feature_name,
                    X=X_feature,
                    y=y_vals,
                    model_name=model_name,
                    output_dir=self.output_dir,
                )
                report_lines.append(
                    f"### {feature_name.replace('_', ' ').title()} vs Goal Probability\n"
                )
                report_lines.append(
                    f"![Feature vs Probability]({Path(feature_prob_path).name})\n"
                )

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
