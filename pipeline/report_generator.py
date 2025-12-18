"""
Report generation module for creating model evaluation reports and visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.plotting_tools import (
    plot_curve_against_actual,
    plot_curve_against_binned,
    plot_model_prediction_heatmap,
    plot_shot_heatmap,
    plot_shots,
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

    def generate_comparison_report(
        self,
        models_evaluations: Dict[str, Dict],
        models_metadata: Optional[Dict[str, Dict]] = None,
        trained_models: Optional[Dict[str, List[Any]]] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        models_features: Optional[Dict[str, List[str]]] = None,
        df: Optional[Any] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive experiment report with model comparison and individual model details.

        Args:
            models_evaluations: Dict mapping model names to evaluation results
            models_metadata: Dict mapping model names to training metadata
            trained_models: Dict mapping model names to list of trained models (one per fold)
            X: Feature data (for generating plots)
            y: Target data (for generating plots)
            models_features: Dict mapping model names to their feature lists
            df: Full dataframe with shot data including x, y coordinates
            save_path: Path to save the report (if None, auto-generated)

        Returns:
            Path where report was saved
        """
        report_lines = []

        # Header
        report_lines.append("# Experiment Report")
        report_lines.append(
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"\n**Number of Models**: {len(models_evaluations)}")
        report_lines.append("\n---\n")

        # ============================================================
        # SECTION: Shot Locations Visualization
        # ============================================================
        if df is not None and not df.empty:
            # Check if dataframe has required columns
            if all(col in df.columns for col in ["x", "y", "goal"]):
                # Generate binned goal probability heatmap
                heatmap_path = plot_shot_heatmap(df, output_dir=str(self.output_dir))
                if heatmap_path:
                    report_lines.append("## Binned Goal Probability Heatmap\n")
                    report_lines.append(
                        f"![Binned Goal Probability Heatmap]({Path(heatmap_path).name})\n"
                    )
                    report_lines.append("---\n")

                # Generate shot locations scatter plot
                shot_plot_path = plot_shots(df, output_dir=str(self.output_dir))
                if shot_plot_path:
                    report_lines.append("## Shot Locations\n")
                    report_lines.append(
                        f"![Shot Locations]({Path(shot_plot_path).name})\n"
                    )
                    report_lines.append("---\n")

        # ============================================================
        # SECTION: Model Comparison
        # ============================================================
        report_lines.append("# Model Comparison\n")

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

        report_lines.append("\n")

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

        report_lines.append("\n")

        # Best model per metric
        report_lines.append("## Best Model per Metric\n")
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

        # ============================================================
        # SECTIONS: Individual Model Details
        # ============================================================
        for model_name in models_evaluations.keys():
            report_lines.append(f"# Model: {model_name}\n")

            evaluation_results = models_evaluations[model_name]
            training_metadata = (
                models_metadata.get(model_name) if models_metadata else None
            )

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

                report_lines.append("\n")

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

                for metric_name, values in evaluation_results[
                    "cv_metrics_summary"
                ].items():
                    report_lines.append(
                        f"| {metric_name.replace('_', ' ').title()} | "
                        f"{values['mean']:.4f} | {values['std']:.4f} | "
                        f"{values['min']:.4f} | {values['max']:.4f} |"
                    )

                report_lines.append("\n### Aggregate Metrics (all folds combined)\n")
                report_lines.append("| Metric | Value |")
                report_lines.append("|--------|-------|")

                for metric_name, value in evaluation_results[
                    "aggregate_metrics"
                ].items():
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

            report_lines.append("\n")

            # Generate plots if model and data are provided
            if (
                trained_models is not None
                and model_name in trained_models
                and X is not None
                and y is not None
                and models_features is not None
                and model_name in models_features
            ):
                features = models_features[model_name]
                models = trained_models[model_name]
                model = models[0] if models else None

                if model is not None:
                    report_lines.append("## Visualizations\n")

                    # Generate model prediction heatmap (for all models)
                    prediction_heatmap_path = plot_model_prediction_heatmap(
                        model=model,
                        model_name=model_name,
                        features=features,
                        output_dir=self.output_dir,
                    )
                    report_lines.append("### Predicted Goal Probability Heatmap\n")
                    report_lines.append(f"![]({Path(prediction_heatmap_path).name})\n")

                    # Generate feature-specific plots only for single-feature models
                    if len(features) == 1:
                        feature_name = features[0]
                        X_feature = X[feature_name].values
                        y_vals = y.values

                        # Generate logistic curve plot
                        binned_plot_path = plot_curve_against_binned(
                            model=model,
                            feature_name=feature_name,
                            X=X_feature,
                            y=y_vals,
                            model_name=model_name,
                            output_dir=self.output_dir,
                        )
                        report_lines.append("### Comparison to binned shots\n")
                        report_lines.append(f"![]({Path(binned_plot_path).name})\n")

                        # Generate feature vs probability plot
                        curve_actual_plot_path = plot_curve_against_actual(
                            model=model,
                            feature_name=feature_name,
                            X=X_feature,
                            y=y_vals,
                            model_name=model_name,
                            output_dir=self.output_dir,
                        )
                        report_lines.append(
                            f"### {feature_name.replace('_', ' ').title()} vs actual shot outcome\n"
                        )
                        report_lines.append(
                            f"![]({Path(curve_actual_plot_path).name})\n"
                        )

            report_lines.append("\n---\n")

        # Save report
        if save_path is None:
            save_path = self.output_dir / "experiment_report.md"
        else:
            save_path = Path(save_path)

        with open(save_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Generated experiment report: {save_path}")
        return str(save_path)
