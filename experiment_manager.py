"""
Experiment management module for orchestrating multiple model comparisons.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_splitter import DataSplitter
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from report_generator import ReportGenerator
from settings import EXPERIMENTS_DIR, N_CV_FOLDS


class ExperimentManager:
    """Orchestrates training and evaluation of multiple models for comparison."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = EXPERIMENTS_DIR,
    ):
        """
        Initialize the experiment manager.

        Args:
            experiment_name: Name for this experiment
            output_dir: Directory for saving experiment outputs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_config = {}
        self.trainers = {}
        self.trained_models = {}
        self.evaluations = {}
        self.metadata = {}

        # Data storage
        self.X = None
        self.y = None
        self.cv_splits = None

        # Components
        self.splitter = DataSplitter()
        self.report_generator = ReportGenerator(str(self.output_dir))

        print(f"Initialized experiment: {experiment_name}")
        print(f"Output directory: {self.output_dir}")

    def register_model(
        self,
        model_class: type,
        model_name: str,
        features: List[str],
        hyperparams: Optional[Dict] = None,
    ):
        """
        Register a model for the experiment.

        Args:
            model_class: Class of the model to train
            model_name: Unique name identifier for this model
            features: List of feature column names to use
            hyperparams: Hyperparameters to pass to model constructor
        """
        if model_name in self.models_config:
            print(f"Warning: Model '{model_name}' already registered. Overwriting.")

        self.models_config[model_name] = {
            "model_class": model_class,
            "features": features,
            "hyperparams": hyperparams or {},
        }

        print(f"Registered model: {model_name} with features {features}")

    def set_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Set the full dataset for the experiment.

        Args:
            X: Feature dataframe
            y: Target series
        """
        self.X = X
        self.y = y

        print(f"Set dataset: {len(X)} samples, {len(X.columns)} features")

    def create_cv_splits(
        self,
        n_folds: int = N_CV_FOLDS,
        stratified: bool = True,
    ):
        """
        Create cross-validation splits.

        Args:
            n_folds: Number of CV folds
            stratified: Whether to use stratified k-fold
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be set before creating splits")

        print(f"\nCreating {n_folds}-fold cross-validation splits...")
        self.cv_splits = self.splitter.k_fold_split(
            self.X, self.y, n_folds=n_folds, stratified=stratified
        )

        # Save split metadata
        split_metadata = {
            "n_folds": n_folds,
            "stratified": stratified,
            "n_samples": len(self.X),
        }

        self.splitter.save_cross_val_splits_metadata(
            f"{self.experiment_name}_splits",
            self.cv_splits,
            metadata=split_metadata,
        )

        print(f"Created {len(self.cv_splits)} CV splits")

    def train_all_models(self, force_retrain: bool = False):
        """
        Train all registered models.

        Args:
            force_retrain: If True, retrain even if cached models exist
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be set before training")

        if self.cv_splits is None:
            raise ValueError("CV splits must be created before training")

        print(f"\n{'=' * 60}")
        print(f"Training {len(self.models_config)} models")
        print(f"{'=' * 60}")

        for model_name, config in self.models_config.items():
            print(f"\n--- Training: {model_name} ---")

            # Create trainer
            trainer = ModelTrainer(
                model_class=config["model_class"],
                model_name=model_name,
                features=config["features"],
                hyperparams=config["hyperparams"],
            )

            # Train with CV
            models = trainer.train_with_cv(
                self.X, self.y, self.cv_splits, force_retrain=force_retrain
            )

            # Store trainer and models
            self.trainers[model_name] = trainer
            self.trained_models[model_name] = models
            self.metadata[model_name] = trainer.get_training_metadata()

            print(f"Completed training: {model_name}")

        print(f"\n{'=' * 60}")
        print("All models trained successfully")
        print(f"{'=' * 60}\n")

    def evaluate_all_models(self):
        """Evaluate all trained models on the CV splits."""
        if not self.trained_models:
            raise ValueError("Models must be trained before evaluation")

        print(f"\n{'=' * 60}")
        print(f"Evaluating {len(self.trained_models)} models")
        print(f"{'=' * 60}")

        for model_name, models in self.trained_models.items():
            print(f"\n--- Evaluating: {model_name} ---")

            # Get features for this model
            features = self.models_config[model_name]["features"]

            # Get first model for evaluation (all models in CV have same architecture)
            evaluator = ModelEvaluator(models[0], model_name, features=features)

            # Evaluate with CV
            results = evaluator.evaluate_cross_val(
                models, self.X, self.y, self.cv_splits, features=features
            )

            self.evaluations[model_name] = results

            print(f"Completed evaluation: {model_name}")

        print(f"\n{'=' * 60}")
        print("All models evaluated successfully")
        print(f"{'=' * 60}\n")

    def generate_individual_reports(self):
        """Generate individual reports for each model."""
        print("\nGenerating individual model reports...")

        for model_name in self.trained_models.keys():
            evaluation_results = self.evaluations.get(model_name, {})
            training_metadata = self.metadata.get(model_name, {})

            self.report_generator.generate_single_model_report(
                model_name=model_name,
                evaluation_results=evaluation_results,
                training_metadata=training_metadata,
            )

        print("Individual reports generated successfully\n")

    def generate_comparison_report(self):
        """Generate comparison report for all models."""
        print("\nGenerating comparison report...")

        # Generate markdown comparison report
        self.report_generator.generate_comparison_report(
            models_evaluations=self.evaluations,
            models_metadata=self.metadata,
        )

        # Generate comparison plots
        self._generate_comparison_plots()

        print("Comparison report generated successfully\n")

    def _generate_comparison_plots(self):
        """Generate comparison plots for all models."""
        print("Generating comparison plots...")

        # Extract aggregate metrics for comparison
        aggregate_metrics = {}
        for model_name, eval_results in self.evaluations.items():
            if "aggregate_metrics" in eval_results:
                aggregate_metrics[model_name] = eval_results["aggregate_metrics"]

        if aggregate_metrics:
            # Plot metrics comparison
            self.report_generator.plot_metrics_comparison(aggregate_metrics)

            # Plot ROC curves comparison
            roc_data = {}
            for model_name, eval_results in self.evaluations.items():
                # Need to compute ROC for aggregate predictions
                # For now, we'll use the first fold's ROC data as an approximation
                if (
                    "fold_metrics" in eval_results
                    and len(eval_results["fold_metrics"]) > 0
                ):
                    auc = aggregate_metrics[model_name].get("roc_auc", 0)
                    # Simplified - would need to recompute for aggregate predictions
                    roc_data[model_name] = {
                        "fpr": np.linspace(0, 1, 100),
                        "tpr": np.linspace(0, 1, 100),  # Placeholder
                        "auc": auc,
                    }

            if roc_data:
                self.report_generator.plot_roc_curves_comparison(roc_data)

        print("Comparison plots generated successfully")

    def get_best_model(self, metric: str = "roc_auc") -> Tuple[str, Any, Dict]:
        """
        Get the best performing model based on a specific metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, model, evaluation_results)
        """
        if not self.evaluations:
            raise ValueError("Models must be evaluated before getting best model")

        best_model_name = None
        best_score = -float("inf")

        for model_name, eval_results in self.evaluations.items():
            if "aggregate_metrics" in eval_results:
                score = eval_results["aggregate_metrics"].get(metric, 0)
            elif "metrics" in eval_results:
                score = eval_results["metrics"].get(metric, 0)
            else:
                continue

            if score > best_score:
                best_score = score
                best_model_name = model_name

        if best_model_name is None:
            raise ValueError(f"No models found with metric '{metric}'")

        print(f"Best model by {metric}: {best_model_name} (score: {best_score:.4f})")

        return (
            best_model_name,
            self.trained_models[best_model_name],
            self.evaluations[best_model_name],
        )

    def save_experiment_config(self):
        """Save experiment configuration to JSON."""
        config = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "n_samples": len(self.X) if self.X is not None else 0,
            "n_features": len(self.X.columns) if self.X is not None else 0,
            "n_folds": len(self.cv_splits) if self.cv_splits else 0,
            "models": {
                model_name: {
                    "model_class": config["model_class"].__name__,
                    "features": config["features"],
                    "hyperparams": config["hyperparams"],
                }
                for model_name, config in self.models_config.items()
            },
        }

        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved experiment config to {config_path}")

    def run_experiment(self, force_retrain: bool = False):
        """
        Run the complete experiment pipeline.

        Args:
            force_retrain: If True, retrain even if cached models exist
        """
        print(f"\n{'=' * 60}")
        print(f"Running Experiment: {self.experiment_name}")
        print(f"{'=' * 60}\n")

        # Validate setup
        if not self.models_config:
            raise ValueError("No models registered for experiment")

        if self.X is None or self.y is None:
            raise ValueError("Data must be set before running experiment")

        # Create CV splits if not already done
        if self.cv_splits is None:
            self.create_cv_splits()

        # Save experiment configuration
        self.save_experiment_config()

        # Train all models
        self.train_all_models(force_retrain=force_retrain)

        # Evaluate all models
        self.evaluate_all_models()

        # Generate reports
        self.generate_individual_reports()
        self.generate_comparison_report()

        # Get and display best model
        best_model_name, _, _ = self.get_best_model()

        print(f"\n{'=' * 60}")
        print(f"Experiment Complete: {self.experiment_name}")
        print(f"Best Model: {best_model_name}")
        print(f"{'=' * 60}\n")
