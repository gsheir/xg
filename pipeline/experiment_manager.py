"""
Experiment management module for orchestrating multiple model comparisons.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pipeline.data_preprocessor import DataPreprocessor
from pipeline.data_splitter import DataSplitter
from pipeline.model_evaluator import ModelEvaluator
from pipeline.model_trainer import ModelTrainer
from pipeline.report_generator import ReportGenerator


class ExperimentManager:
    """Orchestrates training and evaluation of multiple models for comparison."""

    def __init__(
        self,
        experiment_name: str,
        experiment_settings: Dict[str, Any],
    ):
        """
        Initialize the experiment manager.

        Args:
            experiment_name: Name for this experiment
            output_dir: Directory for saving experiment outputs
        """
        self.experiment_name = experiment_name
        self.experiment_settings = experiment_settings

        self.output_dir = Path(experiment_settings["output_dir"]) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_config = {}
        self.trainers = {}
        self.trained_models = {}
        self.evaluations = {}
        self.metadata = {}

        # Data storage
        self.df = None
        self.X = None
        self.y = None
        self.splits = None

        # Components
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter(
            self.experiment_name, experiment_settings["split_settings"]
        )
        self.report_generator = ReportGenerator(str(self.output_dir))

        print(f"Initialized experiment: {experiment_name}")
        print(f"Output directory: {self.output_dir}")

    def register_dataset(self, df: pd.DataFrame):
        """
        Set the full dataset for the experiment.

        Args:
            X: Feature dataframe
            y: Target series
        """
        self.df = df

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

    def create_splits(
        self,
    ):
        """
        Create cross-validation splits.

        Args:
            n_folds: Number of CV folds
        """

        if self.X is None or self.y is None:
            raise ValueError("Data must be set before creating splits")

        self.splits = self.splitter.split(self.X, self.y)

        print(f"Created {len(self.splits)} cross-validation splits")

    def train_all_models(self, force_retrain: bool = False):
        """
        Train all registered models.

        Args:
            force_retrain: If True, retrain even if cached models exist
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be set before training")

        if self.splits is None:
            raise ValueError("Cross-validation splits must be created before training")

        print(f"Training {len(self.models_config)} models")

        for model_name, config in self.models_config.items():
            print(f"\n--- Training: {model_name} ---")

            # Create trainer
            trainer = ModelTrainer(
                model_class=config["model_class"],
                model_name=model_name,
                features=config["features"],
                hyperparams=config["hyperparams"],
            )

            if self.experiment_settings["split_settings"]["split_type"] == "train_test":
                # Train with train-test split
                model = trainer.train(
                    self.X,
                    self.y,
                    force_retrain=force_retrain,
                )
                models = [model]
            else:
                # Train with cross-validation
                models = trainer.train_with_cross_val(
                    self.X,
                    self.y,
                    self.splits["cv_splits"],
                    force_retrain=force_retrain,
                )

            # Store trainer and models
            self.trainers[model_name] = trainer
            self.trained_models[model_name] = models
            self.metadata[model_name] = trainer.get_training_metadata()

            print(f"Completed training: {model_name}")

        print("All models trained successfully")

    def evaluate_all_models(self):
        """Evaluate all trained models on the cross-validation splits."""
        if not self.trained_models:
            raise ValueError("Models must be trained before evaluation")

        print(f"Evaluating {len(self.trained_models)} models")

        for model_name, models in self.trained_models.items():
            print(f"\n--- Evaluating: {model_name} ---")

            # Get features for this model
            features = self.models_config[model_name]["features"]

            # Get first model for evaluation (all models in cross-validation have same architecture)
            evaluator = ModelEvaluator(models[0], model_name, features=features)

            # Evaluate with cross-validation
            results = evaluator.evaluate_cross_val(
                models, self.X, self.y, self.splits["cv_splits"], features=features
            )

            self.evaluations[model_name] = results

            print(f"Completed evaluation: {model_name}")

        print("All models evaluated successfully")

    def generate_experiment_report(self):
        """Generate comprehensive experiment report with all models."""
        print("\nGenerating experiment report...")

        # Prepare models features dictionary
        models_features = {
            model_name: config["features"]
            for model_name, config in self.models_config.items()
        }

        # Generate comprehensive experiment report
        self.report_generator.generate_comparison_report(
            models_evaluations=self.evaluations,
            models_metadata=self.metadata,
            trained_models=self.trained_models,
            X=self.X,
            y=self.y,
            models_features=models_features,
            df=self.df,
        )

        print("Experiment report generated successfully")

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
            "models": {
                model_name: {
                    "model_class": config["model_class"].__name__,
                    "features": config["features"],
                    "hyperparams": config["hyperparams"],
                }
                for model_name, config in self.models_config.items()
            },
            "experiment_settings": self.experiment_settings,
        }

        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved experiment config to {config_path}")

    def run(self, force_retrain: bool = False):
        """
        Run the complete experiment pipeline.

        Args:
            force_retrain: If True, retrain even if cached models exist
        """
        print(f"Running Experiment: {self.experiment_name}")

        # Validate setup
        if not self.models_config:
            raise ValueError("No models registered for experiment")

        if self.df is None:
            raise ValueError("No dataset registered for experiment")

        # Preprocess data
        self.preprocessor.load_data(self.df)
        self.df = self.preprocessor.prepare_xg_data()

        if self.df.empty:
            print("After preprocessing, no data is available.")
            return

        # Set features and target
        feature_columns = self.experiment_settings["feature_columns"]
        target_column = self.experiment_settings["target_column"]

        self.X = self.df[feature_columns]
        self.y = self.df[target_column]

        # Create cross-validation splits if not already done
        if self.splits is None:
            self.create_splits()

        # Save experiment configuration
        self.save_experiment_config()

        # Train all models
        self.train_all_models(force_retrain=force_retrain)

        # Evaluate all models
        self.evaluate_all_models()

        # Generate comprehensive experiment report
        self.generate_experiment_report()

        # Get and display best model
        best_model_name, _, _ = self.get_best_model()

        print(f"Experiment Complete: {self.experiment_name}")
        print(f"Best Model: {best_model_name}")
