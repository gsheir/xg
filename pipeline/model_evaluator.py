"""
Model evaluation module for computing metrics and generating evaluation results.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluates trained models and computes comprehensive metrics."""

    def __init__(
        self,
        model: Any,
        model_name: str = "model",
        features: Optional[List[str]] = None,
    ):
        """
        Initialize the model evaluator.

        Args:
            model: Trained model to evaluate
            model_name: Name identifier for the model
            features: List of feature names the model was trained on (for feature selection)
        """
        self.model = model
        self.model_name = model_name
        self.features = features
        self.evaluation_results = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from the model.

        Args:
            X: Features to predict on

        Returns:
            Predicted probabilities
        """
        # Select only the features the model was trained on
        if self.features is not None:
            X = X[self.features]
        return self.model.predict(X)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict:
        """
        Calculate comprehensive regression metrics for xG evaluation.

        Args:
            y_true: True labels (0 or 1 for goals)
            y_pred_proba: Predicted probabilities (continuous 0-1)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred_proba),
            "mse": mean_squared_error(y_true, y_pred_proba),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred_proba)),
            "r2_score": r2_score(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

        return metrics

    def calculate_calibration_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve data by binning predictions.

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Tuple of (bin_midpoints, actual_rates, bin_counts)
        """
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_midpoints = []
        actual_rates = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_midpoints.append((bins[i] + bins[i + 1]) / 2)
                actual_rates.append(y_true[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_midpoints.append((bins[i] + bins[i + 1]) / 2)
                actual_rates.append(np.nan)
                bin_counts.append(0)

        return (
            np.array(bin_midpoints),
            np.array(actual_rates),
            np.array(bin_counts),
        )

    def calculate_expected_calibration_error(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            ECE value
        """
        bin_midpoints, actual_rates, bin_counts = self.calculate_calibration_curve(
            y_true, y_pred_proba, n_bins
        )

        # Filter out bins with no samples
        valid_bins = bin_counts > 0
        bin_midpoints = bin_midpoints[valid_bins]
        actual_rates = actual_rates[valid_bins]
        bin_counts = bin_counts[valid_bins]

        # Calculate weighted average of absolute differences
        total_samples = bin_counts.sum()
        ece = np.sum(bin_counts / total_samples * np.abs(bin_midpoints - actual_rates))

        return float(ece)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bins: int = 10,
    ) -> Dict:
        """
        Perform full evaluation on test set.

        Args:
            X_test: Test features
            y_test: Test targets
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with evaluation results
        """
        # Get predictions
        y_pred_proba = self.predict(X_test)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred_proba)

        # Calculate calibration curve
        bin_midpoints, actual_rates, bin_counts = self.calculate_calibration_curve(
            y_test.values, y_pred_proba, n_bins
        )

        # Calculate ECE
        ece = self.calculate_expected_calibration_error(
            y_test.values, y_pred_proba, n_bins
        )
        metrics["ece"] = ece

        # Store results
        self.evaluation_results = {
            "model_name": self.model_name,
            "metrics": metrics,
            "calibration_curve": {
                "bin_midpoints": bin_midpoints.tolist(),
                "actual_rates": [
                    float(x) if not np.isnan(x) else None for x in actual_rates
                ],
                "bin_counts": bin_counts.tolist(),
            },
            "n_samples": len(y_test),
            "n_goals": int(y_test.sum()),
            "n_non_goals": int(len(y_test) - y_test.sum()),
            "mean_prediction": float(y_pred_proba.mean()),
            "std_prediction": float(y_pred_proba.std()),
        }

        return self.evaluation_results

    def evaluate_cross_val(
        self,
        models: List[Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        n_bins: int = 10,
        features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate models trained with cross-validation.

        Args:
            models: List of trained models (one per fold)
            X: Full feature dataset
            y: Full target series
            cv_splits: List of (train_indices, val_indices) tuples
            n_bins: Number of bins for calibration curve
            features: List of feature names to use (if None, uses self.features)

        Returns:
            Dictionary with aggregated CV evaluation results
        """
        # Use provided features or fall back to self.features
        feature_list = features if features is not None else self.features

        fold_metrics = []
        all_y_true = []
        all_y_pred_proba = []

        for fold_idx, (model, (train_idx, val_idx)) in enumerate(
            zip(models, cv_splits)
        ):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Evaluate this fold
            evaluator = ModelEvaluator(
                model, f"{self.model_name}_fold{fold_idx}", features=feature_list
            )
            fold_results = evaluator.evaluate(X_val, y_val, n_bins=n_bins)

            fold_metrics.append(fold_results["metrics"])
            all_y_true.extend(y_val.values)
            all_y_pred_proba.extend(evaluator.predict(X_val))

        # Calculate aggregate metrics
        all_y_true = np.array(all_y_true)
        all_y_pred_proba = np.array(all_y_pred_proba)

        aggregate_metrics = self.calculate_metrics(all_y_true, all_y_pred_proba)

        # Calculate aggregate calibration
        bin_midpoints, actual_rates, bin_counts = self.calculate_calibration_curve(
            all_y_true, all_y_pred_proba, n_bins
        )
        ece = self.calculate_expected_calibration_error(
            all_y_true, all_y_pred_proba, n_bins
        )
        aggregate_metrics["ece"] = ece

        # Calculate mean and std for each metric across folds
        metrics_summary = {}
        for metric_name in fold_metrics[0].keys():
            values = [fold[metric_name] for fold in fold_metrics]
            metrics_summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Store results
        self.evaluation_results = {
            "model_name": self.model_name,
            "cv_metrics_summary": metrics_summary,
            "aggregate_metrics": aggregate_metrics,
            "aggregate_calibration_curve": {
                "bin_midpoints": bin_midpoints.tolist(),
                "actual_rates": [
                    float(x) if not np.isnan(x) else None for x in actual_rates
                ],
                "bin_counts": bin_counts.tolist(),
            },
            "fold_metrics": fold_metrics,
            "n_folds": len(models),
        }

        return self.evaluation_results

    def get_evaluation_dict(self) -> Dict:
        """
        Get the evaluation results dictionary.

        Returns:
            Dictionary with evaluation results
        """
        return self.evaluation_results.copy()
