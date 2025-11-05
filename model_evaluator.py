"""
Model evaluation module for computing metrics and generating evaluation results.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

    def predict_classes(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Get class predictions from the model.

        Args:
            X: Features to predict on
            threshold: Probability threshold for classification

        Returns:
            Predicted classes (0 or 1)
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Calculate comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted classes
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["log_loss"] = log_loss(y_true, y_proba)

        return metrics

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted classes

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    def roc_curve_data(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve data.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_proba)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Perform full evaluation on test set.

        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation results
        """
        # Get predictions
        y_proba = self.predict(X_test)
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred, y_proba)

        # Calculate confusion matrix
        cm = self.confusion_matrix(y_test.values, y_pred)

        # Calculate ROC curve data
        fpr, tpr, thresholds = self.roc_curve_data(y_test.values, y_proba)

        # Store results
        self.evaluation_results = {
            "model_name": self.model_name,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "n_samples": len(y_test),
            "n_positive": int(y_test.sum()),
            "n_negative": int(len(y_test) - y_test.sum()),
            "threshold": threshold,
        }

        return self.evaluation_results

    def evaluate_cross_val(
        self,
        models: List[Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.5,
        features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate models trained with cross-validation.

        Args:
            models: List of trained models (one per fold)
            X: Full feature dataset
            y: Full target series
            cv_splits: List of (train_indices, val_indices) tuples
            threshold: Classification threshold
            features: List of feature names to use (if None, uses self.features)

        Returns:
            Dictionary with aggregated CV evaluation results
        """
        # Use provided features or fall back to self.features
        feature_list = features if features is not None else self.features

        fold_metrics = []
        all_y_true = []
        all_y_proba = []

        for fold_idx, (model, (train_idx, val_idx)) in enumerate(
            zip(models, cv_splits)
        ):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Evaluate this fold
            evaluator = ModelEvaluator(
                model, f"{self.model_name}_fold{fold_idx}", features=feature_list
            )
            fold_results = evaluator.evaluate(X_val, y_val, threshold=threshold)

            fold_metrics.append(fold_results["metrics"])
            all_y_true.extend(y_val.values)
            all_y_proba.extend(evaluator.predict(X_val))

        # Calculate aggregate metrics
        all_y_true = np.array(all_y_true)
        all_y_proba = np.array(all_y_proba)
        all_y_pred = (all_y_proba >= threshold).astype(int)

        aggregate_metrics = self.calculate_metrics(all_y_true, all_y_pred, all_y_proba)

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
            "fold_metrics": fold_metrics,
            "n_folds": len(models),
            "threshold": threshold,
        }

        return self.evaluation_results

    def get_evaluation_dict(self) -> Dict:
        """
        Get the evaluation results dictionary.

        Returns:
            Dictionary with evaluation results
        """
        return self.evaluation_results.copy()
