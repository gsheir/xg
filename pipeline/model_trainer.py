"""
Model training module with support for caching and cross-validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.model_persistence import ModelPersistence


class ModelTrainer:
    """Handles model training with caching and cross-validation support."""

    def __init__(
        self,
        model_class: type,
        model_name: str,
        features: List[str],
        hyperparams: Optional[Dict] = None,
        cache_dir: str = "models/saved",
    ):
        """
        Initialize the model trainer.

        Args:
            model_class: Class of the model to train
            model_name: Name identifier for this model
            features: List of feature column names to use
            hyperparams: Hyperparameters to pass to model constructor
            cache_dir: Directory for caching trained models
        """
        self.model_class = model_class
        self.model_name = model_name
        self.features = features
        self.hyperparams = hyperparams or {}
        self.persistence = ModelPersistence(cache_dir)

        self.model = None
        self.training_metadata = {}

    def _get_model_identifier(self, suffix: str = "") -> str:
        """
        Generate a unique identifier for the model.

        Args:
            suffix: Optional suffix to add to identifier

        Returns:
            Model identifier string
        """
        feature_str = "_".join(self.features)
        identifier = f"{self.model_name}_{feature_str}"

        if suffix:
            identifier = f"{identifier}_{suffix}"

        return identifier

    def check_pretrained_exists(self, suffix: str = "") -> bool:
        """
        Check if a pretrained model exists.

        Args:
            suffix: Optional suffix for model identifier

        Returns:
            True if pretrained model exists, False otherwise
        """
        identifier = self._get_model_identifier(suffix)
        return self.persistence.model_exists(identifier)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        force_retrain: bool = False,
        suffix: str = "",
    ) -> Any:
        """
        Train a model on the provided data.

        Args:
            X_train: Training features
            y_train: Training targets
            force_retrain: If True, retrain even if cached model exists
            suffix: Optional suffix for model identifier

        Returns:
            Trained model
        """
        identifier = self._get_model_identifier(suffix)

        # Check for pretrained model
        if not force_retrain and self.persistence.model_exists(identifier):
            print(f"Loading pretrained model: {identifier}")
            self.model = self.persistence.load_model(identifier)
            self.training_metadata = self.persistence.get_model_metadata(identifier)
            return self.model

        # Train new model
        print(f"Training new model: {identifier}")
        print(f"Features: {self.features}")
        print(f"Training samples: {len(X_train)}")

        # Select only the specified features
        X_train_selected = X_train[self.features]

        # Create and train model
        self.model = self.model_class(**self.hyperparams)
        start_time = datetime.now()
        self.model.fit(X_train_selected, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Store metadata
        self.training_metadata = {
            "model_class": self.model_class.__name__,
            "model_name": self.model_name,
            "features": self.features,
            "hyperparams": self.hyperparams,
            "training_samples": len(X_train),
            "training_time_seconds": round(training_time, 2),
            "trained_at": datetime.now().isoformat(),
        }

        # Save model and metadata
        self.persistence.save_model(self.model, identifier, self.training_metadata)

        return self.model

    def train_with_cross_val(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        force_retrain: bool = False,
    ) -> List[Any]:
        """
        Train models using cross-validation splits.

        Args:
            X: Full feature dataset
            y: Full target series
            cv_splits: List of (train_indices, val_indices) tuples
            force_retrain: If True, retrain even if cached models exist

        Returns:
            List of trained models for each fold
        """
        models = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n--- Training fold {fold_idx + 1}/{len(cv_splits)} ---")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]

            # Train model for this fold
            fold_suffix = f"fold{fold_idx}"
            model = self.train(
                X_train_fold,
                y_train_fold,
                force_retrain=force_retrain,
                suffix=fold_suffix,
            )

            models.append(model)

        return models

    def get_training_metadata(self) -> Dict:
        """
        Get metadata about the training process.

        Returns:
            Dictionary with training metadata
        """
        return self.training_metadata.copy()

    def load_pretrained(self, suffix: str = "") -> Any:
        """
        Load a pretrained model.

        Args:
            suffix: Optional suffix for model identifier

        Returns:
            Loaded model
        """
        identifier = self._get_model_identifier(suffix)
        self.model = self.persistence.load_model(identifier)
        self.training_metadata = self.persistence.get_model_metadata(identifier)
        return self.model

    def load_cv_models(self, n_folds: int) -> List[Any]:
        """
        Load all CV fold models.

        Args:
            n_folds: Number of folds to load

        Returns:
            List of loaded models
        """
        models = []
        for fold_idx in range(n_folds):
            fold_suffix = f"fold{fold_idx}"
            model = self.load_pretrained(suffix=fold_suffix)
            models.append(model)

        return models
