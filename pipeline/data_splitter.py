"""
Data splitting module for train/test splits and cross-validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from pipeline.settings import N_FOLDS, OUTPUT_DIR, RANDOM_SEED, TEST_SIZE


class DataSplitter:
    """Handles data splitting for train/test and cross-validation."""

    def __init__(
        self, experiment_name, settings: Dict[str, Any], splits_dir: str = OUTPUT_DIR
    ):
        """Initialize the data splitter with a directory for saving splits."""
        self.splits_dir = Path(splits_dir) / "experiments" / experiment_name
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.settings = settings

        self.X = None
        self.y = None

        self.splits = None

    def split(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform data splitting based on settings.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Dictionary with split information
        """
        self.splits = {}

        self.X = X
        self.y = y

        if self.settings["split_type"] == "cross_validation":
            n_folds = self.settings.get("n_folds", N_FOLDS)
            stratified = self.settings.get("stratify", True)

            cv_splits = self.k_fold_split(
                n_folds=n_folds,
                stratified=stratified,
                random_state=RANDOM_SEED,
            )
            self.splits["cv_splits"] = cv_splits

            self.save_split_metadata()

        elif self.settings["split_type"] == "train_test":
            stratify = self.settings.get("stratify", True)

            X_train, X_test, y_train, y_test = self.train_test_split(
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                stratify=stratify,
            )
            self.splits["X_train"] = X_train
            self.splits["X_test"] = X_test
            self.splits["y_train"] = y_train
            self.splits["y_test"] = y_test

            self.save_split_metadata(
                train_indices=X_train.index.to_numpy(),
                test_indices=X_test.index.to_numpy(),
            )

        else:
            raise ValueError(f"Unknown split type: {self.settings['split_type']}")

        return self.splits

    def train_test_split(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEED,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform train/test split.

        Args:
            X: Feature dataframe
            y: Target series
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split based on target

        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_param = self.y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )

        return X_train, X_test, y_train, y_test

    def k_fold_split(
        self,
        n_folds: int = N_FOLDS,
        stratified: bool = True,
        random_state: int = RANDOM_SEED,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create k-fold cross-validation splits.

        Args:
            X: Feature dataframe
            y: Target series
            n_folds: Number of folds
            stratified: Whether to use stratified k-fold
            random_state: Random seed for reproducibility

        Returns:
            List of (train_indices, val_indices) tuples
        """
        if stratified:
            kfold = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        cv_splits = []
        for train_idx, val_idx in kfold.split(self.X, self.y):
            cv_splits.append((train_idx, val_idx))

        return cv_splits

    def save_split_metadata(
        self,
    ):
        """
        Save split indices and metadata to JSON file.

        Args:
            split_name: Name for this split
            train_indices: Training set indices
            test_indices: Test set indices
            metadata: Additional metadata to save
        """
        if self.settings["split_type"] == "cross_validation":
            splits_data = {
                "split_name": self.experiment_name,
                "split_type": "cross_validation",
                "n_folds": self.settings["n_folds"],
                "folds": [
                    {
                        "fold": i,
                        "train_indices": train_idx.tolist(),
                        "val_indices": val_idx.tolist(),
                        "train_size": len(train_idx),
                        "val_size": len(val_idx),
                    }
                    for i, (train_idx, val_idx) in enumerate(self.splits["cv_splits"])
                ],
            }
        elif self.settings["split_type"] == "train_test":
            splits_data = {
                "split_name": self.experiment_name,
                "split_type": "train_test",
                "train_indices": self.splits["X_train"].index.to_numpy().tolist(),
                "test_indices": self.splits["X_test"].index.to_numpy().tolist(),
                "train_size": len(self.splits["X_train"]),
                "test_size": len(self.splits["X_test"]),
            }
        else:
            raise ValueError(f"Unknown split type: {self.settings['split_type']}")

        filepath = self.splits_dir / f"{self.experiment_name}_splits.json"
        with open(filepath, "w") as f:
            json.dump(splits_data, f, indent=2)

        print(f"Saved split metadata to {filepath}")

    def load_split_metadata(self) -> Dict:
        """
        Load split metadata from JSON file.

        Args:
            split_name: Name of the split to load

        Returns:
            Dictionary with split data
        """
        filepath = self.splits_dir / f"{self.experiment_name}_splits.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Split metadata not found: {filepath}")

        with open(filepath, "r") as f:
            split_data = json.load(f)

        # Convert indices back to numpy arrays
        split_data["train_indices"] = np.array(split_data["train_indices"])
        split_data["test_indices"] = np.array(split_data["test_indices"])

        return split_data

    def load_cross_val_splits_metadata(self, split_name: str) -> Dict:
        """
        Load cross-validation split metadata from JSON file.

        Args:
            split_name: Name of the CV split set to load

        Returns:
            Dictionary with CV splits data
        """
        filepath = self.splits_dir / f"{split_name}_cv.json"

        if not filepath.exists():
            raise FileNotFoundError(f"CV split metadata not found: {filepath}")

        with open(filepath, "r") as f:
            splits_data = json.load(f)

        # Convert indices back to numpy arrays
        for fold in splits_data["folds"]:
            fold["train_indices"] = np.array(fold["train_indices"])
            fold["val_indices"] = np.array(fold["val_indices"])

        return splits_data
