"""
Data splitting module for train/test splits and cross-validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from settings import N_CV_FOLDS, RANDOM_SEED, SPLITS_DIR, TEST_SIZE


class DataSplitter:
    """Handles data splitting for train/test and cross-validation."""

    def __init__(self, splits_dir: str = SPLITS_DIR):
        """Initialize the data splitter with a directory for saving splits."""
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
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
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )

        return X_train, X_test, y_train, y_test

    def k_fold_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = N_CV_FOLDS,
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

        splits = []
        for train_idx, val_idx in kfold.split(X, y):
            splits.append((train_idx, val_idx))

        return splits

    def save_split_metadata(
        self,
        split_name: str,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        """
        Save split indices and metadata to JSON file.

        Args:
            split_name: Name for this split
            train_indices: Training set indices
            test_indices: Test set indices
            metadata: Additional metadata to save
        """
        split_data = {
            "split_name": split_name,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "metadata": metadata or {},
        }

        filepath = self.splits_dir / f"{split_name}.json"
        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=2)

        print(f"Saved split metadata to {filepath}")

    def save_cross_val_splits_metadata(
        self,
        split_name: str,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        metadata: Optional[Dict] = None,
    ):
        """
        Save cross-validation split indices and metadata to JSON file.

        Args:
            split_name: Name for this CV split set
            cv_splits: List of (train_indices, val_indices) tuples
            metadata: Additional metadata to save
        """
        splits_data = {
            "split_name": split_name,
            "n_folds": len(cv_splits),
            "folds": [
                {
                    "fold": i,
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                }
                for i, (train_idx, val_idx) in enumerate(cv_splits)
            ],
            "metadata": metadata or {},
        }

        filepath = self.splits_dir / f"{split_name}_cv.json"
        with open(filepath, "w") as f:
            json.dump(splits_data, f, indent=2)

        print(f"Saved CV split metadata to {filepath}")

    def load_split_metadata(self, split_name: str) -> Dict:
        """
        Load split metadata from JSON file.

        Args:
            split_name: Name of the split to load

        Returns:
            Dictionary with split data
        """
        filepath = self.splits_dir / f"{split_name}.json"

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
