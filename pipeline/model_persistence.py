"""
Model persistence module for saving and loading trained models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from pipeline.settings import MODELS_DIR


class ModelPersistence:
    """Handles saving and loading of trained models with metadata."""

    def __init__(self, base_dir: str = MODELS_DIR):
        """Initialize the model persistence handler."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, identifier: str) -> Path:
        """Get the directory path for a specific model."""
        model_dir = self.base_dir / identifier
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_model_path(self, identifier: str) -> Path:
        """Get the file path for the model weights."""
        return self._get_model_dir(identifier) / "model.joblib"

    def _get_metadata_path(self, identifier: str) -> Path:
        """Get the file path for the model metadata."""
        return self._get_model_dir(identifier) / "metadata.json"

    def save_model(
        self,
        model: Any,
        identifier: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Save a trained model and its metadata.

        Args:
            model: Trained model object
            identifier: Unique identifier for this model
            metadata: Additional metadata to save with the model
        """
        model_path = self._get_model_path(identifier)
        metadata_path = self._get_metadata_path(identifier)

        # Save model using joblib
        joblib.dump(model, model_path)

        # Prepare metadata
        full_metadata = {
            "identifier": identifier,
            "saved_at": datetime.now().isoformat(),
            "model_path": str(model_path),
        }

        if metadata:
            full_metadata.update(metadata)

        # Save metadata as JSON
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        print(f"Saved model to {model_path}")
        print(f"Saved metadata to {metadata_path}")

    def load_model(self, identifier: str) -> Any:
        """
        Load a trained model.

        Args:
            identifier: Unique identifier for the model

        Returns:
            Loaded model object
        """
        model_path = self._get_model_path(identifier)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        return model

    def model_exists(self, identifier: str) -> bool:
        """
        Check if a model exists.

        Args:
            identifier: Unique identifier for the model

        Returns:
            True if model exists, False otherwise
        """
        return self._get_model_path(identifier).exists()

    def get_model_metadata(self, identifier: str) -> Dict:
        """
        Get metadata for a saved model.

        Args:
            identifier: Unique identifier for the model

        Returns:
            Dictionary with model metadata
        """
        metadata_path = self._get_metadata_path(identifier)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata

    def list_saved_models(self) -> list:
        """
        List all saved models.

        Returns:
            List of model identifiers
        """
        if not self.base_dir.exists():
            return []

        models = []
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir() and self._get_model_path(model_dir.name).exists():
                models.append(model_dir.name)

        return sorted(models)

    def delete_model(self, identifier: str):
        """
        Delete a saved model and its metadata.

        Args:
            identifier: Unique identifier for the model
        """
        model_dir = self._get_model_dir(identifier)

        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return

        # Delete all files in the model directory
        for file in model_dir.iterdir():
            file.unlink()

        # Delete the directory
        model_dir.rmdir()

        print(f"Deleted model: {identifier}")

    def get_model_size(self, identifier: str) -> float:
        """
        Get the size of a saved model in MB.

        Args:
            identifier: Unique identifier for the model

        Returns:
            Model size in megabytes
        """
        model_path = self._get_model_path(identifier)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        return round(size_mb, 2)
