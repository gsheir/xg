class BaseModel:
    """Base class for all models with common interface."""

    def fit(self, X, y):
        """
        Train the model on the given data.

        Args:
            X: Training features
            y: Training targets
        """
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, X):
        """
        Make predictions on the given data.

        Args:
            X: Features to predict on

        Returns:
            Predicted probabilities for positive class
        """
        raise NotImplementedError("Subclasses should implement this method")

    def calculate_auc_score(self, X, y):
        """
        Calculate AUC score on the given data.

        Args:
            X: Features
            y: True targets

        Returns:
            AUC score
        """
        raise NotImplementedError("Subclasses should implement this method")
