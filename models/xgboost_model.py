from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from models.base_model import BaseModel
from settings import XGBOOST_PARAMS


class XGBoostModel(BaseModel):
    """XGBoost classifier for xG prediction."""

    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.

        Args:
            **kwargs: Additional parameters to override defaults
        """
        # Merge default params with any overrides
        params = {**XGBOOST_PARAMS, **kwargs}
        self.model = XGBClassifier(**params)
        self.fitted = False

    def fit(self, X, y):
        """Train the XGBoost model."""
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """
        Predict probabilities for the positive class.

        Args:
            X: Features to predict on

        Returns:
            Predicted probabilities for goal (positive class)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]

    def calculate_auc_score(self, X, y):
        """Calculate AUC score."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating AUC score")
        preds = self.predict(X)
        return roc_auc_score(y, preds)

    def get_feature_importance(self):
        """
        Get feature importance scores.

        Returns:
            Feature importance array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing feature importance")
        return self.model.feature_importances_
