from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.base_model import BaseModel
from settings import LOGISTIC_REGRESSION_PARAMS


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        # Merge default params with any overrides
        params = {**LOGISTIC_REGRESSION_PARAMS, **kwargs}
        self.model = LogisticRegression(**params)
        self.fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]

    def calculate_auc_score(self, X, y):
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating AUC score")
        preds = self.predict(X)
        return roc_auc_score(y, preds)

    def get_coefficients(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.model.coef_
