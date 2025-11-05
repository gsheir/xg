class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method")

    def calculate_auc_score(self, X, y):
        raise NotImplementedError("Subclasses should implement this method")
