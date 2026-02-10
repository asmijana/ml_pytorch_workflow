import numpy as np
from sklearn.preprocessing import StandardScaler

class TabularPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fit = False

    def fit(self, X_train: np.ndarray) -> "TabularPreprocessor":
        self.scaler.fit(X_train)
        self.is_fit = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Preprocessor not fit. Call fit() on train data first.")
        return self.scaler.transform(X).astype(np.float32)
    
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)