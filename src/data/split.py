import numpy as np
from sklearn.model_selection import train_test_split
def make_splits(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float, random_state: int):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_frac_of_trainval = val_size/(1.0-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_frac_of_trainval, random_state=random_state)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)