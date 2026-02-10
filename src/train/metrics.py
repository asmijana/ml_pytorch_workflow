import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true-y_pred)**2))
    ss_tot = float(np.sum((y_true-np.mean(y_true))**2))
    return 1.0 - (ss_res/(ss_tot + 1e-12))


