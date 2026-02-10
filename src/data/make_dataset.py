from pathlib import Path
import pandas as pd
import numpy as np

def load_california_csv(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_dir = Path(data_dir)
    csv_path = data_dir / "raw" / "california_housing.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run: python scripts/download_data.py")
    
    df = pd.read_csv(csv_path)
    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]

    X=df[feature_cols].to_numpy(dtype=np.float32)
    y=df[target_col].to_numpy(dtype=np.float32).reshape(-1,1)
    return X, y, feature_cols