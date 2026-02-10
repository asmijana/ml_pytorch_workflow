from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

def main():
    out_dir = Path("data") / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = fetch_california_housing(as_frame=True)
    df = ds.frame #includes features + target column

    out_path = out_dir / "california_housing.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved : {out_path} | shape={df.shape}")

if __name__ == "__main__":
    main()