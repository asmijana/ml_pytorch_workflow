import argparse
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import seed_everything
from src.utils.io import save_json, save_checkpoint, load_checkpoint
from src.data.make_dataset import load_california_csv
from src.data.split import make_splits
from src.data.preprocess import TabularPreprocessor
from src.models.mlp import MLPRegressor
from src.train.engine import train_one_epoch, evaluate
from src.train.callbacks import EarlyStopping

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else"cpu")
    return torch.device(device_str)

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers : int):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    preds = model(xb).detach().cpu().numpy()
    return preds

def plot_loss_curves(metrics_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    
def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))

    plt.figure()
    plt.scatter(yt, yp, s=10, alpha=0.5)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    exp_name = cfg["experiment"]["name"]
    out_root = ensure_dir(Path("runs") / exp_name)
    ensure_dir(out_root / "checkpoints")

    seed_everything(cfg["runtime"]["seed"])
    device = resolve_device(cfg["runtime"]["device"])
    use_amp = bool(cfg["runtime"].get("use_amp", False))

    #Load new data
    data_dir = cfg["data"]["out_dir"]
    X, y, feature_cols = load_california_csv(data_dir)

    #Splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(
        X, y,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
    )

    #Preprocess (fit on train only)
    prep = TabularPreprocessor()
    X_train_s = prep.fit_transform(X_train)
    X_val_s = prep.transform(X_val)
    X_test_s = prep.transform(X_test)

    #Save feature names + split sizes
    save_json({
        "features": feature_cols,
        "n_train": int(X_train_s.shape[0]),
        "n_val": int(X_val_s.shape[0]),
        "n_test": int(X_test_s.shape[0]),
        "device": str(device),
    }, out_root / "data_summary.json")

    #Loaders
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_loader = make_loader(X_train_s, y_train, bs, True, nw)
    val_loader = make_loader(X_val_s, y_val, bs, False, nw)
    test_loader = make_loader(X_test_s, y_test, bs, False, nw)

    #Model
    model = MLPRegressor(
        in_dim = X_train_s.shape[1],
        hidden_dims = cfg["model"]["hidden_dims"],
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = float(cfg["train"]["lr"]),
        weight_decay = float(cfg["train"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    early = EarlyStopping(
        patience = int(cfg["train"]["patience"]),
        min_delta = float(cfg["train"]["min_delta"]),
    )

    #Metrics logging
    metrics_path = out_root / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse", "val_mae", "val_r2"])

    best_ckpt = out_root / "checkpoints" / "best.pt"

    max_epochs = int(cfg["train"]["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_res = evaluate(model, val_loader, loss_fn, device)

        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_res.loss, val_res.rmse, val_res.mae, val_res.r2])

        improved = early.step(val_res.loss)
        if improved:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_res.loss,
                "config": cfg,
            }, best_ckpt)
        
        print(
            f"Epoch {epoch:03d} | train_loss = {train_loss:.4f} |"
            f"val_loss = {val_res.loss:.4f} val_rmse = {val_res.rmse:.4f} val_mae = {val_res.mae:.4f} val_r2 = {val_res.r2:.4f} "
            f"{'(best)' if improved else ''}"
        )

        if early.should_stop:
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss = {early.best:.4f}")
            break

    #Plot train and val loss curves from metrics.csv
    plot_loss_curves(metrics_path, out_root / "loss_curve.png")

    #Load best checkpoint and run final test exactly once
    ckpt = load_checkpoint(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_res = evaluate(model, test_loader, loss_fn, device)

    summary = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["val_loss"]),
        "test_loss": float(test_res.loss),
        "test_rmse": float(test_res.rmse),
        "test_mae": float(test_res.mae),
        "test_r2": float(test_res.r2),
    }
    save_json(summary, out_root / "test_summary.json")
    print("Final test:", summary)

    #Plot parity plots
    yhat_train = predict(model, X_train_s, device)
    yhat_val = predict(model, X_val_s, device)
    yhat_test = predict(model, X_test_s, device)

    plot_parity(y_train, yhat_train, "Parity plot: Train", out_root / "parity_train.png")
    plot_parity(y_val, yhat_val, "Parity plot: Validation", out_root / "parity_val.png")
    plot_parity(y_test, yhat_test, "Parity plot: Test", out_root / "parity_test.png")

if __name__ == "__main__":
    main()