from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import rmse, mae, r2

@dataclass
class EvalResult:
    loss : float
    rmse : float
    mae : float
    r2 : float

def train_one_epoch(model, loader: DataLoader, optimizer, loss_fn, device):
    model.train() #Enable training behavior
    
    running = 0.0
    n = 0

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        #Clear old gradients
        optimizer.zero_grad(set_to_none=True)

        #Forward pass
        preds = model(xb)

        #Compute loss
        loss = loss_fn(preds, yb)

        #Backpropagation
        loss.backward()

        #Update parameters
        optimizer.step()

        #Track average loss over all samples
        bs = xb.size(0)
        running += loss.item() * bs
        n += bs

    return running / max(n, 1)

@torch.no_grad()
def evaluate(model, loader: DataLoader, loss_fn, device) -> EvalResult:
    model.eval()
    losses = []
    y_true_all = []
    y_pred_all = []

    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)

        losses.append(loss.item())
        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    return EvalResult(
        loss = float(np.mean(losses)),
        rmse = rmse(y_true, y_pred),
        mae = mae(y_true, y_pred),
        r2 = r2(y_true, y_pred),
    )

