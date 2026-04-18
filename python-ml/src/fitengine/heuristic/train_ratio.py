"""
train_ratio.py — GPU training script for RatioRegressor.

Usage (from repo root):
    .venv/Scripts/python -m fitengine.heuristic.train_ratio \
        --data    python-ml/src/data/synthetic_train.jsonl \
        --out     models/ratio_regressor.pt \
        --epochs  25 \
        --batch   2048 \
        --lr      3e-4

Features:
    * Mixed-precision (AMP) on CUDA, falls back to float32 on CPU
    * Cosine LR schedule with warm-up
    * Adjacent-accuracy metric (within 1 class = acceptable)
    * Checkpoint keeps best validation adjacent-accuracy
    * Rich console progress via tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split

from fitengine.heuristic.ratio_regressor import (
    COLLAR_CLASSES,
    COLLAR_IDX,
    INPUT_DIM,
    JACKET_CLASSES,
    JACKET_IDX,
    TROUSER_CLASSES,
    TROUSER_IDX,
    RatioRegressor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — pre-load everything into a single GPU TensorDataset
# ---------------------------------------------------------------------------

# elbow_width_ratio & arm_span_ratio are excluded: they reflect random arm-angle
# variation, not body size → pure noise for collar/jacket/trouser prediction.
_FEATURE_KEYS = [
    "shoulder_width_ratio",
    "chest_width_ratio",
    "hip_width_ratio",
    "torso_depth_ratio",
    "shoulder_depth_ratio",
    "torso_height_px",
    "leg_length_ratio",
    "inseam_ratio",
    "head_width_ratio",
    "hip_shoulder_taper",
    "side_hip_depth_ratio",
    "height_norm",
]


def load_jsonl_to_tensors(
    jsonl_path: str | Path,
    device: torch.device,
) -> TensorDataset:
    """
    Parse the JSONL once, build contiguous numpy arrays, move to device in one
    H2D transfer.  All subsequent DataLoader batches are pure GPU slices — no
    per-batch CPU→GPU copies, no Python __getitem__ overhead.
    """
    xs:       list[np.ndarray] = []
    y_collar:  list[int]       = []
    y_jacket:  list[int]       = []
    y_trouser: list[int]       = []
    bad = 0

    path = Path(jsonl_path)
    logger.info("Loading dataset from %s …", path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ci = COLLAR_IDX.get(str(row.get("collar", "")))
            ji = JACKET_IDX.get(str(row.get("jacket", "")))
            ti = TROUSER_IDX.get(str(row.get("trouser_waist", "")))
            if ci is None or ji is None or ti is None:
                bad += 1
                continue
            feat = np.array(
                [float(row.get(k, 0.0)) for k in _FEATURE_KEYS],
                dtype=np.float32,
            )
            feat = np.clip(feat, -4.0, 4.0)
            xs.append(feat)
            y_collar.append(ci)
            y_jacket.append(ji)
            y_trouser.append(ti)

    if bad:
        logger.warning("Skipped %d rows with unmapped labels", bad)
    n = len(xs)
    logger.info("Loaded %d samples — transferring to %s …", n, device)

    X  = torch.tensor(np.array(xs),       dtype=torch.float32).to(device)
    yc = torch.tensor(y_collar,           dtype=torch.long).to(device)
    yj = torch.tensor(y_jacket,           dtype=torch.long).to(device)
    yt = torch.tensor(y_trouser,          dtype=torch.long).to(device)
    logger.info("Dataset resident on %s  (X shape %s)", device, tuple(X.shape))
    return TensorDataset(X, yc, yj, yt)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def adjacent_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, margin: int = 1
) -> float:
    """Fraction of predictions within `margin` classes of the true class."""
    pred  = logits.argmax(dim=1)
    diff  = (pred - targets).abs()
    return (diff <= margin).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    adj_acc    = {"collar": 0.0, "jacket": 0.0, "trouser": 0.0}

    for xs, yc, yj, yt in loader:
        with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
            cl, jl, tl = model(xs)
            loss = criterion(cl, yc) + criterion(jl, yj) + criterion(tl, yt)

        total_loss          += loss.item()
        adj_acc["collar"]   += adjacent_accuracy(cl, yc)
        adj_acc["jacket"]   += adjacent_accuracy(jl, yj)
        adj_acc["trouser"]  += adjacent_accuracy(tl, yt)
        n_batches           += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in adj_acc.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s  AMP=%s", device, use_amp)

    # ── Data — pre-load everything to GPU once ──
    full_ds = load_jsonl_to_tensors(args.data, device)
    n_total = len(full_ds)
    n_val   = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("Train=%d  Val=%d", n_train, n_val)

    # num_workers=0 required — CUDA tensors can't be shared across worker processes
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 4, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ──
    model     = RatioRegressor(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    # Cosine LR with linear warm-up
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = min(500, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(device="cuda" if use_amp else "cpu", enabled=use_amp)

    # ── Training ──
    out_path     = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_adj_acc = 0.0
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for xs, yc, yj, yt in train_loader:
            # tensors already on device — no .to() needed
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                cl, jl, tl = model(xs)
                loss = criterion(cl, yc) + criterion(jl, yj) + criterion(tl, yt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

        elapsed = time.time() - t0
        val_loss, val_adj = evaluate(model, val_loader, criterion, device, use_amp)
        mean_adj_acc = float(np.mean(list(val_adj.values())))

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  "
            "adj_acc collar=%.3f jacket=%.3f trouser=%.3f  MEAN=%.3f  "
            "lr=%.2e  time=%.0fs",
            epoch, args.epochs,
            epoch_loss / max(len(train_loader), 1),
            val_loss,
            val_adj["collar"], val_adj["jacket"], val_adj["trouser"],
            mean_adj_acc,
            scheduler.get_last_lr()[0],
            elapsed,
        )

        if mean_adj_acc > best_adj_acc:
            best_adj_acc = mean_adj_acc
            torch.save(
                {
                    "model":      model.state_dict(),
                    "val_adj_acc": val_adj,
                    "epoch":      epoch,
                },
                out_path,
            )
            logger.info("  ✓ Saved new best checkpoint (adj_acc=%.4f)", best_adj_acc)

    logger.info("Training complete. Best adj_acc=%.4f  Checkpoint: %s", best_adj_acc, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train RatioRegressor on synthetic JSONL data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      default="python-ml/src/data/synthetic_train.jsonl",
                   help="Path to training JSONL file")
    p.add_argument("--out",       default="models/ratio_regressor.pt",
                   help="Output checkpoint path")
    p.add_argument("--epochs",    type=int,   default=25)
    p.add_argument("--batch",     type=int,   default=4096)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--wd",        type=float, default=1e-4,    help="Weight decay")
    p.add_argument("--dropout",   type=float, default=0.3)
    p.add_argument("--val-split", type=float, default=0.05,   dest="val_split")
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
