"""
train_image_model.py — Train DualStreamCNN on pre-generated synthetic images.

Strategy: pre-generate all N samples once (~4 min CPU, ~400 MB RAM for 150k),
cache as uint8 arrays, then train 30 epochs at full GPU utilisation.

Usage (from repo root):
    .venv/Scripts/python -m fitengine.train_image_model ^
        --samples 150000 --epochs 30 --batch 64 --out models/dual_stream_cnn.pt

Flags:
    --compare    compare adj_acc vs ratio_regressor.pt after training
    --dry-run    single batch smoke test only
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

from fitengine.image_net import DualStreamCNN
from fitengine.heuristic.ratio_regressor import COLLAR_IDX, JACKET_IDX, TROUSER_IDX
from fitengine.render import render_body_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 128


# ---------------------------------------------------------------------------
# Pre-generation -> uint8 RAM cache
# ---------------------------------------------------------------------------

def prebuild_dataset(n: int, img_size: int, seed: int) -> dict:
    """
    Generate n synthetic bodies, render stick figures, return numpy arrays:
        imgs_front : uint8  [N, 1, H, W]
        imgs_side  : uint8  [N, 1, H, W]
        height_norm: float32[N, 1]
        labels     : int32  [N, 3]   (collar, jacket, trouser class indices)

    Memory: ~400 MB for N=150k at 128x128.
    """
    from fitengine.data_gen.synthetic import (
        SyntheticDataGenerator, _snap_collar, _snap_jacket, _snap_trouser,
    )
    from fitengine.measurements import BodyProxyMeasurements

    gen = SyntheticDataGenerator(seed=seed)
    rng = gen._rng

    imgs_f  = np.empty((n, 1, img_size, img_size), dtype=np.uint8)
    imgs_s  = np.empty((n, 1, img_size, img_size), dtype=np.uint8)
    hn_arr  = np.empty((n, 1),                     dtype=np.float32)
    lbl_arr = np.empty((n, 3),                     dtype=np.int32)

    t0    = time.time()
    saved = 0

    while saved < n:
        bodies = gen._sample_bodies(1)
        h, sh, ch, wa, ne, al, ins, hc, hipc, eb = bodies[0]

        hip_half    = (hipc / np.pi) / 2.0
        chest_depth = ch / np.pi
        h_over_w    = float(np.clip(rng.normal(1.78, 0.12), 1.4, 2.1))
        fill_f      = float(rng.uniform(0.55, 0.88))
        fill_s      = float(rng.uniform(0.25, 0.75))

        kp_f = gen._place_skeleton(h, sh,          hip_half,        fill_f, h_over_w, view="front",
                                   arm_length_cm=al, inseam_cm=ins, head_circ_cm=hc, elbow_breadth_cm=eb)
        kp_s = gen._place_skeleton(h, chest_depth, hip_half * 0.85, fill_s, h_over_w, view="side",
                                   arm_length_cm=al, inseam_cm=ins, head_circ_cm=hc, elbow_breadth_cm=eb)

        m = BodyProxyMeasurements.from_keypoints(kp_f, kp_s)
        if not m.valid:
            continue

        ci = COLLAR_IDX.get(_snap_collar(ne / 2.54))
        ji = JACKET_IDX.get(_snap_jacket(ch / 2.54))
        ti = TROUSER_IDX.get(_snap_trouser(wa / 2.54))
        if ci is None or ji is None or ti is None:
            continue

        img_f_raw = render_body_image(kp_f, img_size)   # float32 [1,H,W]
        img_s_raw = render_body_image(kp_s, img_size)

        # Front hflip augmentation (50%)
        if rng.random() < 0.5:
            img_f_raw = img_f_raw[:, :, ::-1].copy()

        imgs_f[saved]  = (img_f_raw * 255).astype(np.uint8)
        imgs_s[saved]  = (img_s_raw * 255).astype(np.uint8)
        hn_arr[saved]  = (h - 175.0) / 12.0
        lbl_arr[saved] = (ci, ji, ti)
        saved += 1

        if saved % 10_000 == 0:
            rate = saved / (time.time() - t0)
            logger.info("  Pre-generated %d / %d  (%.0f/s, ~%.1f min left)",
                        saved, n, rate, (n - saved) / rate / 60)

    logger.info("Pre-generation done: %d samples in %.0fs", n, time.time() - t0)
    return {"imgs_front": imgs_f, "imgs_side": imgs_s,
            "height_norm": hn_arr, "labels": lbl_arr}


# ---------------------------------------------------------------------------
# Disk cache for prebuild
# ---------------------------------------------------------------------------

def load_or_build(args, img_size: int) -> dict:
    """Return prebuild dict from disk cache if (n, seed) match; else rebuild + save."""
    cache = Path(args.cache)
    if cache.exists():
        logger.info("Found prebuild cache at %s — checking metadata …", cache)
        try:
            meta = np.load(cache, allow_pickle=False)
            if int(meta["n"]) == args.samples and int(meta["seed"]) == args.seed:
                logger.info("Cache hit (%d samples, seed=%d) — loading …",
                            args.samples, args.seed)
                t0 = time.time()
                # .copy() forces mmap → heap so Windows won't page-fault during training
                data = {k: meta[k].copy()
                        for k in ("imgs_front", "imgs_side", "height_norm", "labels")}
                logger.info("  Loaded %.1f MB into heap in %.0fs",
                            sum(v.nbytes for v in data.values()) / 1e6,
                            time.time() - t0)
                return data
            logger.info(
                "Cache mismatch (cached n=%s seed=%s, requested n=%d seed=%d) "
                "— rebuilding …", meta["n"], meta["seed"], args.samples, args.seed,
            )
        except Exception as exc:
            logger.warning("Could not read cache (%s) — rebuilding …", exc)

    data = prebuild_dataset(args.samples, img_size, args.seed)
    logger.info("Saving prebuild cache to %s …", cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, n=np.int64(args.samples), seed=np.int64(args.seed), **data)
    logger.info("  Cache saved (%.1f MB)", cache.stat().st_size / 1e6)
    return data


# ---------------------------------------------------------------------------
# Map-style dataset (kept for backwards-compat; training uses FastImageBatchLoader)
# ---------------------------------------------------------------------------

class PrebuiltDataset(Dataset):
    def __init__(self, data: dict) -> None:
        self.imgs_f = data["imgs_front"]
        self.imgs_s = data["imgs_side"]
        self.hn     = data["height_norm"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img_f = torch.from_numpy(self.imgs_f[idx].astype(np.float32)) / 255.0
        img_s = torch.from_numpy(self.imgs_s[idx].astype(np.float32)) / 255.0
        hn    = torch.from_numpy(self.hn[idx])
        lbl   = self.labels[idx].astype(np.int64)
        return img_f, img_s, hn, int(lbl[0]), int(lbl[1]), int(lbl[2])


# ---------------------------------------------------------------------------
# Fast batch loader — bypasses DataLoader/per-sample overhead on Windows
# ---------------------------------------------------------------------------

class FastImageBatchLoader:
    """
    Directly batch-indexes numpy uint8 arrays and pushes to GPU.
    Replaces DataLoader + PrebuiltDataset on Windows where num_workers > 0
    is broken for large in-memory datasets.

    Speedup:  1 fancy numpy-index call per batch  vs  N __getitem__ calls
              + zip(*batch) collation in the main thread.
    """

    def __init__(self, data: dict, indices: np.ndarray,
                 batch_size: int, device: str,
                 shuffle: bool = True, drop_last: bool = False) -> None:
        self.imgs_f    = data["imgs_front"]   # uint8  [N, 1, H, W]
        self.imgs_s    = data["imgs_side"]    # uint8  [N, 1, H, W]
        self.hn        = data["height_norm"]  # float32[N, 1]
        self.labels    = data["labels"]       # int32  [N, 3]
        self.idx       = indices
        self.bs        = batch_size
        self.device    = device
        self.shuffle   = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.idx)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        idx = self.idx.copy()
        if self.shuffle:
            np.random.shuffle(idx)
        n   = len(idx)
        end = (n // self.bs) * self.bs if self.drop_last else n
        dev = self.device
        for start in range(0, end, self.bs):
            b   = idx[start : start + self.bs]
            # batch-level conversion: uint8 → float32 in PyTorch (1 alloc, in-place div)
            f   = torch.from_numpy(self.imgs_f[b]).float().div_(255.0)
            s   = torch.from_numpy(self.imgs_s[b]).float().div_(255.0)
            h   = torch.from_numpy(self.hn[b])
            lbl = self.labels[b].astype(np.int64)
            yc  = torch.from_numpy(lbl[:, 0])
            yj  = torch.from_numpy(lbl[:, 1])
            yt  = torch.from_numpy(lbl[:, 2])
            if dev == "cuda":
                f  = f.to(dev,  non_blocking=True)
                s  = s.to(dev,  non_blocking=True)
                h  = h.to(dev,  non_blocking=True)
                yc = yc.to(dev, non_blocking=True)
                yj = yj.to(dev, non_blocking=True)
                yt = yt.to(dev, non_blocking=True)
            yield f, s, h, yc, yj, yt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def adjacent_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                      margin: int = 1) -> float:
    return ((logits.argmax(1) - targets).abs() <= margin).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device: str,
             amp: bool) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    adj = {"collar": 0.0, "jacket": 0.0, "trouser": 0.0}
    n = 0
    for img_f, img_s, hn, yc, yj, yt in loader:
        img_f = img_f.to(device); img_s = img_s.to(device)
        hn    = hn.to(device);    yc    = yc.to(device)
        yj    = yj.to(device);    yt    = yt.to(device)
        with autocast(device_type=device, enabled=amp):
            cl, jl, tl = model(img_f, img_s, hn)
            loss = criterion(cl, yc) + criterion(jl, yj) + criterion(tl, yt)
        total_loss    += loss.item()
        adj["collar"] += adjacent_accuracy(cl, yc)
        adj["jacket"] += adjacent_accuracy(jl, yj)
        adj["trouser"]+= adjacent_accuracy(tl, yt)
        n += 1
    nb = max(n, 1)
    return total_loss / nb, {k: v / nb for k, v in adj.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    logger.info("Device=%s  AMP=%s", device, use_amp)

    data    = load_or_build(args, IMG_SIZE)

    n       = len(data["labels"])
    n_val   = min(int(args.samples * 0.10), 1000)   # ≤10%, cap at 1000
    n_val   = max(n_val, 1)
    n_train = n - n_val

    # Use torch seeded permutation so val split is identical on resume
    g    = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    logger.info("Train=%d  Val=%d  (FastImageBatchLoader)", n_train, n_val)

    train_dl = FastImageBatchLoader(data, train_idx, args.batch,  device, shuffle=True,  drop_last=True)
    val_dl   = FastImageBatchLoader(data, val_idx,   args.batch * 4, device, shuffle=False, drop_last=False)

    model = DualStreamCNN(dropout=args.dropout).to(device)
    logger.info("DualStreamCNN  params=%s",
                f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_total  = args.epochs * math.ceil(n_train / args.batch)
    warmup_steps = min(300, steps_total // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(steps_total - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(device=device, enabled=use_amp)

    # -----------------------------------------------------------------------
    # Epoch resume — load if a resume checkpoint exists
    # -----------------------------------------------------------------------
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out_path.with_name(out_path.stem + "_resume.pt")
    best_adj    = 0.0
    start_epoch = 0

    if resume_path.exists():
        logger.info("Found resume checkpoint %s — restoring …", resume_path)
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt["epoch"])
            best_adj    = float(ckpt["best_adj"])
            logger.info("  Resuming from epoch %d (best adj=%.4f)", start_epoch, best_adj)
        except Exception as exc:
            logger.warning("Could not load resume checkpoint (%s) — starting fresh", exc)
            start_epoch = 0
            best_adj    = 0.0

    if start_epoch >= args.epochs:
        logger.info("Already completed %d/%d epochs — nothing to do.", start_epoch, args.epochs)
        return

    if args.dry_run:
        logger.info("Dry run — testing single batch …")
        for batch in train_dl:
            imgs_f, imgs_s, hn, yc, yj, yt = [t.to(device) for t in batch]
            with autocast(device_type=device, enabled=use_amp):
                cl, jl, tl = model(imgs_f, imgs_s, hn)
                loss = criterion(cl, yc) + criterion(jl, yj) + criterion(tl, yt)
            logger.info("Batch ok  loss=%.4f  shapes=(%s,%s,%s)",
                        loss.item(), cl.shape, jl.shape, tl.shape)
            break
        logger.info("Dry run passed.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        nb      = 0
        t0      = time.time()

        for imgs_f, imgs_s, hn, yc, yj, yt in train_dl:
            imgs_f = imgs_f.to(device, non_blocking=True)
            imgs_s = imgs_s.to(device, non_blocking=True)
            hn     = hn.to(device,     non_blocking=True)
            yc     = yc.to(device,     non_blocking=True)
            yj     = yj.to(device,     non_blocking=True)
            yt     = yt.to(device,     non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, enabled=use_amp):
                cl, jl, tl = model(imgs_f, imgs_s, hn)
                loss = criterion(cl, yc) + criterion(jl, yj) + criterion(tl, yt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += loss.item()
            nb      += 1
            step    += 1

        val_loss, val_adj = evaluate(model, val_dl, criterion, device, use_amp)
        mean_adj = float(np.mean(list(val_adj.values())))

        logger.info(
            "Epoch %02d/%02d  train=%.4f  val=%.4f  "
            "adj collar=%.3f jacket=%.3f trouser=%.3f  MEAN=%.3f  "
            "lr=%.2e  %.0fs",
            epoch, args.epochs, ep_loss / max(nb, 1), val_loss,
            val_adj["collar"], val_adj["jacket"], val_adj["trouser"],
            mean_adj, scheduler.get_last_lr()[0], time.time() - t0,
        )

        if mean_adj > best_adj:
            best_adj = mean_adj
            torch.save({"model": model.state_dict(), "val_adj_acc": val_adj,
                        "val_adj_acc_mean": mean_adj,
                        "epoch": epoch, "img_size": IMG_SIZE}, out_path)
            logger.info("  ✓ New best adj_acc=%.4f  saved → %s", best_adj, out_path)

        # Always save a resume checkpoint so training can be continued after a crash.
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "epoch":     epoch,
            "best_adj":  best_adj,
            "img_size":  IMG_SIZE,
        }, resume_path)

    logger.info("Done. Best adj_acc=%.4f", best_adj)

    if args.compare:
        _compare_with_ratio_regressor(args.seed, device)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _compare_with_ratio_regressor(seed: int, device: str) -> None:
    from fitengine.heuristic.ratio_regressor import (
        RatioRegressor, COLLAR_IDX, JACKET_IDX, TROUSER_IDX,
    )
    from fitengine.data_gen.synthetic import (
        SyntheticDataGenerator, _snap_collar, _snap_jacket, _snap_trouser,
    )
    from fitengine.measurements import BodyProxyMeasurements

    ckpt = Path("models/ratio_regressor.pt")
    if not ckpt.exists():
        logger.info("ratio_regressor.pt not found, skipping comparison")
        return

    rr = RatioRegressor()
    rr.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)["model"]
    )
    rr.to(device).eval()

    gen = SyntheticDataGenerator(seed=seed + 1)
    adjs: dict[str, list[bool]] = {"collar": [], "jacket": [], "trouser": []}

    for _ in range(5000):
        bodies = gen._sample_bodies(1)
        h, sh, ch, wa, ne, al, ins, hc, hipc, eb = bodies[0]
        rng      = gen._rng
        hip_half = (hipc / np.pi) / 2.0
        ho       = float(np.clip(rng.normal(1.78, .12), 1.4, 2.1))
        kp_f     = gen._place_skeleton(h, sh,       hip_half,        float(rng.uniform(.55,.88)), ho, "front",
                                       arm_length_cm=al, inseam_cm=ins, head_circ_cm=hc, elbow_breadth_cm=eb)
        kp_s     = gen._place_skeleton(h, ch/np.pi, hip_half * 0.85, float(rng.uniform(.25,.75)), ho, "side",
                                       arm_length_cm=al, inseam_cm=ins, head_circ_cm=hc, elbow_breadth_cm=eb)
        m        = BodyProxyMeasurements.from_keypoints(kp_f, kp_s)
        if not m.valid:
            continue

        feat = RatioRegressor.features_from_measurements(m, h)
        x    = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            cl, jl, tl = rr(x)

        tc = COLLAR_IDX.get(_snap_collar(ne / 2.54), 0)
        tj = JACKET_IDX.get(_snap_jacket(ch / 2.54), 0)
        tt = TROUSER_IDX.get(_snap_trouser(wa / 2.54), 0)

        adjs["collar"].append( abs(int(cl.argmax(1)) - tc) <= 1)
        adjs["jacket"].append( abs(int(jl.argmax(1)) - tj) <= 1)
        adjs["trouser"].append(abs(int(tl.argmax(1)) - tt) <= 1)

    logger.info(
        "RatioRegressor baseline  collar=%.3f jacket=%.3f trouser=%.3f MEAN=%.3f",
        np.mean(adjs["collar"]), np.mean(adjs["jacket"]), np.mean(adjs["trouser"]),
        np.mean([np.mean(v) for v in adjs.values()]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--samples",  type=int,   default=150_000)
    p.add_argument("--out",      default="models/dual_stream_cnn.pt")
    p.add_argument("--cache",    default="python-ml/src/data/cnn_prebuild.npz",
                   help="Path to save/load prebuild .npz cache  (keyed by --samples + --seed)")
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--workers",  type=int,   default=2,
                   help="DataLoader prefetch workers (0=off, 2 recommended on Windows)")
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--wd",       type=float, default=1e-4)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--compare",  action="store_true")
    p.add_argument("--dry-run",  action="store_true", dest="dry_run")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
