"""
FitEngine trainer — Phase 2 training loop.

Loss formula:
    L = 0.4 * CE_collar + 0.4 * CE_jacket + 0.2 * CE_trouser + 0.1 * MSE_beta

Optimiser:    AdamW, lr=1e-4, weight_decay=1e-2
Scheduler:    CosineAnnealingLR with 5-epoch linear warmup
Batch size:   256  (gradient accumulation 4 → effective 1024)
Precision:    AMP (torch.cuda.amp)

Checkpointing:
    - best.pt saved on jacket_adjacent_accuracy (NOT val loss)
    - resume.pt overwritten every epoch

Training phases:
    Phase 1 training (Gate 3):  30 epochs, synthetic data only
    Phase 2 fine-tune (Gate 4): 20 epochs, lr=1e-5, 70% synthetic + 30% Blender

# TODO (Phase 2, Month 2): implement after PoseDataset is live and
#       10k smoke test passes Gate 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class FitEngineTrainer:
    """
    Training orchestrator for DualViewRegressor + FitEngineClassifierBundle.

    # TODO (Phase 2, Month 2)
    """

    def __init__(
        self,
        train_h5: str | Path,
        val_h5: str | Path,
        checkpoint_dir: str | Path = "checkpoints",
        epochs: int = 30,
        batch_size: int = 256,
        grad_accum_steps: int = 4,
        base_lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 5,
        loss_weights: Optional[dict] = None,
        amp: bool = True,
        seed: int = 42,
    ) -> None:
        raise NotImplementedError(
            "FitEngineTrainer is a Phase 2 implementation target. "
            "Implement after Gate 3 passes (jacket_adj_acc >= 0.70 on 10k set)."
        )

    def train(self) -> None:
        raise NotImplementedError

    def validate(self, epoch: int) -> dict:
        """Returns {'collar_adj_acc', 'jacket_adj_acc', 'trouser_adj_acc', 'beta_mse'}."""
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError
