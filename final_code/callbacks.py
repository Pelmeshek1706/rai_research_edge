"\"\"\"Lightning callbacks used by the online adversarial pipeline.\"\"\""
from __future__ import annotations

import lightning as L


class PhaseValMetricsCallback(L.Callback):
    """Collects validation accuracy per epoch and summarizes it by phase."""

    def __init__(self, phaseA_epochs: int):
        """Save the number of Phase A epochs so we can cut the history into two halves."""
        super().__init__()
        self.phaseA_epochs = int(phaseA_epochs)
        self.val_acc_per_epoch = []

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        """Hook called by Lightning after each validation epoch to record accuracy."""
        acc = trainer.callback_metrics.get("val/acc")
        if acc is None:
            return
        try:
            acc_val = float(acc.detach().cpu())
        except Exception:
            acc_val = float(acc)
        epoch = trainer.current_epoch
        self.val_acc_per_epoch.append((epoch, acc_val))

    def compute_phase_metrics(self) -> dict:
        """Return best/last accuracy for Phase A and Phase B so the runner can log them."""
        phaseA_vals = [v for e, v in self.val_acc_per_epoch if e < self.phaseA_epochs]
        phaseB_vals = [v for e, v in self.val_acc_per_epoch if e >= self.phaseA_epochs]

        out = {}
        if phaseA_vals:
            out["phaseA_val_acc_last"] = phaseA_vals[-1]
            out["phaseA_val_acc_best"] = max(phaseA_vals)
        if phaseB_vals:
            out["phaseB_val_acc_last"] = phaseB_vals[-1]
            out["phaseB_val_acc_best"] = max(phaseB_vals)
        return out


__all__ = ["PhaseValMetricsCallback"]
