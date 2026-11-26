"""Learning rates indexed by successful optimizer updates, and gradient diagnostics."""
from __future__ import annotations

import math

import torch


class UpdateLRScheduler:
    """Prepare update n+1 after n successful updates; update one uses the start LR."""

    @staticmethod
    def validate_options(options, peak_lr: float, total_steps: int) -> None:
        if set(options) - {"type", "warmup_steps", "warmup_start_factor", "min_lr"}:
            raise ValueError("Unknown training.scheduler key")
        kind = options.get("type", "constant")
        if kind not in {"constant", "cosine"}:
            raise ValueError("scheduler.type must be constant or cosine")
        if kind == "constant":
            return
        warmup = options.get("warmup_steps", 500)
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("scheduler.warmup_steps must be a nonnegative integer")
        if warmup == 1 or warmup >= total_steps or total_steps < 2:
            raise ValueError("Cosine requires warmup_steps=0 or 2..max_steps-1 and max_steps>=2")
        factor, minimum = options.get("warmup_start_factor", .01), options.get("min_lr", 1e-5)
        if isinstance(factor, bool) or not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("scheduler.warmup_start_factor must be in (0,1]")
        if isinstance(minimum, bool) or not math.isfinite(minimum) or not 0 <= minimum <= peak_lr:
            raise ValueError("scheduler.min_lr must be in [0, learning_rate]")

    def __init__(self, optimizer, options, total_steps: int, peak_lr: float):
        options = dict(options or {"type": "constant"})
        self.validate_options(options, peak_lr, total_steps)
        self.optimizer = optimizer
        self.spec = dict(type=options.get("type", "constant"), total_steps=total_steps, peak_lr=peak_lr)
        if self.spec["type"] == "cosine":
            self.spec.update(warmup_steps=options.get("warmup_steps", 500),
                             warmup_start_factor=options.get("warmup_start_factor", .01),
                             min_lr=options.get("min_lr", 1e-5))
        self.completed_updates = 0
        self._prepare()

    def lr_for_update(self, update: int) -> float:
        s = self.spec
        if s["type"] == "constant":
            return s["peak_lr"]
        update = max(1, min(update, s["total_steps"]))
        warmup = s["warmup_steps"]
        if warmup and update <= warmup:
            factor = s["warmup_start_factor"]
            return s["peak_lr"] * (factor + (1-factor) * (update-1)/(warmup-1))
        start = warmup if warmup else 1
        fraction = (update-start) / (s["total_steps"]-start)
        return s["min_lr"] + .5*(s["peak_lr"]-s["min_lr"])*(1+math.cos(math.pi*fraction))

    def phase(self, update: int) -> str:
        if self.spec["type"] == "constant":
            return "constant"
        return "warmup" if update <= self.spec["warmup_steps"] else "cosine"

    def _prepare(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_for_update(self.completed_updates + 1)

    def step(self) -> None:
        self.completed_updates += 1
        self._prepare()

    def state_dict(self) -> dict:
        return {"spec": dict(self.spec), "completed_updates": self.completed_updates}

    def restore(self, state: dict | None, completed_updates: int) -> None:
        if state is None:
            if self.spec["type"] != "constant":
                raise ValueError("Cannot resume cosine without saved scheduler state; use a fresh run")
        elif state["spec"] != self.spec or int(state["completed_updates"]) != completed_updates:
            raise ValueError("Resume scheduler settings/step differ from checkpoint; preserve the LR trajectory")
        self.completed_updates = completed_updates
        self._prepare()


def optimizer_update(optimizer, scaler, scheduler: UpdateLRScheduler) -> bool:
    """A GradScaler overflow reduces its scale and skips optimizer.step()."""
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    if updated:
        scheduler.step()
    return updated


def gradient_metrics(model, max_norm: float, *, module_norms: bool = True) -> dict[str, torch.Tensor]:
    """Called after unscale, before one global clip. No extra backward passes."""
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    before = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    coefficient = (max_norm / (before + 1e-6)).clamp(max=1.)
    # Measure actual post-clip gradients (also exposes non-finite gradients).
    def norm(params):
        norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
        return torch.stack(norms).norm() if norms else before.new_zeros(())
    metrics = {"train/gradient_norm": before.detach(),
               "train/gradient_norm_before_clipping": before.detach(),
               "train/gradient_norm_after_clipping": norm(parameters),
               "train/gradient_clipping_coefficient": coefficient.detach(),
               "train/gradient_clipping_fraction": (before > max_norm).float()}
    # Parameter-owning modules give non-overlapping, architecture-independent norms.
    for name, module in model.named_modules() if module_norms else ():
        params = [p for p in module.parameters(recurse=False) if p.grad is not None]
        if params:
            post = norm(params)
            metrics[f"train/gradient_norm_module/{name}"] = post / coefficient
    return metrics
