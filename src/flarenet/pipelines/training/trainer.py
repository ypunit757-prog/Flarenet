"""Training loop: mixup, weighted sampling, cosine LR, weight averaging, early stopping on validation skill."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from flarenet.core.config import Config
from flarenet.core.constants import N_CLASSES
from flarenet.core.schema import WindowSet
from flarenet.core.utils import get_logger, resolve_device, set_seed
from flarenet.models import ENCODER_MODULES, EnsembleModel, build_model, count_parameters
from flarenet.models.losses import FocalBCE, LabelSmoothedCE, class_weights
from flarenet.pipelines.evaluation.metrics import best_threshold, safe_auc
from flarenet.pipelines.training.checkpoint import load_checkpoint, save_checkpoint
from flarenet.pipelines.training.dataset import WindowDataset

log = get_logger(__name__)


def mixup_batch(X, y_fc, alpha: float = 0.15):
    """Mix a batch with a shuffled copy of itself. Returns mixed X/y_fc, shuffled index, lambda."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], lam * y_fc + (1 - lam) * y_fc[idx], idx, lam


class WeightEMA:
    """Exponential moving average of a model's weights and BatchNorm statistics.

    ``shadow`` is a frozen copy kept in eval mode; evaluate and checkpoint that instead of the live
    model. The decay warms up (``(1 + t) / (10 + t)``) so the first steps are not dominated by the
    random initialisation.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.steps = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for s, m in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(d).add_(m.detach(), alpha=1 - d)
        for s, m in zip(self.shadow.buffers(), model.buffers(), strict=True):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m, alpha=1 - d)
            else:
                s.copy_(m)


def selection_score(metric: str, tss: float, auc: float) -> float:
    """Number that early stopping maximises. AUC is smoother than TSS (no threshold), so choosing
    the epoch on it (alone or mixed with TSS) is less sensitive to validation noise."""
    if metric == "tss" or not np.isfinite(auc):
        return tss
    return auc if metric == "auc" else 0.5 * (tss + auc)


def forecast_sampling(y_fc: np.ndarray, tc) -> tuple[np.ndarray, float]:
    """Per-window sampling weight and the BCE ``pos_weight`` for the forecast head.

    ``fixed`` is the original rule (positives x ``positive_oversample``, capped class weight).
    ``auto`` balances the sampled batches to 50 / 50 whichever class is the rare one, which also
    fixes the original rule's habit of oversampling positives when they are already the majority.
    """
    pos_frac = float(np.mean(y_fc)) if len(y_fc) else 0.5
    if tc.forecast_balance == "auto" and 0.0 < pos_frac < 1.0:
        w = np.where(y_fc > 0.5, 0.5 / pos_frac, 0.5 / (1.0 - pos_frac))
        return w.astype(np.float32), 1.0
    w = np.where(y_fc > 0.5, tc.positive_oversample, 1.0).astype(np.float32)
    return w, min((1 - pos_frac) / max(pos_frac, 1e-3), 3.0)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, Xn: np.ndarray, device, batch_size: int = 512):
    """(nowcast logits, forecast probabilities) for already-normalised windows."""
    model.eval()
    now, fc = [], []
    for i in range(0, len(Xn), batch_size):
        n_l, f_l, _ = model(torch.from_numpy(Xn[i : i + batch_size]).to(device))
        now.append(n_l.cpu().numpy())
        fc.append(torch.sigmoid(f_l).cpu().numpy())
    return np.concatenate(now), np.concatenate(fc)


def tune_nowcast_bias(logits: np.ndarray, y: np.ndarray, grid=None, passes: int = 2) -> np.ndarray:
    """Per-class logit offsets (class 0 fixed at 0) that maximise nowcast macro-F1 on ``y``.

    Rare classes are under-predicted by a model trained on imbalanced labels; a small additive
    shift on the logits moves the decision boundaries without retraining. Coordinate ascent over a
    coarse grid keeps it from fitting the validation set too finely.
    """
    from sklearn.metrics import f1_score

    grid = np.arange(-3.0, 3.01, 0.25) if grid is None else grid
    k = logits.shape[1]
    bias = np.zeros(k)

    def score(b):
        return f1_score(y, (logits + b).argmax(1), average="macro")

    best = score(bias)
    for _ in range(passes):
        for c in range(1, k):
            for v in grid:
                trial = bias.copy()
                trial[c] = v
                sc = score(trial)
                if sc > best + 1e-9:
                    best, bias = sc, trial
    return bias.astype(np.float32)


def run_epoch(
    model,
    loader,
    optimizer,
    ce,
    bce,
    device,
    train: bool,
    fc_weight: float,
    grad_clip: float,
    frozen=None,
    on_step=None,
):
    """One pass over `loader`. Returns (loss, nowcast accuracy, forecast targets, forecast probs)."""
    model.train() if train else model.eval()
    if train and frozen:
        for m in frozen:
            m.eval()  # keep BatchNorm statistics of a frozen encoder fixed
    total_loss = correct = seen = 0
    all_t, all_p = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y_now, y_fc in loader:
            X, y_now, y_fc = X.to(device), y_now.to(device), y_fc.to(device)
            if train:
                Xm, y_fc_m, idx, lam = mixup_batch(X, y_fc)
                now_logits, fc_logits, _ = model(Xm)
                loss_now = lam * ce(now_logits, y_now) + (1 - lam) * ce(now_logits, y_now[idx])
                target_fc = y_fc_m
            else:
                now_logits, fc_logits, _ = model(X)
                loss_now = ce(now_logits, y_now)
                target_fc = y_fc
            loss = loss_now + fc_weight * bce(fc_logits, target_fc)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if on_step is not None:
                    on_step(model)
            total_loss += loss.item() * X.size(0)
            correct += (now_logits.argmax(1) == y_now).sum().item()
            seen += X.size(0)
            all_t.append(y_fc.detach().cpu().numpy())
            all_p.append(torch.sigmoid(fc_logits).detach().cpu().numpy())
    return total_loss / seen, correct / seen, np.concatenate(all_t), np.concatenate(all_p)


def _fit_member(
    cfg: Config, train: WindowSet, train_ds, val_ds, base, sample_w, pos_weight, device, seed: int
) -> dict:
    """Train one network (one ensemble member) and return it with its early-stopping record."""
    tc = cfg.training
    set_seed(seed)
    train_dl = DataLoader(
        train_ds, tc.batch_size,
        sampler=WeightedRandomSampler(sample_w, len(sample_w)), num_workers=0,
    )  # fmt: skip
    val_dl = DataLoader(val_ds, tc.batch_size, shuffle=False, num_workers=0)

    ce = LabelSmoothedCE(
        N_CLASSES, tc.label_smoothing,
        class_weights(train.y_now, N_CLASSES, tc.nowcast_weight_cap, tc.nowcast_weight_power).to(device),
    )  # fmt: skip
    bce = FocalBCE(tc.focal_gamma, torch.tensor(pos_weight).to(device))

    model = build_model(cfg.model, in_channels=len(train.channels)).to(device)
    if base is not None:
        # An ensemble checkpoint fine-tunes from its first member.
        src = base.model.members[0] if isinstance(base.model, EnsembleModel) else base.model
        model.load_state_dict(src.state_dict())
        log.info("Fine-tuning from %s", tc.finetune_from)
    frozen = None
    if tc.freeze_encoder:
        frozen = [getattr(model, n) for n in ENCODER_MODULES]
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model: %s parameters (%s trainable) | device=%s | train=%d val=%d windows",
        f"{count_parameters(model):,}", f"{n_train_params:,}", device, len(train_ds), len(val_ds),
    )  # fmt: skip
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tc.lr, weight_decay=tc.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(tc.epochs, 1), eta_min=tc.min_lr
    )
    ema = WeightEMA(model, tc.ema_decay) if tc.ema_decay > 0 else None
    eval_model = ema.shadow if ema is not None else model

    best_score, best_tss, best_thr, bad = -2.0, -2.0, 0.5, 0
    best_state = None
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_tss", "val_auc")}
    t_start = time.time()
    for epoch in range(1, tc.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(
            model, train_dl, optimizer, ce, bce, device, True,
            tc.forecast_loss_weight, tc.grad_clip, frozen,
            ema.update if ema is not None else None,
        )  # fmt: skip
        va_loss, va_acc, vt, vp = run_epoch(
            eval_model, val_dl, optimizer, ce, bce, device, False,
            tc.forecast_loss_weight, tc.grad_clip,
        )  # fmt: skip
        scheduler.step()
        thr, v_tss = best_threshold(vt, vp, smooth=tc.threshold_smooth)
        v_auc = safe_auc(vt, vp)
        score = selection_score(tc.select_metric, v_tss, v_auc)
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc, v_tss, v_auc), strict=True):
            hist[k].append(float(v))
        log.info(
            "Epoch %3d | loss %.4f/%.4f | acc %.3f/%.3f | val AUC %.3f | val TSS@%.2f %.3f | %.0fs",
            epoch, tr_loss, va_loss, tr_acc, va_acc, v_auc, thr, v_tss, time.time() - t_start,
        )  # fmt: skip
        if score > best_score:
            best_score, best_tss, best_thr, bad = score, v_tss, thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.patience:
                log.info(
                    "Early stopping at epoch %d (best val %s %.4f)",
                    epoch,
                    tc.select_metric,
                    best_score,
                )
                break

    if best_state is not None:
        eval_model.load_state_dict(best_state)
    eval_model.eval()
    return {
        "model": eval_model, "history": hist, "threshold": best_thr, "best_val_tss": best_tss,
        "n_trainable": n_train_params,
    }  # fmt: skip


def train_model(
    cfg: Config,
    train: WindowSet,
    val: WindowSet,
    out_dir: str | Path,
    source_weights: dict[str, float] | None = None,
) -> dict:
    """Train on ``train``, early-stop on ``val``, save ``checkpoint.pt``.

    With ``training.finetune_from`` the weights start from an existing checkpoint (its channel
    list must match); ``freeze_encoder`` then trains only the two heads. With
    ``training.ensemble_size > 1`` several networks are trained from different seeds and their
    probabilities averaged. The alert threshold (and, optionally, the nowcast class offsets) are
    chosen on the validation windows with the final model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg.training
    device = resolve_device(tc.device)
    set_seed(cfg.seed)

    base = load_checkpoint(tc.finetune_from, device) if tc.finetune_from else None
    if base is not None and base.channels != train.channels:
        raise ValueError(
            f"Fine-tuning needs identical feature channels: checkpoint has {base.channels}, "
            f"this run has {train.channels}"
        )
    mu_sg = (base.mu, base.sg) if base is not None and not tc.refit_normaliser else (None, None)

    train_ds = WindowDataset(
        train.X,
        train.y_now,
        train.y_fc,
        *mu_sg,
        augment=True,
        channels=list(train.channels),
        level_jitter=tc.aug_level_jitter,
        hxr_dropout=tc.aug_hxr_dropout,
        noise_max=tc.aug_noise_max,
    )
    val_ds = WindowDataset(val.X, val.y_now, val.y_fc, mu=train_ds.mu, sg=train_ds.sg)

    sw = source_weights or {}
    per_source = np.array([sw.get(n, 1.0) for n in train.source_names], dtype=np.float32)
    w_fc, pos_weight = forecast_sampling(train.y_fc, tc)
    sample_w = torch.from_numpy(w_fc * per_source[train.source_idx])

    members = []
    for m in range(tc.ensemble_size):
        if tc.ensemble_size > 1:
            log.info("--- ensemble member %d / %d", m + 1, tc.ensemble_size)
        members.append(
            _fit_member(
                cfg,
                train,
                train_ds,
                val_ds,
                base,
                sample_w,
                pos_weight,
                device,
                cfg.seed + 1000 * m,
            )  # fmt: skip
        )
    first = members[0]
    if tc.ensemble_size == 1:
        model, best_thr, best_tss = first["model"], first["threshold"], first["best_val_tss"]
        now_val = None
    else:
        model = EnsembleModel([mm["model"] for mm in members]).to(device).eval()
        now_val, fc_val = predict_logits(model, val_ds.X, device)
        best_thr, best_tss = best_threshold(val.y_fc, fc_val, smooth=tc.threshold_smooth)

    bias = None
    if tc.nowcast_bias_tuning:
        if now_val is None:
            now_val, _ = predict_logits(model, val_ds.X, device)
        bias = tune_nowcast_bias(now_val, val.y_now)
        log.info("Nowcast class offsets tuned on validation: %s", np.round(bias, 2).tolist())

    save_checkpoint(
        out_dir / "checkpoint.pt", model, train_ds.mu, train_ds.sg, best_thr, cfg,
        train.channels, [n for n in train.source_names if n != "_"], bias,
    )  # fmt: skip
    log.info("Best val TSS %.4f at alert threshold %.2f", best_tss, best_thr)
    return {
        "history": first["history"], "threshold": best_thr, "best_val_tss": best_tss,
        "n_parameters": count_parameters(model), "n_trainable": first["n_trainable"] * len(members),
        "now_bias": None if bias is None else bias.tolist(),
        "member_histories": [mm["history"] for mm in members],
    }  # fmt: skip
