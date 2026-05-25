"""Learned active cooperative perception (A6 / Phase 5): learn *whom to trust* and *what to communicate*.

The A2 consensus filter fuses every neighbour's deck fix by **inverse-variance** weighting — provably
optimal when the fixes are homogeneous, zero-mean Gaussian. Real cooperative perception is not: a drone can
produce a **confident outlier** (a reflection or a wrong marker decode reported with high confidence), and
bandwidth is finite so a drone cannot fuse every neighbour every step. This module adds the two pieces the
collaborative-perception literature (V2VNet / Where2comm) is built around:

1. **Learned graph-attention fusion** (`AttentionFusion`) — a permutation-invariant attention over the set
   of neighbour fixes whose output is a **convex combination of the actual fixes** (it cannot hallucinate a
   position). Its key input is each fix's **deviation from the group median** — a translation-invariant
   *agreement* feature that inverse-variance weighting structurally cannot use — so it learns to **reject
   confident outliers**, not just down-weight low-confidence ones. Trained supervised (fast, CPU).

2. **Value-of-information communication gating** (`select_broadcasters`) — under a **bandwidth budget** of
   B messages, pick the B broadcasters that most reduce the fused estimate's variance (greedy by expected
   precision over the comms graph). This is the "what to send / whom to query" decision: it reaches
   near-full accuracy at a fraction of the messages.

Honest expectation: on **homogeneous** Gaussian fixes the learned fusion only **ties** inverse-variance
(which is already optimal there — the same lesson as our MARL); the learned model **wins** exactly where
the modelling assumption breaks — **heterogeneous reliability / confident outliers** — which is the
realistic regime. Separate module (swarm stays separate); no ground truth at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FEATURES = 4   # [confidence, link_quality, deviation-from-median, range proxy]


@dataclass
class FusionResult:
    estimate: np.ndarray         # fused deck XY
    weights: np.ndarray          # per-message weight (sums to 1 over valid messages)


# --------------------------------------------------------------------- feature construction
def message_features(fixes: np.ndarray, conf: np.ndarray, link: np.ndarray,
                     rng_proxy: np.ndarray) -> np.ndarray:
    """Translation-invariant per-message features. ``fixes`` (M,2); conf/link/rng_proxy (M,).

    The decisive feature is **deviation from the group median** — how much a fix disagrees with the crowd —
    which lets a learner reject a confident outlier that inverse-variance weighting would trust."""
    med = np.median(fixes, axis=0)
    deviation = np.linalg.norm(fixes - med, axis=1)
    return np.stack([conf, link, deviation, rng_proxy], axis=1).astype(np.float32)


# --------------------------------------------------------------------- baselines (numpy)
def fuse_equal(fixes: np.ndarray) -> np.ndarray:
    return fixes.mean(axis=0)


def fuse_inverse_variance(fixes: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Inverse-variance fusion (the consensus weighting): weight ~ confidence (precision)."""
    w = np.clip(conf, 1e-3, None)
    w = w / w.sum()
    return (w[:, None] * fixes).sum(axis=0)


# --------------------------------------------------------------------- value-of-information gating
def select_broadcasters(conf: np.ndarray, link: np.ndarray, budget: int) -> np.ndarray:
    """Pick the ``budget`` message indices that most reduce fused variance (greedy by expected precision).

    Each message's precision contribution ~ confidence·link_quality; fusing the top-B maximizes the fused
    precision (= minimizes variance) under the budget. Returns the selected indices."""
    score = np.clip(conf, 1e-3, None) * np.clip(link, 1e-3, None)
    order = np.argsort(-score)
    return np.sort(order[:max(1, min(budget, len(score)))])


# --------------------------------------------------------------------- synthetic CP episodes
def sample_episode(rng: np.random.Generator, n_max: int = 8, outlier_p: float = 0.25):
    """One cooperative-perception scene: M neighbour fixes of a true deck, with heterogeneous reliability
    and occasional **confident outliers** (high reported confidence but a large bias). Returns
    (fixes, conf, link, rng_proxy, true_xy)."""
    m = int(rng.integers(3, n_max + 1))
    true = rng.uniform(-3, 3, size=2)
    conf = rng.uniform(0.2, 1.0, size=m)
    link = rng.uniform(0.5, 1.0, size=m)
    rng_proxy = rng.uniform(0.0, 1.0, size=m)
    noise_std = 0.05 + 0.5 * (1.0 - conf)            # low confidence -> noisier (calibrated case)
    fixes = true[None, :] + rng.normal(0, 1, (m, 2)) * noise_std[:, None]
    # inject confident outliers: a large bias but a deceptively HIGH reported confidence
    for i in range(m):
        if rng.random() < outlier_p:
            bias = rng.normal(0, 1, 2)
            bias *= (1.5 + rng.random()) / (np.linalg.norm(bias) + 1e-9)
            fixes[i] = true + bias
            conf[i] = rng.uniform(0.7, 1.0)          # confidently wrong
    return fixes.astype(np.float32), conf.astype(np.float32), link.astype(np.float32), \
        rng_proxy.astype(np.float32), true.astype(np.float32)


# --------------------------------------------------------------------- learned attention fusion (torch)
def _build_module():
    import torch
    from torch import nn

    class AttentionFusion(nn.Module):
        """Permutation-invariant attention over neighbour fixes -> convex combination of the actual fixes."""

        def __init__(self, hidden: int = 32):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(N_FEATURES, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))

        def forward(self, feats, fixes, mask):
            # feats (B,M,F), fixes (B,M,2), mask (B,M) in {0,1}
            s = self.score(feats).squeeze(-1)                       # (B,M)
            s = s.masked_fill(mask < 0.5, float("-inf"))
            w = torch.softmax(s, dim=1)                             # attention over the set
            est = (w.unsqueeze(-1) * fixes).sum(dim=1)              # convex combo -> cannot hallucinate
            return est, w

    return AttentionFusion


class LearnedFusion:
    """Wraps the trained attention net for numpy inference (drop-in alongside the consensus baseline)."""

    def __init__(self, model=None):
        self.model = model

    def fuse(self, fixes: np.ndarray, conf: np.ndarray, link: np.ndarray,
             rng_proxy: np.ndarray) -> np.ndarray:
        import torch
        feats = message_features(fixes, conf, link, rng_proxy)
        with torch.no_grad():
            est, _ = self.model(torch.from_numpy(feats[None]),
                                torch.from_numpy(fixes[None]),
                                torch.ones(1, len(fixes)))
        return est[0].numpy()

    def save(self, path: str) -> None:
        import torch
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        import torch
        model = _build_module()()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return cls(model)


def train_fusion(steps: int = 4000, batch: int = 128, lr: float = 2e-3, n_max: int = 8,
                 outlier_p: float = 0.25, seed: int = 0, device: str = "cpu"):
    """Supervised training of the attention fusion on synthetic heterogeneous CP scenes."""
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = _build_module()().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def make_batch():
        F = np.zeros((batch, n_max, N_FEATURES), np.float32)
        X = np.zeros((batch, n_max, 2), np.float32)
        Mk = np.zeros((batch, n_max), np.float32)
        Y = np.zeros((batch, 2), np.float32)
        for b in range(batch):
            fixes, conf, link, rp, true = sample_episode(rng, n_max, outlier_p)
            m = len(fixes)
            F[b, :m] = message_features(fixes, conf, link, rp)
            X[b, :m] = fixes
            Mk[b, :m] = 1.0
            Y[b] = true
        return (torch.from_numpy(F).to(device), torch.from_numpy(X).to(device),
                torch.from_numpy(Mk).to(device), torch.from_numpy(Y).to(device))

    model.train()
    for it in range(steps):
        F, X, Mk, Y = make_batch()
        est, _ = model(F, X, Mk)
        loss = ((est - Y) ** 2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, steps // 5) == 0:
            print(f"  step {it + 1}/{steps}  train RMSE {float(loss.item()) ** 0.5:.3f} m")
    model.eval()
    return LearnedFusion(model)


def evaluate(learned: "LearnedFusion", episodes: int = 4000, n_max: int = 8, outlier_p: float = 0.25,
             seed: int = 123) -> dict:
    """A/B the learned fusion vs equal-mean and inverse-variance on heterogeneous (outlier) scenes."""
    rng = np.random.default_rng(seed)
    errs = {"equal": [], "inverse_variance": [], "learned": []}
    for _ in range(episodes):
        fixes, conf, link, rp, true = sample_episode(rng, n_max, outlier_p)
        errs["equal"].append(np.linalg.norm(fuse_equal(fixes) - true))
        errs["inverse_variance"].append(np.linalg.norm(fuse_inverse_variance(fixes, conf) - true))
        errs["learned"].append(np.linalg.norm(learned.fuse(fixes, conf, link, rp) - true))
    return {k: float(np.mean(v)) for k, v in errs.items()}


def bandwidth_pareto(episodes: int = 3000, n_max: int = 8, outlier_p: float = 0.0, seed: int = 7) -> dict:
    """Accuracy vs communication budget: value-of-information selection vs random, fused inverse-variance.

    Outliers off here (this isolates the *bandwidth* question): how few messages does active selection need
    to match fusing everything? Returns {budget: {'active': err, 'random': err}, 'all': err}."""
    rng = np.random.default_rng(seed)
    budgets = [1, 2, 3, 4]
    out = {b: {"active": [], "random": []} for b in budgets}
    all_err = []
    for _ in range(episodes):
        fixes, conf, link, rp, true = sample_episode(rng, n_max, outlier_p)
        all_err.append(np.linalg.norm(fuse_inverse_variance(fixes, conf) - true))
        for b in budgets:
            sel = select_broadcasters(conf, link, b)
            out[b]["active"].append(np.linalg.norm(fuse_inverse_variance(fixes[sel], conf[sel]) - true))
            ridx = rng.choice(len(fixes), size=min(b, len(fixes)), replace=False)
            out[b]["random"].append(np.linalg.norm(fuse_inverse_variance(fixes[ridx], conf[ridx]) - true))
    res = {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in out.items()}
    res["all"] = float(np.mean(all_err))
    return res
