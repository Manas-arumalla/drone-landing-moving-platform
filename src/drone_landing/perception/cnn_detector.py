"""Learned CNN deck-pose detector (B3) — the proper markerless perception.

The classical markerless fallback ([markerless.py]) segments the bright pad and back-projects its
centroid; it works when the pad is a clean saturated blob but is brittle to clutter, glare, partial
occlusion, or a non-white deck. A small **CNN** trained on rendered deck images (labels are *free* from
the simulator — the true deck pose projects to a known pixel) learns to localize the deck centre directly,
and keeps tracking when ArUco decoding fails.

Design — **learn perception, keep the geometry analytic**: the CNN regresses the deck-centre **pixel**
(normalised to [-1, 1]) + a **visibility** logit; that pixel is then back-projected through the same
pinhole + rangefinder geometry as the classical tracker to a relative-XY fix. Separating the learned
2-D detection from the known projection is robust (no need to learn camera geometry) and makes the
detector a **drop-in** for the markerless fallback / EKF fusion. GPU-trained, CPU-inferable.

This module defines the network + the detector wrapper (both CPU-testable now); dataset generation
(`generate_dataset`, renders labelled frames from the sim) and training (`train`, GPU) are below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.perception.camera import CameraModel

INPUT_HW = 96        # network input resolution (RGB) — large enough to keep the bright pad after resize
IN_CH = 3            # RGB (grayscale at 64px washed the small pad out -> no learnable signal)


def build_net():
    """Lazily construct the torch CNN class (so importing this module needs no torch)."""
    import torch.nn as nn

    import torch

    class SpatialSoftmax(nn.Module):
        """Soft-argmax over each feature-channel's activation map -> its expected (x,y) coordinate.

        The proven, sample-efficient design for regressing the *location* of a salient feature (Levine et
        al. 2016): it computes a differentiable centroid, so the network learns 'where the deck is' fast
        instead of having to discover position from a flattened map."""

        def forward(self, feat):
            b, c, h, w = feat.shape
            xs = torch.linspace(-1.0, 1.0, w, device=feat.device)
            ys = torch.linspace(-1.0, 1.0, h, device=feat.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            prob = torch.softmax(feat.reshape(b, c, h * w), dim=2)
            ex = (prob * gx.reshape(1, 1, -1)).sum(2)        # (b,c) expected x per channel
            ey = (prob * gy.reshape(1, 1, -1)).sum(2)        # (b,c) expected y per channel
            return torch.cat([ex, ey], dim=1)                # (b, 2c)

    class DeckCNN(nn.Module):
        """Conv stack -> **spatial-softmax** (expected coordinates) -> small head ->
        (px_norm, py_norm, visibility_logit). The spatial-softmax keeps explicit position information,
        which a flatten/global-pool head fails to regress for this blob-localization task."""

        def __init__(self, ch: int = 32):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(IN_CH, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 48
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),     # 24
                nn.Conv2d(32, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),                      # 24x24xch
            )
            self.softargmax = SpatialSoftmax()
            self.head = nn.Sequential(nn.Linear(2 * ch, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            out = self.head(self.softargmax(self.features(x)))
            # px, py in [-1,1] via tanh; the 3rd channel is the raw visibility logit
            return torch.cat([torch.tanh(out[:, :2]), out[:, 2:3]], dim=1)

    return DeckCNN


@dataclass
class CNNDetection:
    found: bool
    rel_xy: np.ndarray       # deck-centre minus drone, world frame (nadir cam) [m]
    confidence: float        # sigmoid(visibility logit)


class CNNDeckDetector:
    """Runs a trained :func:`DeckCNN` on the downward image and back-projects the predicted deck-centre
    pixel to a relative-XY fix (drop-in with the markerless fallback)."""

    def __init__(self, camera: CameraModel, weights_path: str | None = None, device: str = "cpu",
                 conf_thresh: float = 0.5):
        import torch

        self.cam = camera
        self.device = device
        self.conf_thresh = conf_thresh
        self.net = build_net()().to(device).eval()
        if weights_path is not None:
            self.net.load_state_dict(torch.load(weights_path, map_location=device))

    def detect(self, image, range_m: float) -> CNNDetection:
        import cv2
        import torch

        rgb = np.asarray(image)
        h0, w0 = rgb.shape[:2]
        x = cv2.resize(rgb, (INPUT_HW, INPUT_HW)).astype(np.float32) / 255.0   # (HW,HW,3)
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)         # (1,3,HW,HW)
        with torch.no_grad():
            px_n, py_n, vis = self.net(t)[0].cpu().numpy()
        conf = float(1.0 / (1.0 + np.exp(-vis)))
        if conf < self.conf_thresh:
            return CNNDetection(False, np.zeros(2), conf)
        # map normalised [-1,1] prediction -> full-image pixel, then back-project (same geometry as markerless)
        px = (px_n + 1.0) * 0.5 * w0
        py = (py_n + 1.0) * 0.5 * h0
        rel_xy = np.array([range_m * (px - self.cam.cx) / self.cam.fx,
                           -range_m * (py - self.cam.cy) / self.cam.fy])
        return CNNDetection(True, rel_xy, conf)


# --------------------------------------------------------------------- dataset + training (GPU, deferred)

def generate_dataset(n_frames: int = 4000, out_path: str = "runs/cnn/deck_dataset.npz",
                     seed: int = 0) -> str:
    """Render labelled downward frames with **randomized viewpoints** so the deck projects across the whole
    image (a proper detection dataset, not a near-constant one). Each frame: advance the ship deck, sample
    a target deck-centre pixel (uniform, incl. slightly off-frame for negatives) + an altitude, teleport the
    drone to the offset that places the deck there, render, and label with that pixel + visibility.

    Labels are free from the sim (the geometry is known). Uses the MuJoCo renderer (GPU/EGL), so run it
    *after* any GPU training, not concurrently."""
    import cv2
    import mujoco

    from drone_landing.cli import CAM_FOVY, CAM_H, CAM_W
    from drone_landing.perception import CameraModel
    from drone_landing.sim.platforms import sea_state
    from drone_landing.sim.world import LandingWorld, LandingWorldConfig

    rng = np.random.default_rng(seed)
    world = LandingWorld(LandingWorldConfig(world="x2_landing_ship", ship=sea_state("moderate")))
    cam = CameraModel(CAM_W, CAM_H, CAM_FOVY)
    renderer = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
    margin = 6

    def pad_label(rgb):
        """Self-consistent label = the actual rendered pad centroid (full-res, saturated-white, border-gated)
        -> normalised pixel + visibility. Labelling from the render (not a forward projection) guarantees the
        label matches the image regardless of camera-mount / deck-origin offsets."""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, m = cv2.threshold(g, 235, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return [0.0, 0.0, 0.0]
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c) / (CAM_W * CAM_H)
        bx, by, bw, bh = cv2.boundingRect(c)
        clipped = bx <= margin or by <= margin or bx + bw >= CAM_W - margin or by + bh >= CAM_H - margin
        M = cv2.moments(c)
        if not (0.002 <= area <= 0.6) or clipped or M["m00"] <= 0:
            return [0.0, 0.0, 0.0]                                  # no clean pad -> negative
        cpx, cpy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        return [2 * cpx / CAM_W - 1, 2 * cpy / CAM_H - 1, 1.0]

    imgs, labels = [], []
    try:
        world.reset(seed)
        qa = world.qadr
        for _ in range(n_frames):
            ps = world.platform.step(world.control_dt)          # advance the deck
            world._drive_platform(ps)
            deck = np.asarray(ps.pos, dtype=float)
            # teleport the drone to a randomized viewpoint so the pad lands at varied pixels. Mostly keep
            # it in-frame (positives) with ~25% wide shots for off-frame/edge negatives; the LABEL is read
            # back from the render, so the placement geometry need not be exact.
            spread = 0.55 if rng.random() < 0.75 else 1.15
            px_n, py_n = rng.uniform(-spread, spread, 2)
            h = float(rng.uniform(1.4, 3.0))
            px, py = (px_n + 1) * 0.5 * CAM_W, (py_n + 1) * 0.5 * CAM_H
            rel_x = (px - cam.cx) * h / cam.fx
            rel_y = -(py - cam.cy) * h / cam.fy
            world.data.qpos[qa:qa + 3] = [deck[0] - rel_x, deck[1] - rel_y, deck[2] + h]
            world.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
            world.data.qvel[:] = 0.0
            world._drive_gimbal()
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, camera="down")
            rgb = renderer.render()                                     # (CAM_H,CAM_W,3) RGB
            imgs.append(cv2.resize(rgb, (INPUT_HW, INPUT_HW)))
            labels.append(pad_label(rgb))
    finally:
        renderer.close()
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=np.array(imgs, dtype=np.uint8),
                        labels=np.array(labels, dtype=np.float32))
    return out_path


def train(dataset_path: str = "runs/cnn/deck_dataset.npz", epochs: int = 30, device: str = "cuda",
          out_path: str = "runs/cnn/deck_cnn.pt", seed: int = 0) -> str:
    """Train the DeckCNN (pixel regression + visibility) on the rendered dataset. GPU; run after the
    swarm GNN training (one heavy job at a time)."""
    import torch
    import torch.nn.functional as F

    data = np.load(dataset_path)
    imgs = data["images"].astype(np.float32) / 255.0          # (N, HW, HW, 3)
    X = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()   # (N, 3, HW, HW)
    Y = torch.from_numpy(data["labels"])
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    X, Y = X[perm], Y[perm]
    ntr = int(0.9 * n)
    dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    net = build_net()().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr, Ytr, Xva, Yva = X[:ntr].to(dev), Y[:ntr].to(dev), X[ntr:].to(dev), Y[ntr:].to(dev)
    bs = 128
    for ep in range(epochs):
        net.train()
        for i in range(0, ntr, bs):
            xb, yb = Xtr[i:i + bs], Ytr[i:i + bs]
            pred = net(xb)
            vis = yb[:, 2:3]
            loss_xy = (vis * (pred[:, :2] - yb[:, :2]) ** 2).sum() / (vis.sum() + 1e-6)   # only when visible
            loss_vis = F.binary_cross_entropy_with_logits(pred[:, 2:3], vis)
            loss = loss_xy + loss_vis
            opt.zero_grad(); loss.backward(); opt.step()
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(net.state_dict(), out_path)
    # validation pixel error (on visible frames), in normalised units
    net.eval()
    with torch.no_grad():
        pv = net(Xva)
        m = Yva[:, 2] > 0.5
        px_err = float(((pv[m, :2] - Yva[m, :2]) ** 2).sum(1).sqrt().mean()) if m.any() else float("nan")
    print(f"[cnn] trained {epochs} ep on {n} frames; val normalised pixel error = {px_err:.4f}")
    return out_path
