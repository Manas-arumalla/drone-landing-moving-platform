"""Per-drone onboard vision for cooperative perception (CP).

This is the realism upgrade over the modeled ``SwarmSensing`` deck estimate: each drone now runs **real
vision on its own rendered downward camera**. It detects the bright landing pad (saturated-white blob,
border-gated — the validated markerless method) and back-projects the centroid through the pinhole +
rangefinder altitude to a **world-frame deck-position fix**.

A drone whose camera *does not* see the pad (it has drifted away, the deck left the field of view, or it
is too tilted) returns ``None`` — it is *blind* and must rely on the cooperative consensus (A2) fusing
the fixes its neighbours *do* have. That is the cooperative-perception loop: drones that see the deck
share their visual fix; blind drones land on the shared estimate.

Rendering N cameras per step is expensive, so vision runs at a reduced rate (``period`` control steps)
and only for active drones. Nadir-approximation back-projection (the drones hold/approach near-level);
a stabilized gimbal / full ray-cast is a future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisionConfig:
    width: int = 200
    height: int = 200
    fovy: float = 120.0           # wide downward FOV so holding drones (ring radius ~ altitude) see the pad
    bright_thresh: int = 235      # saturated-white pad
    min_area_frac: float = 0.002
    max_area_frac: float = 0.7
    border_margin: int = 4        # reject a clipped pad (biased centroid)
    period: int = 5               # render every `period` control steps (vision rate << control rate)


class SwarmVision:
    """Renders each drone's downward camera and returns its world-frame deck fix (or None if blind)."""

    def __init__(self, world, config: VisionConfig | None = None, camera_ids=None):
        import mujoco

        from drone_landing.perception.camera import CameraModel
        self.world = world
        self.cfg = config or VisionConfig()
        # heterogeneous fleet (P2.4): only these drones carry a camera; the rest are camera-less and must
        # rely on neighbours' shared fixes (cooperative perception). None => every drone has a camera.
        self.camera_ids = None if camera_ids is None else set(camera_ids)
        self.cam = CameraModel(self.cfg.width, self.cfg.height, self.cfg.fovy)
        self._renderer = mujoco.Renderer(world.model, height=self.cfg.height, width=self.cfg.width)
        self._k = 0
        self._last: dict[int, np.ndarray | None] = {}

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass

    def _detect_one(self, i: int, own_pos_est: np.ndarray, deck_z: float):
        """Render drone i's camera, find the pad, back-project to a world deck fix (or None)."""
        import cv2

        c = self.cfg
        self._renderer.update_scene(self.world.data, camera=f"cam_{i}")
        img = self._renderer.render()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, c.bright_thresh, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        blob = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(blob) / (c.width * c.height)
        bx, by, bw, bh = cv2.boundingRect(blob)
        clipped = (bx <= c.border_margin or by <= c.border_margin
                   or bx + bw >= c.width - c.border_margin or by + bh >= c.height - c.border_margin)
        m = cv2.moments(blob)
        if not (c.min_area_frac <= area <= c.max_area_frac) or clipped or m["m00"] <= 0:
            return None
        px, py = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # rangefinder: vertical distance from the drone to the pad (a real downward sensor)
        rng_m = max(self.world.drone_pos(i)[2] - deck_z, 0.1)
        # nadir back-projection -> pad position relative to the drone (camera looks down -Z)
        rel_x = rng_m * (px - self.cam.cx) / self.cam.fx
        rel_y = -rng_m * (py - self.cam.cy) / self.cam.fy
        # world fix = own-pose ESTIMATE + relative offset (own-localization error enters here, realistically)
        return np.array([own_pos_est[0] + rel_x, own_pos_est[1] + rel_y, deck_z])

    def sense(self, own_pos_est: dict[int, np.ndarray], deck_z: float, landed: set) -> dict:
        """Return ``{i: deck_fix_world or None}`` for active drones (cached between vision frames)."""
        import mujoco

        active = [i for i in range(self.world.n) if i not in landed]
        has_cam = (lambda i: self.camera_ids is None or i in self.camera_ids)
        if self._k % max(self.cfg.period, 1) == 0:
            self.world.drive_gimbals()                    # hold cameras nadir at current drone poses
            mujoco.mj_forward(self.world.model, self.world.data)   # propagate mocap pose for rendering
            self._last = {i: (self._detect_one(i, own_pos_est[i], deck_z) if has_cam(i) else None)
                          for i in active}
        self._k += 1
        return {i: self._last.get(i) for i in active}
