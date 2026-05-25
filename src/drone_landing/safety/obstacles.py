"""Static-obstacle field + a no-cheats onboard range sensor (P3.1 / P3.2).

A real deck approach is cluttered: the offshore-support-vessel carries a **wheelhouse, mast and raised
bow** fore of the helideck (see ``assets/mujoco/worlds/x2_landing_offshore.xml``). A drone recovering onto
the pad must *sense and avoid* that superstructure, not be handed its coordinates. This module supplies:

* :class:`Obstacle` / :class:`ObstacleField` — the world's static superstructure as simple vertical
  primitives (a circular footprint = mast/post, an axis-aligned box footprint = wheelhouse/hull), each
  spanning ``[z_min, z_max]``. Cheap to query and exact for ray-casting.
* :class:`RangeSensor` — an **onboard 2-D scanning rangefinder** (a LiDAR slice at the drone's altitude).
  It casts ``n_beams`` rays around the drone, returns the range to the nearest obstacle surface within
  ``max_range`` (with bearing-quantized beams, Gaussian range noise and random dropout), and gives back
  **world-frame surface points** — *not* obstacle identities or centres. That is the no-cheats contract
  (:doc:`no-cheats-realism`): avoidance must run on what a real sensor returns (surface hits), exactly as
  the controller runs on the EKF estimate rather than ground-truth pose.

The sensor models a **horizontal scan plane** (2.5-D): a beam at altitude ``z`` can only strike an obstacle
whose vertical span contains ``z``. That matches a fixed 2-D LiDAR; a tilting/3-D scan is a future
refinement. Pure geometry — no MuJoCo dependency — so it is fast and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """A static vertical obstacle. ``shape='circle'`` uses ``radius``; ``shape='box'`` uses ``half`` (ex,ey).

    The footprint is in the world XY plane at ``center``; the obstacle exists for altitudes in
    ``[z_min, z_max]``. Boxes are axis-aligned (the OSV superstructure is)."""

    center: tuple[float, float]
    z_min: float
    z_max: float
    shape: str = "circle"
    radius: float = 0.3
    half: tuple[float, float] = (0.3, 0.3)
    name: str = ""

    def spans(self, z: float) -> bool:
        """True if a horizontal beam at altitude ``z`` would intersect this obstacle's vertical extent."""
        return self.z_min <= z <= self.z_max

    def ray_distance(self, o: np.ndarray, d: np.ndarray, max_range: float) -> float | None:
        """Distance along unit ray (origin ``o``, direction ``d``) to the surface, or None if it misses."""
        c = np.asarray(self.center, dtype=float)
        if self.shape == "circle":
            return _ray_circle(o, d, c, self.radius, max_range)
        return _ray_box(o, d, c, np.asarray(self.half, dtype=float), max_range)

    def signed_distance(self, p: np.ndarray) -> float:
        """Signed distance from XY point ``p`` to the footprint surface (negative inside)."""
        c = np.asarray(self.center, dtype=float)
        if self.shape == "circle":
            return float(np.linalg.norm(p - c)) - self.radius
        q = np.abs(p - c) - np.asarray(self.half, dtype=float)         # box SDF (Inigo Quilez form)
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(q[0], q[1]), 0.0))
        return outside + inside


def _ray_circle(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float, max_range: float) -> float | None:
    f = o - c
    b = float(f @ d)
    cc = float(f @ f) - r * r
    disc = b * b - cc
    if disc < 0.0:
        return None
    s = np.sqrt(disc)
    t = -b - s
    if t < 0.0:
        t = -b + s                                                     # origin inside the circle
    if 0.0 <= t <= max_range:
        return float(t)
    return None


def _ray_box(o: np.ndarray, d: np.ndarray, c: np.ndarray, half: np.ndarray, max_range: float):
    """Slab method for an axis-aligned box footprint (2-D)."""
    lo, hi = c - half, c + half
    tmin, tmax = 0.0, max_range
    for k in range(2):
        if abs(d[k]) < 1e-12:
            if o[k] < lo[k] or o[k] > hi[k]:
                return None                                            # parallel and outside the slab
            continue
        inv = 1.0 / d[k]
        t1, t2 = (lo[k] - o[k]) * inv, (hi[k] - o[k]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    return float(tmin)


@dataclass
class ObstacleField:
    """A collection of static obstacles with ray-cast and nearest-surface queries (world XY + altitude)."""

    obstacles: list[Obstacle] = field(default_factory=list)

    def add(self, obs: Obstacle) -> "ObstacleField":
        self.obstacles.append(obs)
        return self

    def nearest_surface_distance(self, p: np.ndarray, z: float) -> float:
        """Smallest footprint signed distance over obstacles spanning altitude ``z`` (inf if none)."""
        ds = [o.signed_distance(np.asarray(p, float)) for o in self.obstacles if o.spans(z)]
        return min(ds) if ds else float("inf")

    @staticmethod
    def offshore_osv(deck_xy: tuple[float, float] = (0.0, 0.0), deck_z: float = 0.0) -> "ObstacleField":
        """The OSV superstructure as obstacles, matching ``x2_landing_offshore.xml`` (deck-relative).

        Wheelhouse (box) + mast (thin circle) sit ~1.5 m fore of the helideck centre; the raised bow is a
        box further fore. Heights are deck-relative (the pad surface is the altitude datum)."""
        dx, dy = deck_xy
        f = ObstacleField()
        f.add(Obstacle((dx + 1.5, dy), z_min=deck_z + 0.0, z_max=deck_z + 0.95, shape="box",
                       half=(0.5, 0.62), name="wheelhouse"))
        f.add(Obstacle((dx + 1.5, dy), z_min=deck_z + 0.95, z_max=deck_z + 1.4, shape="circle",
                       radius=0.10, name="mast"))
        f.add(Obstacle((dx + 2.1, dy), z_min=deck_z + 0.0, z_max=deck_z + 0.36, shape="box",
                       half=(0.32, 0.6), name="bow"))
        return f


@dataclass(frozen=True)
class RangeSensorConfig:
    n_beams: int = 36              # beams per scan (10 deg spacing at 360 deg)
    max_range: float = 6.0         # m  detection range
    min_range: float = 0.15        # m  blind zone (returns inside this are discarded)
    fov: float = 2 * np.pi         # rad full azimuth coverage (default 360 deg)
    bearing0: float = 0.0          # rad scan start bearing (world frame)
    sigma: float = 0.02            # m  range noise (1-sigma)
    dropout: float = 0.05          # probability a beam returns nothing despite a hit


class RangeSensor:
    """Onboard 2-D scanning rangefinder: returns world-frame surface points, never obstacle identities."""

    def __init__(self, field: ObstacleField, config: RangeSensorConfig | None = None):
        self.field = field
        self.cfg = config or RangeSensorConfig()

    def scan(self, p_xy: np.ndarray, z: float, rng: np.random.Generator | None = None) -> list[np.ndarray]:
        """Return the list of detected surface points (world XY) for a scan from ``p_xy`` at altitude ``z``."""
        c = self.cfg
        rng = rng or np.random.default_rng()
        o = np.asarray(p_xy, dtype=float)
        live = [ob for ob in self.field.obstacles if ob.spans(z)]
        hits: list[np.ndarray] = []
        for k in range(c.n_beams):
            ang = c.bearing0 + (c.fov * k / c.n_beams if c.fov < 2 * np.pi else 2 * np.pi * k / c.n_beams)
            d = np.array([np.cos(ang), np.sin(ang)])
            best = None
            for ob in live:
                t = ob.ray_distance(o, d, c.max_range)
                if t is not None and (best is None or t < best):
                    best = t
            if best is None or best < c.min_range:
                continue
            if rng.random() < c.dropout:
                continue
            rmeas = best + rng.normal(0.0, c.sigma)                    # noisy range return
            hits.append(o + max(rmeas, 0.0) * d)
        return hits
