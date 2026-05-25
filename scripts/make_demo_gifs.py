"""P6 dissemination — render demo GIFs of the closed-loop autopilot + swarm into ``docs/media/``.

Renders the chase ("track") camera for single-drone scenarios and a framed free camera for the swarm, at a
modest size, into looping GIFs for the README/portfolio. Vision autopilot, no truth in the loop.

  python scripts/make_demo_gifs.py                 # all
  python scripts/make_demo_gifs.py ship offshore   # a subset
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from drone_landing.cli import CAM_H, CAM_W   # perception render size MUST match the autopilot's CameraModel

MEDIA = Path(__file__).resolve().parent.parent / "docs" / "media"
W, H, EVERY, FPS = 360, 270, 6, 12            # render size, frame subsample, GIF fps (keep files small)


def _episode(scenario: str, seed: int, max_t: float, spec_kw):
    """Run one episode, return (track-cam frames, outcome)."""
    import mujoco

    from drone_landing.cli import SimSpec, build
    world, ap = build(SimSpec(scenario, "geometric", **spec_kw))
    sensors = world.reset(seed); ap.reset()
    r = mujoco.Renderer(world.model, height=H, width=W)
    cam_r = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)   # perception cam at the calibrated size
    frames, k, t = [], 0, 0.0
    while t < max_t:
        img = None
        if ap.wants_frame():
            cam_r.update_scene(world.data, camera="down"); img = cam_r.render()
        ctrl = ap.step(img, sensors, world.observe_truth()["support_feet"])
        step = world.step(ctrl); sensors = step.sensors
        if k % EVERY == 0:
            r.update_scene(world.data, camera="track"); frames.append(r.render())
        k += 1; t += world.control_dt
        if step.terminated or step.truncated:
            for _ in range(FPS):
                frames.append(frames[-1])
            break
    r.close(); cam_r.close()
    return frames, step.info["termination"]


def _single(scenario: str, seed: int, max_t: float = 14.0, **spec_kw):
    """Render a *successful* landing: retry seeds until one succeeds (so the demo shows a clean landing)."""
    last = None
    for s in range(seed, seed + 6):
        frames, outcome = _episode(scenario, s, max_t, spec_kw)
        last = (frames, outcome)
        if outcome == "success":
            return frames, f"success (seed {s})"
    return last[0], last[1]


def _swarm(n: int, seed: int, max_t: float = 18.0):
    import mujoco

    from drone_landing_swarm.coordinator import SwarmConfig
    from drone_landing_swarm.mujoco_runner import MujocoSwarmCoordinator
    coord = MujocoSwarmCoordinator(SwarmConfig(n_drones=n, scenario="ship", sea="moderate", offshore=True))
    coord.reset(seed)
    r = mujoco.Renderer(coord.world.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 9.0, 130.0, -25.0
    frames, k = [], 0
    while coord.t < max_t and len(coord.landed) < n:
        coord.step()
        if k % EVERY == 0:
            cam.lookat[:] = coord.deck_pos
            r.update_scene(coord.world.data, camera=cam); frames.append(r.render())
        k += 1
    for _ in range(FPS):
        frames.append(frames[-1])
    r.close()
    return frames, f"{len(coord.landed)}/{n} landed"


DEMOS = {
    "ground": lambda: _single("ground", 1),
    "ship": lambda: _single("ship", 0, sea="rough"),
    "offshore": lambda: _single("offshore", 0, sea="moderate"),
    "swarm": lambda: _swarm(4, 0),
}


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or list(DEMOS)
    for name in which:
        print(f"rendering {name} ...", flush=True)
        frames, outcome = DEMOS[name]()
        path = MEDIA / f"{name}.gif"
        imageio.mimsave(path, [np.asarray(f) for f in frames], fps=FPS, loop=0)
        kb = path.stat().st_size / 1024
        print(f"  -> {path}  ({len(frames)} frames, {kb:.0f} KB, {outcome})", flush=True)


if __name__ == "__main__":
    main()
