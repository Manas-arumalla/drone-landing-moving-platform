"""Phase-0 smoke check: load the X2 landing world, print its structure, step it at hover,
and render the downward + tracking cameras. Verifies the model compiles and is physically sane.

Usage:
    $env:PYTHONPATH="src"; python scripts/check_world.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from drone_landing.sim import load_model, repo_root


def names(model, objtype, count):
    out = []
    for i in range(count):
        n = mujoco.mj_id2name(model, objtype, i)
        out.append(n if n is not None else f"<{i}>")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="x2_landing_ground")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    model = load_model(args.world)
    data = mujoco.MjData(model)

    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    drone_mass = float(model.body_mass[drone_id])
    ideal_hover = drone_mass * 9.81 / model.nu
    print(f"World: {args.world}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nsensordata={model.nsensordata}")
    print(f"  bodies:    {names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)}")
    print(f"  actuators: {names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)}")
    print(f"  sensors:   {names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)}")
    print(f"  cameras:   {names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)}")
    print(f"  drone mass = {drone_mass:.4f} kg  (weight = {drone_mass * 9.81:.3f} N)")
    print(f"  ideal hover thrust/motor = {ideal_hover:.4f} N")

    # Load the hover keyframe and run with constant hover thrust.
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    mujoco.mj_resetDataKeyframe(model, data, key)
    hover_ctrl = model.key_ctrl[key].copy()
    print(f"  hover ctrl = {hover_ctrl}  (per-motor; sum={hover_ctrl.sum():.3f} N)")

    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    z0 = float(data.qpos[qadr + 2])
    zmin, zmax = z0, z0
    for _ in range(args.steps):
        data.ctrl[:] = hover_ctrl
        mujoco.mj_step(model, data)
        z = float(data.qpos[qadr + 2])
        zmin, zmax = min(zmin, z), max(zmax, z)

    zf = float(data.qpos[qadr + 2])
    finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    print(f"  hover test: z0={z0:.3f} -> zf={zf:.3f} (range [{zmin:.3f},{zmax:.3f}]) finite={finite}")
    drift = abs(zf - z0)
    verdict = "STABLE" if (finite and drift < 0.5) else "CHECK"
    print(f"  altitude drift over {args.steps} steps = {drift:.3f} m -> {verdict}")

    if not args.no_render:
        out = repo_root() / "runs" / "phase0"
        out.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            for cam, (w, h) in {"down": (320, 320), "track": (960, 540)}.items():
                renderer = mujoco.Renderer(model, height=h, width=w)
                renderer.update_scene(data, camera=cam)
                Image.fromarray(renderer.render()).save(out / f"{cam}.png")
                renderer.close()
            print(f"  rendered cameras to {out}")
        except Exception as exc:  # rendering is a bonus; loading+stepping is the gate
            print(f"  [render skipped] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
