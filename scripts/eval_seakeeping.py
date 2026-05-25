"""B1 maritime fidelity: show the spectral deck model matches the validated motion, and the air-wake
measurably stresses the approach (with the disturbance observer recovering it).

  python scripts/eval_seakeeping.py              # spectrum/motion stats (fast)
  python scripts/eval_seakeeping.py --episodes 14  # + full-pipeline air-wake A/B (slow, renders)
"""

from __future__ import annotations

import argparse

import numpy as np

from drone_landing.sim.platforms import sea_state
from drone_landing.sim.platforms.ship import ShipDeckMotion
from drone_landing.sim.platforms.wave_spectrum import SPECTRAL_SEA_STATES, jonswap, significant_height


def motion_stats(cfg, T=180.0, dt=0.05, seed=0) -> dict:
    m = ShipDeckMotion(cfg)
    m.reset(np.random.default_rng(seed))
    zs, rolls, pitches = [], [], []
    for _ in range(int(T / dt)):
        s = m.step(dt)
        zs.append(s.pos[2])
        w, x, y, z = s.quat
        rolls.append(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitches.append(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return dict(z_rms=float(np.std(zs)), roll=float(np.degrees(rolls).std()),
                pitch=float(np.degrees(pitches).std()))


def spectrum_table() -> None:
    print("Wave spectrum (JONSWAP) — significant-height recovery from the spectral model:")
    w = np.linspace(0.05, 4.0, 3000)
    for nm, sp in SPECTRAL_SEA_STATES.items():
        s = jonswap(w, sp.hs, sp.tp, sp.gamma)
        print(f"  {nm:<9s} Hs={sp.hs:.2f} m  Tp={sp.tp:.1f} s  gamma={sp.gamma:.1f}  "
              f"-> recovered Hs={significant_height(w, s):.2f} m")
    print("\nDeck motion RMS — validated sum-of-sinusoids vs spectral (calibrated to match):")
    for nm in ["calm", "moderate", "rough"]:
        a = motion_stats(sea_state(nm))
        b = motion_stats(sea_state(nm, spectral=True))
        print(f"  {nm:<9s} sinusoid: z={a['z_rms']:.3f}m roll={a['roll']:.2f} pitch={a['pitch']:.2f}"
              f"  |  spectral: z={b['z_rms']:.3f}m roll={b['roll']:.2f} pitch={b['pitch']:.2f}")


def airwake_ab(episodes: int) -> None:
    import mujoco

    from drone_landing.cli import CAM_H, CAM_W, SimSpec, build, run_episode

    def run(label, spec):
        world, ap = build(spec)
        r = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
        succ = 0
        try:
            for i in range(episodes):
                succ += run_episode(world, ap, r, i)["outcome"] == "success"
        finally:
            r.close()
        print(f"  {label:<36s} {succ:2d}/{episodes} = {100 * succ / episodes:3.0f}%")

    print(f"\nAir-wake stressor + DOB mitigation (full vision pipeline, {episodes} eps):")
    run("ship moderate (no airwake)", SimSpec("ship", "geometric", sea="moderate"))
    run("ship moderate + AIR-WAKE", SimSpec("ship", "geometric", sea="moderate", airwake=True))
    run("ship moderate + AIR-WAKE + DOB", SimSpec("ship", "geometric", sea="moderate",
                                                  airwake=True, dob=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="B1 maritime fidelity evaluation")
    ap.add_argument("--episodes", type=int, default=0, help="run the full-pipeline air-wake A/B (slow)")
    args = ap.parse_args()
    spectrum_table()
    if args.episodes > 0:
        airwake_ab(args.episodes)


if __name__ == "__main__":
    main()
