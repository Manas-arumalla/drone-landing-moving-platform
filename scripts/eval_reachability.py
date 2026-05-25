"""B2: visualize the HJ safe-landing set and show the runtime shield prevents hard landings.

  python scripts/eval_reachability.py            # ASCII safe-set map + shielded-vs-unshielded A/B
"""

from __future__ import annotations

import numpy as np

from drone_landing.control.reachability import LandingReachability, ReachabilityConfig


def ascii_safe_set(R: LandingReachability, cols: int = 56, rows: int = 22) -> None:
    """Print the safe set over (vertical-speed, altitude); '#'=safe, '.'=unsafe, '|'=analytic boundary."""
    print("Safe-landing set (vertical channel): '#' safe, '.' unsafe to commit, 'o' analytic braking curve")
    print(f"  altitude 0..{R.cfg.h_max:.0f} m (top=high)   vertical speed {-R.cfg.w_max:.0f}..{R.cfg.w_max:.0f} m/s (left=descending)\n")
    ws = np.linspace(-R.cfg.w_max, R.cfg.w_max, cols)
    hs = np.linspace(R.cfg.h_max, 0.0, rows)
    brake = R.braking_boundary(ws)
    for h in hs:
        line = []
        for j, w in enumerate(ws):
            if w < 0 and abs(h - brake[j]) < (R.cfg.h_max / rows) / 2:
                line.append("o")
            else:
                line.append("#" if R.is_safe(float(h), float(w)) else ".")
        print("  " + "".join(line))
    print("  " + "".join("^" if abs(w) < R.cfg.w_max / cols else " " for w in ws) + "   (^ = w=0)")


def shield_ab(R: LandingReachability, trials: int = 200, seed: int = 0) -> None:
    """Reckless nominal controller (dive hard) + random wind; compare touchdown speed with/without shield."""
    c = R.cfg
    rng = np.random.default_rng(seed)
    for shielded in (False, True):
        hard = 0
        tds = []
        for _ in range(trials):
            h, w = 2.5 + 0.5 * rng.random(), 0.0
            for _ in range(4000):
                a_nom = c.a_min + 1.0                      # reckless: near-max descent command
                a = R.safe_action(h, w, a_nom)[0] if shielded else float(np.clip(a_nom, c.a_min, c.a_max))
                d = float(rng.uniform(-c.d_max, c.d_max))   # random wind/air-wake
                w += (a + d) * c.dt
                h += w * c.dt
                if h <= 0:
                    tds.append(abs(w))
                    hard += abs(w) > c.w_land
                    break
        tag = "SHIELDED  " if shielded else "unshielded"
        print(f"  {tag}: hard landings {hard:3d}/{trials} = {100 * hard / trials:3.0f}%   "
              f"mean |touchdown v| = {np.mean(tds):.2f} m/s")


def main() -> None:
    R = LandingReachability(ReachabilityConfig())
    print(f"Computed safe set: grid {R.cfg.nh}x{R.cfg.nw}, {100 * R.safe.mean():.0f}% of states safe, "
          f"w_land={R.cfg.w_land} m/s, disturbance bound {R.cfg.d_max} m/s^2\n")
    ascii_safe_set(R)
    print("\nRuntime-assurance shield vs a reckless 'dive' controller under random wind:")
    shield_ab(R)


if __name__ == "__main__":
    main()
