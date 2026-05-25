"""Phase-1 physics validation: can a drone actually land on the moving rover, with strict
contact (no weld lock)?

This uses a geometric SE(3) cascade controller fed with GROUND TRUTH. That is allowed *only* here
as a physics-validation tool (clearly not a deployable controller) to confirm the world is sane and
landable before perception/estimation exist. The deployable controllers (Phases 3-4) consume only
sensor-derived state. See docs/REALISM_CHARTER.md.

Usage:
    $env:PYTHONPATH="src"; python scripts/validate_landing.py --episodes 5 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from drone_landing.sim.world import LandingWorld, LandingWorldConfig


def quat2R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class GeometricCascade:
    """Position (outer) + geometric SE(3) attitude (inner) + control allocation."""

    def __init__(self, world: LandingWorld):
        self.m = float(world.model.body_mass[world.drone_bid])
        self.J = world.model.body_inertia[world.drone_bid].copy()
        self.g = 9.81
        self.deck_top = world.deck_top_z
        # control allocation: [T, tau_x, tau_y, tau_z] -> 4 motor thrusts (corrected quad-X)
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-.18, .18, .18, -.18],
            [.14, .14, -.14, -.14],
            [.0201, -.0201, .0201, -.0201],
        ])
        self.A_inv = np.linalg.inv(A)
        self.tmax = world.thrust_max
        # gains
        self.kp_xy, self.kd_xy = 1.5, 2.0
        self.a_xy_max = 3.5
        self.kp_vz = 4.5
        self.kpR, self.kdR = 130.0, 22.0
        # touchdown commit state
        self.dt = world.control_dt
        self.committed = False
        self.commit_t = 0.0
        self.commit_clearance = 0.20
        self.commit_ramp = 0.5  # seconds to ramp collective to zero

    def reset(self) -> None:
        self.committed = False
        self.commit_t = 0.0

    def _attitude_mix(self, f_des: np.ndarray, R: np.ndarray, omega: np.ndarray, level: bool) -> np.ndarray:
        zb = R[:, 2]
        T = max(0.0, float(f_des @ zb))
        if level:
            R_des = np.eye(3)
        else:
            zb_des = f_des / np.linalg.norm(f_des)
            yb = np.cross(zb_des, np.array([1.0, 0.0, 0.0]))
            yb /= np.linalg.norm(yb)
            xb = np.cross(yb, zb_des)
            R_des = np.column_stack([xb, yb, zb_des])
        eR = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        M = self.J * (-self.kpR * eR - self.kdR * omega) + np.cross(omega, self.J * omega)
        f = self.A_inv @ np.array([T, M[0], M[1], M[2]])
        return np.clip(f, 0.0, self.tmax)

    def act(self, truth: dict) -> np.ndarray:
        """Convenience wrapper for the truth-based validation path."""
        rel_pos = truth["platform_pos"] - truth["drone_pos"]
        rel_vel = truth["platform_vel"] - truth["drone_vel"]
        return self.act_core(rel_pos, rel_vel, quat2R(truth["drone_quat"]),
                             truth["drone_angvel"], truth["support_feet"])

    def act_core(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R: np.ndarray,
                 omega: np.ndarray, support_feet: int, allow_descent: bool = True) -> np.ndarray:
        """Geometric cascade from the platform-relative state (works on truth OR estimate).

        ``rel_pos`` / ``rel_vel`` are platform-minus-drone in the world frame; ``R`` is the drone
        attitude (body->world); ``omega`` is the body angular rate; ``support_feet`` is the gear
        contact count. When ``allow_descent`` is False (e.g. the marker is not currently tracked),
        the drone holds altitude and re-centres instead of descending blind.
        """
        e_xy = -rel_pos[:2]            # drone - platform
        ev_xy = -rel_vel[:2]
        clearance = -rel_pos[2]        # drone height above deck
        horiz_err = float(np.linalg.norm(rel_pos[:2]))

        a_xy = -self.kp_xy * e_xy - self.kd_xy * ev_xy
        n = np.linalg.norm(a_xy)
        if n > self.a_xy_max:
            a_xy *= self.a_xy_max / n

        if support_feet >= 3:
            return np.zeros(4)  # planted: real deck friction now carries the drone

        # vertical descent profile (gentle near the deck); descend only when aligned AND tracked
        if not allow_descent or horiz_err > 0.4:
            vz_des = 0.0
        elif clearance > 0.6:
            vz_des = -0.5
        elif clearance > 0.30:
            vz_des = -0.3
        else:
            vz_des = -0.15
        a_z = self.kp_vz * (vz_des - (-rel_vel[2]))  # drone vz = -rel_vel_z (deck vz ~ 0)
        f_des = self.m * np.array([a_xy[0], a_xy[1], a_z + self.g])

        if support_feet >= 1 and allow_descent:
            f_des = self.m * np.array([a_xy[0], a_xy[1], 0.5 * self.g])
        return self._attitude_mix(f_des, R, omega, level=False)


def run_episode(world: LandingWorld, controller: GeometricCascade, seed: int) -> dict:
    world.reset(seed)
    touchdown_vspeed = None
    min_horiz_at_contact = None
    while True:
        truth = world.observe_truth()
        ctrl = controller.act(truth)
        step = world.step(ctrl)
        if truth["support_feet"] >= 1 and touchdown_vspeed is None:
            touchdown_vspeed = truth["vertical_speed"]
            min_horiz_at_contact = truth["horizontal_error"]
        if step.terminated or step.truncated:
            t = step.truth
            return {
                "outcome": step.info["termination"],
                "time": round(t["time"], 2),
                "horiz_err": round(t["horizontal_error"], 3),
                "touchdown_vspeed": None if touchdown_vspeed is None else round(touchdown_vspeed, 3),
                "horiz_at_contact": None if min_horiz_at_contact is None else round(min_horiz_at_contact, 3),
                "tilt_deg": round(t["tilt_deg"], 2),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    world = LandingWorld(LandingWorldConfig())
    controller = GeometricCascade(world)

    results = []
    for i in range(args.episodes):
        r = run_episode(world, controller, seed=args.seed + i)
        results.append(r)
        print(f"ep {i:02d} seed={args.seed + i}: {r}")

    succ = sum(r["outcome"] == "success" for r in results)
    print(f"\nSUCCESS {succ}/{args.episodes} = {100 * succ / args.episodes:.0f}%")
    landed = [r for r in results if r["outcome"] == "success"]
    if landed:
        print(f"  mean touchdown horiz err = {np.mean([r['horiz_err'] for r in landed]):.3f} m")
        tv = [r['touchdown_vspeed'] for r in landed if r['touchdown_vspeed'] is not None]
        if tv:
            print(f"  mean contact vspeed      = {np.mean(tv):.3f} m/s")


if __name__ == "__main__":
    main()
