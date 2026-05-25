"""Phase 3 sense-and-avoid + contingency tests (pure geometry/logic, no MuJoCo)."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from drone_landing.safety import (
    AvoidConfig,
    ContingencyConfig,
    ContingencySupervisor,
    GeofenceSpec,
    HealthStatus,
    HOCBFAvoider,
    Obstacle,
    ObstacleField,
    RangeSensor,
    RangeSensorConfig,
    cluster_returns,
)


class ObstacleSensorTests(unittest.TestCase):
    def test_ray_hits_circle_and_box(self):
        circ = Obstacle((3.0, 0.0), z_min=0.0, z_max=2.0, shape="circle", radius=0.5)
        # ray from origin along +x hits the near face at x=2.5
        t = circ.ray_distance(np.zeros(2), np.array([1.0, 0.0]), max_range=10.0)
        self.assertAlmostEqual(t, 2.5, places=6)
        # a ray pointing away misses
        self.assertIsNone(circ.ray_distance(np.zeros(2), np.array([-1.0, 0.0]), 10.0))
        box = Obstacle((3.0, 0.0), z_min=0.0, z_max=2.0, shape="box", half=(0.5, 0.5))
        self.assertAlmostEqual(box.ray_distance(np.zeros(2), np.array([1.0, 0.0]), 10.0), 2.5, places=6)

    def test_signed_distance_sign(self):
        circ = Obstacle((0.0, 0.0), 0.0, 2.0, shape="circle", radius=1.0)
        self.assertLess(circ.signed_distance(np.array([0.2, 0.0])), 0.0)    # inside
        self.assertAlmostEqual(circ.signed_distance(np.array([2.0, 0.0])), 1.0, places=6)

    def test_altitude_gating(self):
        # an obstacle only 1 m tall is not seen by a beam at 3 m altitude
        f = ObstacleField([Obstacle((2.0, 0.0), z_min=0.0, z_max=1.0, shape="circle", radius=0.4)])
        s = RangeSensor(f, RangeSensorConfig(dropout=0.0, sigma=0.0))
        self.assertEqual(len(s.scan(np.zeros(2), z=3.0)), 0)               # above the obstacle top
        self.assertGreater(len(s.scan(np.zeros(2), z=0.5)), 0)            # within its span

    def test_returns_are_surface_points_near_obstacle(self):
        f = ObstacleField([Obstacle((2.0, 0.0), 0.0, 3.0, shape="circle", radius=0.4)])
        s = RangeSensor(f, RangeSensorConfig(dropout=0.0, sigma=0.0, n_beams=72))
        hits = s.scan(np.zeros(2), z=1.0)
        self.assertGreater(len(hits), 0)
        for h in hits:
            # every return lies ~on the circle surface (distance from centre ~ radius)
            self.assertAlmostEqual(float(np.linalg.norm(h - np.array([2.0, 0.0]))), 0.4, delta=0.05)

    def test_max_range_blind(self):
        f = ObstacleField([Obstacle((20.0, 0.0), 0.0, 3.0, shape="circle", radius=0.4)])
        s = RangeSensor(f, RangeSensorConfig(max_range=6.0, dropout=0.0, sigma=0.0))
        self.assertEqual(len(s.scan(np.zeros(2), z=1.0)), 0)               # beyond max_range

    def test_cluster_reduces_returns(self):
        f = ObstacleField([Obstacle((2.0, 0.0), 0.0, 3.0, shape="circle", radius=0.4)])
        s = RangeSensor(f, RangeSensorConfig(dropout=0.0, sigma=0.0, n_beams=72))
        hits = s.scan(np.zeros(2), z=1.0)
        reps = cluster_returns(hits, tol=0.5)
        self.assertLessEqual(len(reps), len(hits))
        self.assertGreaterEqual(len(reps), 1)


class HOCBFTests(unittest.TestCase):
    def test_passthrough_when_clear(self):
        av = HOCBFAvoider()
        a_des = np.array([2.0, 0.0])
        a = av.filter(np.zeros(2), np.zeros(2), a_des, obstacle_points=[])
        np.testing.assert_allclose(a, a_des)

    def test_closed_loop_never_penetrates(self):
        # Double-integrator drone flies from (-4,0) to goal (4,0) with an obstacle dead ahead at (0,0).
        # The HOCBF must keep it outside the keep-out for the whole rollout while it still reaches the goal.
        cfg = AvoidConfig(a_max=4.0, drone_radius=0.25, margin=0.15, latency=0.05)
        av = HOCBFAvoider(cfg)
        obs_c = np.array([0.0, 0.0])
        obs_r = 0.5
        R_keepout = obs_r + cfg.drone_radius + cfg.margin
        p = np.array([-4.0, 0.05]); v = np.zeros(2)
        goal = np.array([4.0, 0.0])
        dt = 0.02
        min_clear = float("inf")
        for _ in range(1500):
            a_des = 3.0 * (goal - p) - 2.5 * v                            # PD to goal
            n = float(np.linalg.norm(a_des))
            if n > cfg.a_max:
                a_des *= cfg.a_max / n
            a = av.filter(p, v, a_des, [obs_c], obstacle_radius=obs_r)
            v = v + a * dt
            p = p + v * dt
            min_clear = min(min_clear, float(np.linalg.norm(p - obs_c)) - R_keepout)
        self.assertGreater(min_clear, -0.05, f"penetrated keep-out (min clearance {min_clear:.3f} m)")
        self.assertLess(float(np.linalg.norm(p - goal)), 0.6)            # still reached the goal
        self.assertGreater(av.report.activations, 0)                     # the filter actually engaged

    def test_offshore_field_sensed_and_deflects(self):
        # The exact in-loop pipeline the autopilot runs with --avoid (deck-relative frame): the offshore
        # superstructure is sensed by the range sensor and the HOCBF deflects a command aimed at it.
        field = ObstacleField.offshore_osv(deck_xy=(0.0, 0.0), deck_z=0.0)
        sensor = RangeSensor(field, RangeSensorConfig(max_range=5.0, sigma=0.0, dropout=0.0))
        av = HOCBFAvoider(AvoidConfig())
        rng = np.random.default_rng(0)
        # drone deck-relative at (0.6, 0) closing on the wheelhouse (deck-relative (1.5,0)), low altitude
        p = np.array([0.6, 0.0]); v = np.array([0.6, 0.0]); alt = 0.5
        hits = cluster_returns(sensor.scan(p, alt, rng))
        self.assertGreater(len(hits), 0)                                 # the wheelhouse IS sensed
        a_des = np.array([3.0, 0.0])                                     # command straight at it
        a_safe = av.filter(p, v, a_des, hits)
        self.assertLess(a_safe[0], a_des[0] - 0.5)                       # toward-obstacle accel is bent back


class ContingencyTests(unittest.TestCase):
    def setUp(self):
        self.sup = ContingencySupervisor(ContingencyConfig(home=(0.0, 0.0)),
                                         GeofenceSpec(center=(0.0, 0.0), radius=10.0, z_max=15.0))

    def test_nominal_passthrough(self):
        c = self.sup.assess(HealthStatus(pos=np.array([1.0, 1.0, 5.0]), battery=0.9, comms_age=0.0))
        self.assertEqual(c.state, "NOMINAL")
        self.assertFalse(c.override)

    def test_low_battery_rtl(self):
        c = self.sup.assess(HealthStatus(pos=np.array([3.0, 0.0, 5.0]), battery=0.1))
        self.assertEqual(c.state, "LOW_BATTERY")
        self.assertTrue(c.override)
        np.testing.assert_allclose(c.target_xy, [0.0, 0.0])              # heads home

    def test_geofence_breach(self):
        c = self.sup.assess(HealthStatus(pos=np.array([20.0, 0.0, 5.0]), battery=0.9))
        self.assertEqual(c.state, "GEOFENCE")
        self.assertTrue(c.override)

    def test_lost_comms_loiter(self):
        c = self.sup.assess(HealthStatus(pos=np.array([1.0, 0.0, 5.0]), battery=0.9, comms_age=5.0))
        self.assertEqual(c.state, "LOST_COMMS")

    def test_rotor_out_has_top_priority_and_descends(self):
        # rotor failure + low battery at once -> ROTOR_OUT wins (highest priority) and sinks
        c = self.sup.assess(HealthStatus(pos=np.array([1.0, 0.0, 5.0]), battery=0.05, rotor_ok=False))
        self.assertEqual(c.state, "ROTOR_OUT")
        self.assertLess(c.vz, 0.0)                                       # bounded spinning descent

    def test_obstacle_abort_hysteresis(self):
        # below abort radius -> abort; between clear and abort -> stays aborting; above clear -> resumes
        s = self.sup
        self.assertEqual(s.assess(HealthStatus(pos=np.array([1.0, 0.0, 5.0]), nearest_obstacle=0.4)).state,
                         "OBSTACLE_ABORT")
        self.assertEqual(s.assess(HealthStatus(pos=np.array([1.0, 0.0, 5.0]), nearest_obstacle=0.9)).state,
                         "OBSTACLE_ABORT")                               # still latched (hysteresis)
        self.assertEqual(s.assess(HealthStatus(pos=np.array([1.0, 0.0, 5.0]), nearest_obstacle=2.0)).state,
                         "NOMINAL")                                      # cleared


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
class CollidableSuperstructureTests(unittest.TestCase):
    """The offshore superstructure is now a real collidable obstacle (P3) -> `hit_structure` termination."""

    def test_ground_world_has_no_obstacles(self):
        from drone_landing.sim.world import LandingWorld, LandingWorldConfig
        w = LandingWorld(LandingWorldConfig(world="x2_landing_ground"))
        self.assertEqual(len(w.obstacle_gids), 0)        # clear deck -> no keep-out geoms, no behaviour change

    def test_offshore_superstructure_is_collidable(self):
        import mujoco

        from drone_landing.sim.world import LandingWorld, LandingWorldConfig
        w = LandingWorld(LandingWorldConfig(world="x2_landing_offshore"))
        self.assertGreaterEqual(len(w.obstacle_gids), 3)  # wheelhouse(+base) + mast + bow
        w.reset(0)
        dp = w._deck_pos
        # drop the drone INTO the wheelhouse (deck-relative ~ (1.5, 0) at the wheelhouse height)
        w.data.qpos[w.qadr:w.qadr + 3] = [dp[0] + 1.5, dp[1], dp[2] + 0.6]
        w.data.qpos[w.qadr + 3:w.qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(w.model, w.data)
        hover = float(w.model.body_mass[w.drone_bid]) * 9.81 / 4.0
        term = None
        for _ in range(30):
            step = w.step(np.full(4, hover))
            if step.terminated:
                term = step.info["termination"]
                break
        self.assertEqual(term, "hit_structure")           # the structure is solid -> a real crash


if __name__ == "__main__":
    unittest.main()
