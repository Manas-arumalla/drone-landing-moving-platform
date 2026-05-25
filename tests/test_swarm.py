"""Unit tests for the swarm coordination algorithms (CBF avoidance + landing scheduler)."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class CBFAvoidanceTests(unittest.TestCase):
    def test_passes_through_when_clear(self):
        import numpy as np

        from drone_landing_swarm import cbf_safe_velocity

        # no neighbour nearby -> desired velocity unchanged
        v = cbf_safe_velocity(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                              [(np.array([10.0, 0.0]), np.array([0.0, 0.0]))], d_min=0.5)
        np.testing.assert_allclose(v, [1.0, 0.0], atol=1e-6)

    def test_blocks_closing_velocity(self):
        import numpy as np

        from drone_landing_swarm import cbf_safe_velocity

        # two drones at the safety distance, desired velocity drives them together -> closing blocked
        p_i, p_j = np.array([0.0, 0.0]), np.array([0.6, 0.0])
        v = cbf_safe_velocity(p_i, np.array([2.0, 0.0]), [(p_j, np.array([0.0, 0.0]))],
                              d_min=0.6, alpha=2.0)
        # the component of velocity that reduces separation (toward +x) must be ~<= 0
        self.assertLessEqual(float(v[0]), 1e-6)

    def test_keeps_separation_in_rollout(self):
        import numpy as np

        from drone_landing_swarm import cbf_safe_velocity, min_pairwise_distance

        # two drones swapping positions head-on; the filter must keep them >= d_min apart
        d_min = 0.6
        p = [np.array([-2.0, 0.0]), np.array([2.0, 0.0])]
        goal = [np.array([2.0, 0.0]), np.array([-2.0, 0.0])]
        dt, worst = 0.05, float("inf")
        for _ in range(400):
            vdes = [np.clip(goal[k] - p[k], -1.0, 1.0) for k in range(2)]
            v = [cbf_safe_velocity(p[k], vdes[k],
                                   [(p[1 - k], np.zeros(2))], d_min=d_min, alpha=3.0, v_max=1.0)
                 for k in range(2)]
            for k in range(2):
                p[k] = p[k] + v[k] * dt
            worst = min(worst, min_pairwise_distance(p))
        # allow a small numerical margin below d_min (discrete time)
        self.assertGreater(worst, d_min - 0.1)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class SchedulerTests(unittest.TestCase):
    def test_clears_only_n_slots(self):
        import numpy as np

        from drone_landing_swarm import LandingScheduler, SchedulerConfig

        sch = LandingScheduler(4, SchedulerConfig(n_slots=1))
        cleared = sch.update(costs=np.array([3.0, 1.0, 2.0, 4.0]),
                             active=np.array([True] * 4), dt=0.1)
        self.assertEqual(len(cleared), 1)
        self.assertIn(1, cleared)  # lowest cost (readiest)

    def test_hysteresis_then_next_after_done(self):
        import numpy as np

        from drone_landing_swarm import LandingScheduler, SchedulerConfig

        sch = LandingScheduler(3, SchedulerConfig(n_slots=1, starvation_weight=0.0))
        active = np.array([True, True, True])
        first = sch.update(np.array([1.0, 2.0, 3.0]), active, 0.1)
        self.assertEqual(first, {0})
        # even if drone 1 becomes readier, the cleared drone stays cleared (hysteresis)
        again = sch.update(np.array([5.0, 0.1, 3.0]), active, 0.1)
        self.assertEqual(again, {0})
        # once drone 0 lands, the next readiest is cleared
        sch.mark_done(0)
        nxt = sch.update(np.array([5.0, 0.1, 3.0]), active, 0.1)
        self.assertEqual(nxt, {1})

    def test_hungarian_assignment(self):
        from drone_landing_swarm import optimal_assignment

        # 2 drones, 2 platforms; identity-ish cost -> diagonal assignment
        pairs = dict(optimal_assignment([[1.0, 9.0], [9.0, 1.0]]))
        self.assertEqual(pairs[0], 0)
        self.assertEqual(pairs[1], 1)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class CoordinatorTests(unittest.TestCase):
    def test_all_land_with_separation(self):
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator

        # 4 drones recovering onto a moving ship deck: all land and never collide
        coord = SwarmCoordinator(SwarmConfig(n_drones=4, scenario="ship", sea="moderate"))
        r = coord.run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class SensingTests(unittest.TestCase):
    """The no-cheats onboard sensing/comms layer (A1)."""

    def test_perfect_sensing_reproduces_truth(self):
        import numpy as np

        from drone_landing_swarm.sensing import SensingConfig, SwarmSensing

        rng = np.random.default_rng(0)
        s = SwarmSensing(3, SensingConfig.perfect(), rng)
        true_pos = {i: np.array([float(i), 0.0, 1.0]) for i in range(3)}
        true_vel = {i: np.array([0.1 * i, 0.0, 0.0]) for i in range(3)}
        deck = np.array([0.0, 0.0, 0.0])
        view = s.sense(true_pos, true_vel, deck, np.zeros(3), set(), float("inf"))
        for i in range(3):
            np.testing.assert_allclose(view["own_pos"][i], true_pos[i])
            np.testing.assert_allclose(view["deck"][i], deck)

    def test_realistic_sensing_is_noisy_and_bounded(self):
        import numpy as np

        from drone_landing_swarm.sensing import SensingConfig, SwarmSensing

        rng = np.random.default_rng(1)
        cfg = SensingConfig()
        s = SwarmSensing(4, cfg, rng)
        true_pos = {i: np.array([float(i), 0.0, 1.0]) for i in range(4)}
        true_vel = {i: np.zeros(3) for i in range(4)}
        deck = np.array([0.0, 0.0, 0.0])
        errs = []
        for _ in range(200):
            view = s.sense(true_pos, true_vel, deck, np.zeros(3), set(), float("inf"))
            errs.append(np.linalg.norm(view["own_pos"][0] - true_pos[0]))
        errs = np.array(errs)
        self.assertGreater(errs.mean(), 0.0)            # genuinely noisy (not truth)
        self.assertLess(errs.mean(), 0.5)               # but calibrated/bounded

    def test_comms_range_limits_neighbors(self):
        import numpy as np

        from drone_landing_swarm.sensing import SensingConfig, SwarmSensing

        # zero dropout/latency so the only filter is range; drones 0,1 close, 2 far away
        cfg = SensingConfig(pos_noise=0.0, vel_noise=0.0, deck_noise=0.0, rel_noise=0.0,
                            latency_steps=0, dropout_p=0.0)
        s = SwarmSensing(3, cfg, np.random.default_rng(2))
        true_pos = {0: np.zeros(3), 1: np.array([1.0, 0.0, 0.0]), 2: np.array([50.0, 0.0, 0.0])}
        true_vel = {i: np.zeros(3) for i in range(3)}
        view = s.sense(true_pos, true_vel, np.zeros(3), np.zeros(3), set(), comms_range=2.0)
        self.assertIn(1, view["neighbors"][0])          # in range
        self.assertNotIn(2, view["neighbors"][0])       # out of comms range

    def test_coordinator_lands_with_realistic_sensing(self):
        # AUDIT: the kinematic coordinator lands all drones with separation using ONLY onboard
        # estimates (no truth in the loop), under realistic noise + a finite comms range.
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator
        from drone_landing_swarm.sensing import SensingConfig

        coord = SwarmCoordinator(SwarmConfig(n_drones=4, scenario="ship", sea="moderate",
                                             sensing=SensingConfig(), comms_range=3.0))
        r = coord.run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class SafetyLayerTests(unittest.TestCase):
    """The non-bypassable CBF safety filter + its formal certificate (A3)."""

    def test_invariance_precondition_enforced(self):
        from drone_landing_swarm import SafetySpec

        # alpha*dt must be < 1 for discrete-time forward invariance; the spec refuses otherwise.
        SafetySpec(alpha=3.0, dt=0.05)                       # 0.15 -> fine
        with self.assertRaises(ValueError):
            SafetySpec(alpha=30.0, dt=0.05)                  # 1.5 -> rejected

    def test_passes_through_when_clear(self):
        import numpy as np

        from drone_landing_swarm import SafetyFilter, SafetySpec

        f = SafetyFilter(SafetySpec(d_min=0.7, alpha=3.0, dt=0.05, v_max=2.0))
        v = f.filter(np.zeros(2), np.array([1.0, 0.0]),
                     [(np.array([10.0, 0.0]), np.zeros(2))])
        np.testing.assert_allclose(v, [1.0, 0.0], atol=1e-6)
        self.assertEqual(f.report.activations, 0)

    def test_certificate_nonnegative_with_exact_inputs(self):
        # The provable property: with exact (perfect) inputs and alpha*dt<1, every one-step certificate
        # h+ stays >= 0 along a head-on closing rollout -> the safe set is forward-invariant.
        import numpy as np

        from drone_landing_swarm import SafetyFilter, SafetySpec

        spec = SafetySpec(d_min=0.6, alpha=3.0, dt=0.05, v_max=1.0)
        fa, fb = SafetyFilter(spec), SafetyFilter(spec)
        pa, pb = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
        for _ in range(400):
            va = fa.filter(pa, pb - pa, [(pb, np.zeros(2))])   # each avoids the other's exact state
            vb = fb.filter(pb, pa - pb, [(pa, np.zeros(2))])
            pa, pb = pa + va * spec.dt, pb + vb * spec.dt
        self.assertGreaterEqual(fa.report.min_certificate, -1e-6)
        self.assertGreaterEqual(fb.report.min_certificate, -1e-6)
        self.assertGreater(float(np.linalg.norm(pa - pb)), spec.d_min - 1e-3)

    def test_verify_separation_passes_on_feasible_swarm(self):
        from drone_landing_swarm import (SensingConfig, SwarmConfig, SwarmCoordinator,
                                         verify_separation)

        mk = lambda: SwarmCoordinator(SwarmConfig(n_drones=5, scenario="ship", sea="moderate",
                                                  sensing=SensingConfig()))
        rep = verify_separation(mk, range(8))
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["n_violations"], 0)


class StaticObstacleTests(unittest.TestCase):
    """P3 sense-and-avoid in the swarm: SENSED static obstacles folded into the non-bypassable CBF."""

    def test_safety_filter_deflects_static_obstacle(self):
        import numpy as np

        from drone_landing_swarm import SafetyFilter, SafetySpec

        f = SafetyFilter(SafetySpec(d_min=0.7, alpha=3.0, dt=0.05, v_max=1.5))
        p, v_des = np.array([0.0, 0.0, 1.0]), np.array([1.2, 0.0, 0.0])  # heading straight at it
        v = f.filter(p, v_des, [], obstacles=[(np.array([1.0, 0.0, 1.0]), 0.8)])
        self.assertLess(v[0], v_des[0] - 0.1)                 # toward-obstacle velocity bent back
        self.assertEqual(f.report.activations, 1)

    def test_static_obstacle_empty_is_passthrough(self):
        import numpy as np

        from drone_landing_swarm import SafetyFilter, SafetySpec

        f = SafetyFilter(SafetySpec(d_min=0.7, alpha=3.0, dt=0.05, v_max=2.0))
        v = f.filter(np.zeros(3), np.array([1.0, 0.0, 0.0]), [], obstacles=[])
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-6)   # no obstacles -> unchanged

    def test_swarm_keeps_clear_of_obstacle_and_lands(self):
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator

        # an obstacle fore of the deck (deck-relative (1.5,0), keep-out 0.8 m) — drones must skirt it
        cfg = SwarmConfig(n_drones=4, scenario="ground", obstacles=((1.5, 0.0, 0.8),))
        r = SwarmCoordinator(cfg).run(seed=0)
        self.assertIsNotNone(r["min_obstacle_clearance"])
        self.assertTrue(r["obstacle_ok"])                     # kept clear of the static obstacle
        self.assertGreaterEqual(r["n_landed"], 3)             # and still recovered the swarm

    def test_multideck_keeps_clear_of_obstacles_and_lands(self):
        from drone_landing_swarm.multi_deck import MultiDeckConfig, MultiDeckCoordinator

        # each vessel carries its superstructure; drones avoid their assigned vessel's obstacles
        cfg = MultiDeckConfig(n_drones=6, n_decks=2, scenario="ship",
                              obstacles=((2.1, 0.0, 0.9), (2.75, 0.0, 1.0)))
        r = MultiDeckCoordinator(cfg).run(seed=0)
        self.assertTrue(r["obstacle_ok"])
        self.assertGreaterEqual(r["n_landed"], 5)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class ConsensusTests(unittest.TestCase):
    """Cooperative consensus deck estimation (A2)."""

    def test_blind_drone_gets_estimate_from_neighbors(self):
        # A drone with NO direct deck view still converges near the truth via consensus with a neighbour
        # that does see the deck.
        import numpy as np

        from drone_landing_swarm.consensus import ConsensusConfig, ConsensusDeckEstimator

        f = ConsensusDeckEstimator(2, dt=0.05, config=ConsensusConfig())
        f.reset()
        deck = np.array([1.0, -0.5, 0.2])
        rng = np.random.default_rng(0)
        for _ in range(300):
            meas = {0: deck + rng.normal(0, 0.05, 3), 1: None}   # drone 1 is blind
            stds = {0: 0.05, 1: 1.0}
            f.step(meas, stds, {0: [1], 1: [0]})                 # they are neighbours
        self.assertLess(float(np.linalg.norm(f.deck_pos(1) - deck)), 0.2)   # blind drone recovered

    def test_consensus_beats_raw_measurement(self):
        # Network-mean fused error < mean raw single-drone measurement error.
        import numpy as np

        from drone_landing_swarm.consensus import ConsensusDeckEstimator
        from drone_landing_swarm.sensing import SensingConfig, SwarmSensing

        n, dt = 4, 0.05
        rng = np.random.default_rng(1)
        sens = SwarmSensing(n, SensingConfig(), rng)
        filt = ConsensusDeckEstimator(n, dt); filt.reset()
        offs = {0: np.array([0.4, 0.0, 1.0]), 1: np.array([2.0, 0.0, 1.4]),
                2: np.array([3.8, 0.5, 1.8]), 3: np.array([5.0, 0.0, 2.0])}
        raw, fused = [], []
        for k in range(400):
            deck = np.array([0.8 * np.sin(0.05 * k), 0.0, 0.0])
            pos = {i: deck + offs[i] for i in range(n)}
            view = sens.sense(pos, {i: np.zeros(3) for i in range(n)}, deck, np.zeros(3),
                              set(), float("inf"))
            nbr = {i: list(view["neighbors"].get(i, {}).keys()) for i in range(n)}
            fx = filt.step(view["deck_meas"], view["deck_std"], nbr)
            if k > 60:
                for i in range(n):
                    if view["deck_meas"][i] is not None:
                        raw.append(float(np.linalg.norm(view["deck_meas"][i] - deck)))
                    fused.append(float(np.linalg.norm(fx[i][:3] - deck)))
        self.assertLess(float(np.mean(fused)), float(np.mean(raw)))

    def test_coordinator_lands_with_consensus_on(self):
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator

        coord = SwarmCoordinator(SwarmConfig(n_drones=4, scenario="ship", sea="moderate",
                                             consensus=True))
        r = coord.run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class GNNPolicyTests(unittest.TestCase):
    """The GNN-MARL policy (A4): permutation-invariant, size-agnostic graph features."""

    def test_graph_obs_shape(self):
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator
        from drone_landing_swarm.marl_gnn import graph_obs_dim

        coord = SwarmCoordinator(SwarmConfig(n_drones=6, scenario="ship"))
        coord.reset(0)
        obs = coord.local_obs_graph(0, max_neighbors=8)
        self.assertEqual(obs.shape, (graph_obs_dim(8),))

    def test_features_permutation_invariant_and_size_agnostic(self):
        import numpy as np
        import torch
        from gymnasium import spaces

        from drone_landing_swarm.marl_gnn import (EGO_DIM, MAX_NEIGHBORS, NB_DIM, build_extractor_class,
                                                  graph_obs_dim)

        Ext = build_extractor_class()
        sp = spaces.Box(-np.inf, np.inf, shape=(graph_obs_dim(),), dtype=np.float32)
        ext = Ext(sp, features_dim=64)
        ext.eval()
        rng = np.random.default_rng(0)
        ego = rng.normal(size=EGO_DIM).astype(np.float32)
        feats = np.zeros((MAX_NEIGHBORS, NB_DIM), np.float32)
        mask = np.zeros(MAX_NEIGHBORS, np.float32)
        for k in range(4):
            feats[k] = rng.normal(size=NB_DIM)
            mask[k] = 1.0
        obs = np.concatenate([ego, feats.ravel(), mask]).astype(np.float32)
        perm = feats.copy(); perm[:4] = feats[[3, 1, 0, 2]]
        obs2 = np.concatenate([ego, perm.ravel(), mask]).astype(np.float32)
        with torch.no_grad():
            f1 = ext(torch.tensor(obs[None]))
            f2 = ext(torch.tensor(obs2[None]))
            # size-agnostic: a different valid-count still yields finite features
            m2 = mask.copy(); m2[2:] = 0
            f3 = ext(torch.tensor(np.concatenate([ego, feats.ravel(), m2]).astype(np.float32)[None]))
        self.assertTrue(torch.allclose(f1, f2, atol=1e-5))     # permutation-invariant
        self.assertTrue(bool(torch.isfinite(f3).all()))        # size-agnostic, finite

    def test_graph_env_steps(self):
        import numpy as np

        from drone_landing_swarm.marl_gnn import SwarmMARLGraphEnv, graph_obs_dim

        env = SwarmMARLGraphEnv(seed=0)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (graph_obs_dim(),))
        obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(obs.shape, (graph_obs_dim(),))
        self.assertIn("min_d", info)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is not installed")
class MultiDeckTests(unittest.TestCase):
    """K moving platforms + dynamic re-tasking (A5)."""

    def test_all_land_on_k_decks_balanced(self):
        from drone_landing_swarm import MultiDeckConfig, MultiDeckCoordinator

        r = MultiDeckCoordinator(MultiDeckConfig(n_drones=9, n_decks=3, scenario="ship",
                                                 sea="moderate")).run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])
        # the Hungarian start balances the load across decks (3 each for 9/3)
        self.assertEqual(sorted(r["per_deck_landed"].values()), [3, 3, 3])

    def test_retasking_recovers_fouled_deck(self):
        # A deck closes mid-recovery: with re-tasking the stranded drones recover onto the other decks;
        # without it they are stranded (fewer than all land).
        from drone_landing_swarm import MultiDeckConfig, MultiDeckCoordinator

        on = MultiDeckCoordinator(MultiDeckConfig(n_drones=9, n_decks=3, scenario="ship",
                                                  offline_deck=2, offline_time=4.0,
                                                  max_time=160.0)).run(seed=0)
        off = MultiDeckCoordinator(MultiDeckConfig(n_drones=9, n_decks=3, scenario="ship",
                                                   offline_deck=2, offline_time=4.0, max_time=160.0,
                                                   reassign_dt=1e9, reassign_hysteresis=1e9)).run(seed=0)
        self.assertTrue(on["all_landed"])              # re-tasking saves the stranded drones
        self.assertGreater(on["n_reassign"], 0)
        self.assertLess(off["n_landed"], 9)            # without it, the fouled deck's drones strand


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("cv2"),
                     "mujoco/opencv not installed")
class CooperativePerceptionTests(unittest.TestCase):
    """Real per-drone onboard vision (CP): detection accuracy + blind-drone recovery via sharing."""

    def test_vision_detects_pad_and_blind_when_far(self):
        import mujoco
        import numpy as np

        from drone_landing.sim.platforms import ShipDeckMotion, sea_state
        from drone_landing_swarm.vision import SwarmVision, VisionConfig
        from drone_landing_swarm.world import SwarmMujocoWorld

        w = SwarmMujocoWorld(2, spawn_radius=2.0, spawn_alt=2.0)
        d0 = ShipDeckMotion(sea_state("calm")).reset(np.random.default_rng(0))
        w.reset(d0, np.random.default_rng(0))
        dp, _ = w.deck_state()
        vis = SwarmVision(w, VisionConfig(period=1))
        try:
            # drone 0 over the pad -> sees it; drone 1 far out -> blind
            w.data.qpos[w.qadr[0]:w.qadr[0] + 3] = [dp[0], dp[1], dp[2] + 2.0]
            w.data.qpos[w.qadr[1]:w.qadr[1] + 3] = [dp[0] + 4.0, dp[1], dp[2] + 2.0]
            for i in (0, 1):
                w.data.qpos[w.qadr[i] + 3:w.qadr[i] + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(w.model, w.data)
            own = {i: w.drone_pos(i) for i in range(2)}
            fixes = vis.sense(own, float(dp[2]), set())
            self.assertIsNotNone(fixes[0])
            self.assertLess(float(np.linalg.norm(fixes[0][:2] - dp[:2])), 0.2)   # accurate fix
            self.assertIsNone(fixes[1])                                          # blind
        finally:
            vis.close()

    def test_blind_drone_recovers_deck_via_shared_vision(self):
        import mujoco
        import numpy as np

        from drone_landing.sim.platforms import ShipDeckMotion, sea_state
        from drone_landing_swarm.consensus import ConsensusDeckEstimator
        from drone_landing_swarm.vision import SwarmVision, VisionConfig
        from drone_landing_swarm.world import SwarmMujocoWorld

        w = SwarmMujocoWorld(2, spawn_radius=2.0, spawn_alt=2.0)
        d0 = ShipDeckMotion(sea_state("calm")).reset(np.random.default_rng(0))
        w.reset(d0, np.random.default_rng(0))
        dp, _ = w.deck_state()
        vis = SwarmVision(w, VisionConfig(period=1))
        try:
            w.data.qpos[w.qadr[0]:w.qadr[0] + 3] = [dp[0], dp[1], dp[2] + 2.0]      # sees pad
            w.data.qpos[w.qadr[1]:w.qadr[1] + 3] = [dp[0] + 4.0, dp[1], dp[2] + 2.0]  # blind
            for i in (0, 1):
                w.data.qpos[w.qadr[i] + 3:w.qadr[i] + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(w.model, w.data)
            own = {i: w.drone_pos(i) for i in range(2)}
            prior = np.array([dp[0] + 3.0, dp[1] + 3.0, dp[2]])     # wrong rough prior

            def blind_err(cooperative):
                filt = ConsensusDeckEstimator(2, 0.05); filt.reset(prior)
                for _ in range(120):
                    fixes = vis.sense(own, float(dp[2]), set()); vis._k = 0
                    stds = {i: 0.05 if fixes.get(i) is not None else 1.0 for i in range(2)}
                    nbrs = {0: [1] if cooperative else [], 1: [0] if cooperative else []}
                    fused = filt.step(fixes, stds, nbrs)
                return float(np.linalg.norm(fused[1][:2] - dp[:2]))   # blind drone (1) error

            self.assertGreater(blind_err(False), 2.0)     # isolated: stuck at the wrong prior
            self.assertLess(blind_err(True), 1.0)         # cooperative: recovered from neighbour's vision
        finally:
            vis.close()

    def test_cameraless_drone_recovers_from_camera_drone(self):
        # Heterogeneous fleet (P2.4): drone 1 carries NO camera at all (not merely out of FOV). It must
        # recover the deck purely from the camera drone's shared fixes via consensus.
        import mujoco
        import numpy as np

        from drone_landing.sim.platforms import ShipDeckMotion, sea_state
        from drone_landing_swarm.consensus import ConsensusDeckEstimator
        from drone_landing_swarm.vision import SwarmVision, VisionConfig
        from drone_landing_swarm.world import SwarmMujocoWorld

        w = SwarmMujocoWorld(2, spawn_radius=2.0, spawn_alt=2.0)
        d0 = ShipDeckMotion(sea_state("calm")).reset(np.random.default_rng(0))
        w.reset(d0, np.random.default_rng(0))
        dp, _ = w.deck_state()
        vis = SwarmVision(w, VisionConfig(period=1), camera_ids={0})   # only drone 0 has a camera
        try:
            w.data.qpos[w.qadr[0]:w.qadr[0] + 3] = [dp[0], dp[1], dp[2] + 2.0]      # sees pad
            w.data.qpos[w.qadr[1]:w.qadr[1] + 3] = [dp[0], dp[1] + 0.3, dp[2] + 2.0]  # over pad but NO camera
            for i in (0, 1):
                w.data.qpos[w.qadr[i] + 3:w.qadr[i] + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(w.model, w.data)
            own = {i: w.drone_pos(i) for i in range(2)}
            fixes = vis.sense(own, float(dp[2]), set())
            self.assertIsNotNone(fixes[0])    # camera drone sees the pad
            self.assertIsNone(fixes[1])       # camera-less drone returns no fix even though it is over the pad
            prior = np.array([dp[0] + 3.0, dp[1] + 3.0, dp[2]])

            def cl_err(cooperative):
                filt = ConsensusDeckEstimator(2, 0.05); filt.reset(prior)
                for _ in range(120):
                    f = vis.sense(own, float(dp[2]), set()); vis._k = 0
                    stds = {i: 0.05 if f.get(i) is not None else 1.0 for i in range(2)}
                    nbrs = {0: [1] if cooperative else [], 1: [0] if cooperative else []}
                    fused = filt.step(f, stds, nbrs)
                return float(np.linalg.norm(fused[1][:2] - dp[:2]))

            self.assertGreater(cl_err(False), 2.0)    # isolated camera-less drone: stuck
            self.assertLess(cl_err(True), 1.0)        # cooperative: recovered from the camera drone
        finally:
            vis.close()


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
class MujocoSwarmTests(unittest.TestCase):
    def test_mujoco_swarm_all_land_with_separation(self):
        # REAL MuJoCo physics + contact: 3 drones recover onto a moving ship deck, all land, no collisions
        from drone_landing_swarm.coordinator import SwarmConfig
        from drone_landing_swarm.mujoco_runner import MujocoSwarmCoordinator

        r = MujocoSwarmCoordinator(SwarmConfig(n_drones=3, scenario="ship", sea="moderate",
                                               max_time=60.0)).run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])

    def test_ground_scenario_runs(self):
        from drone_landing_swarm import SwarmConfig, SwarmCoordinator

        coord = SwarmCoordinator(SwarmConfig(n_drones=3, scenario="ground"))
        r = coord.run(seed=1)
        self.assertEqual(r["n_landed"], 3)

    def test_mujoco_multideck_all_land_balanced(self):
        # P4.1: REAL physics multi-deck -- 4 drones recover onto 2 moving ship decks (2 per deck), all
        # land beside each other (distinct on-deck spots), separation kept, no collisions.
        from drone_landing_swarm.multi_deck import MultiDeckConfig
        from drone_landing_swarm.multideck_runner import MujocoMultiDeckCoordinator

        r = MujocoMultiDeckCoordinator(MultiDeckConfig(
            n_drones=4, n_decks=2, scenario="ship", sea="moderate", max_time=80.0,
            deck_spacing=4.0)).run(seed=0)
        self.assertTrue(r["all_landed"])
        self.assertTrue(r["separation_ok"])
        self.assertEqual(sum(r["per_deck_landed"].values()), 4)        # balanced across the two decks


if __name__ == "__main__":
    unittest.main()
