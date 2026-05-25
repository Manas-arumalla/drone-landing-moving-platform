"""Phase 5 / A6 learned active cooperative perception tests."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from drone_landing_swarm.active_perception import (
    bandwidth_pareto,
    fuse_equal,
    fuse_inverse_variance,
    sample_episode,
    select_broadcasters,
)


class GatingTests(unittest.TestCase):
    """Value-of-information communication gating (numpy, no torch)."""

    def test_selects_high_value_messages(self):
        conf = np.array([0.1, 0.9, 0.2, 0.8])
        link = np.ones(4)
        sel = select_broadcasters(conf, link, budget=2)
        self.assertEqual(set(sel.tolist()), {1, 3})            # the two most reliable

    def test_budget_clamped(self):
        sel = select_broadcasters(np.array([0.5, 0.5]), np.array([1.0, 1.0]), budget=9)
        self.assertEqual(len(sel), 2)
        sel0 = select_broadcasters(np.array([0.5]), np.array([1.0]), budget=0)
        self.assertEqual(len(sel0), 1)                         # always at least one

    def test_active_beats_random_under_budget(self):
        bp = bandwidth_pareto(episodes=2000, seed=1)
        for b in (1, 2, 3):
            self.assertLess(bp[b]["active"], bp[b]["random"])  # informed selection wins at every budget
        self.assertLess(bp[3]["active"], bp["all"] * 1.15)     # near-full accuracy at 3 messages


class FusionBaselineTests(unittest.TestCase):
    def test_inverse_variance_fooled_by_confident_outlier(self):
        # one confident-but-wrong fix drags inverse-variance off; the median-based learner is what fixes it
        fixes = np.array([[0.0, 0.0], [0.02, -0.01], [2.0, 2.0]])   # last is a confident outlier
        conf = np.array([0.9, 0.9, 0.95])
        true = np.zeros(2)
        self.assertGreater(np.linalg.norm(fuse_inverse_variance(fixes, conf) - true), 0.5)
        # equal mean is also dragged; the point is neither rejects the outlier (the learner does — below)
        self.assertGreater(np.linalg.norm(fuse_equal(fixes) - true), 0.4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class LearnedFusionTests(unittest.TestCase):
    def test_attention_permutation_invariant(self):
        import torch

        from drone_landing_swarm.active_perception import _build_module, message_features
        torch.manual_seed(0)
        model = _build_module()()
        rng = np.random.default_rng(0)
        fixes, conf, link, rp, _ = sample_episode(rng, n_max=6, outlier_p=0.3)
        feats = message_features(fixes, conf, link, rp)
        with torch.no_grad():
            e1, _ = model(torch.from_numpy(feats[None]), torch.from_numpy(fixes[None]),
                          torch.ones(1, len(fixes)))
            perm = rng.permutation(len(fixes))
            e2, _ = model(torch.from_numpy(feats[perm][None]), torch.from_numpy(fixes[perm][None]),
                          torch.ones(1, len(fixes)))
        np.testing.assert_allclose(e1.numpy(), e2.numpy(), atol=1e-5)   # invariant to message order

    def test_learned_beats_inverse_variance_on_outliers(self):
        from drone_landing_swarm.active_perception import evaluate, train_fusion
        learned = train_fusion(steps=1500, seed=0)
        r = evaluate(learned, episodes=2000, outlier_p=0.25)
        self.assertLess(r["learned"], r["inverse_variance"])   # rejects confident outliers
        self.assertLess(r["learned"], 0.30)                    # and lands a good absolute error


if __name__ == "__main__":
    unittest.main()
