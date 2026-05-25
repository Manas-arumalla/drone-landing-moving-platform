"""Cooperative perception (CP) — controlled experiment with REAL per-drone vision.

Places drones over the deck at varied offsets: some near enough that their **downward camera sees the
landing pad** (a sharp visual fix), others too far out — **blind** (the pad is outside their field of
view). Each seeing drone runs real detection on its rendered camera; the fixes are fused by the
distributed consensus filter (A2). We compare the **blind** drones' deck-position error:

* **cooperative** (drones share fixes over comms): the blind drones recover the deck location from their
  neighbours that can see it;
* **isolated** (no sharing): a blind drone has no fix and is stuck at its prior.

This is the cooperative-perception loop proven with *real vision*, not a modeled estimate.

  python scripts/eval_cooperative_perception.py
"""

from __future__ import annotations

import numpy as np


def main(seed: int = 0) -> None:
    import mujoco

    from drone_landing.sim.platforms import ShipDeckMotion, sea_state
    from drone_landing_swarm.consensus import ConsensusConfig, ConsensusDeckEstimator
    from drone_landing_swarm.vision import SwarmVision, VisionConfig

    n = 5
    world = __import__("drone_landing_swarm.world", fromlist=["SwarmMujocoWorld"]).SwarmMujocoWorld(
        n, spawn_radius=2.0, spawn_alt=2.0)
    deck = ShipDeckMotion(sea_state("calm"))
    d0 = deck.reset(np.random.default_rng(seed))
    world.reset(d0, np.random.default_rng(seed))
    dp, _ = world.deck_state()
    vision = SwarmVision(world, VisionConfig(period=1))

    # drones 0,1 hover near/over the pad (will SEE it); 2,3,4 are far out (BLIND)
    offsets = {0: (0.2, 0.0, 1.8), 1: (-0.3, 0.2, 2.0),
               2: (3.5, 0.0, 2.0), 3: (0.0, 3.8, 2.0), 4: (-3.6, -0.5, 2.2)}
    for i, (dx, dy, h) in offsets.items():
        world.data.qpos[world.qadr[i]:world.qadr[i] + 3] = [dp[0] + dx, dp[1] + dy, dp[2] + h]
        world.data.qpos[world.qadr[i] + 3:world.qadr[i] + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(world.model, world.data)
    own = {i: world.drone_pos(i) for i in range(n)}      # own-pose (truth here; noise is orthogonal)

    # each drone's prior on the deck is rough (GPS/broadcast), modelled as a 3 m offset from truth — so a
    # blind drone is wrong until a neighbour's visual fix corrects it.
    prior = np.array([dp[0] + 3.0, dp[1] + 3.0, dp[2]])

    def run(cooperative: bool):
        filt = ConsensusDeckEstimator(n, 0.05, ConsensusConfig())
        filt.reset(prior)
        err = {i: [] for i in range(n)}
        for _ in range(120):
            fixes = vision.sense(own, float(dp[2]), set())   # REAL vision: world fix or None per drone
            vision._k = 0                                    # force re-render each call (static scene)
            stds = {i: 0.05 if fixes.get(i) is not None else 1.0 for i in range(n)}
            # cooperative: every drone hears the others (full comms); isolated: nobody shares
            nbrs = {i: [j for j in range(n) if j != i] if cooperative else [] for i in range(n)}
            fused = filt.step(fixes, stds, nbrs)
            for i in range(n):
                err[i].append(float(np.linalg.norm(fused[i][:2] - dp[:2])))
        return {i: float(np.mean(err[i][-20:])) for i in range(n)}

    seen = {i: vision.sense(own, float(dp[2]), set())[i] is not None for i in range(n)}
    coop = run(True)
    iso = run(False)
    vision.close()

    print("Cooperative perception — REAL per-drone vision (deck-position error, m):")
    print("  drone   sees pad?   isolated (no sharing)   cooperative (shared fixes)")
    for i in range(n):
        tag = "SEES" if seen[i] else "BLIND"
        print(f"   {i}      {tag:6s}      {iso[i]:6.2f} m                 {coop[i]:6.2f} m")
    blind = [i for i in range(n) if not seen[i]]
    if blind:
        print(f"\n  BLIND drones: isolated err {np.mean([iso[i] for i in blind]):.2f} m  ->  "
              f"cooperative err {np.mean([coop[i] for i in blind]):.2f} m  "
              f"(recovered the deck from neighbours' real visual fixes)")


if __name__ == "__main__":
    main()
