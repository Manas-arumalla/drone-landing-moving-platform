"""P2.4 — heterogeneous cooperative perception: camera drones guide CAMERA-LESS drones.

Unlike `eval_cooperative_perception.py` (where "blind" drones simply have the pad outside their FOV), here
some drones carry **no camera at all** -- a genuinely heterogeneous fleet. The camera-equipped drones
detect the pad on their rendered cameras and broadcast world-frame fixes; the consensus filter (A2) fuses
them, and the **camera-less drones recover the deck location entirely from their neighbours**. We compare
the camera-less drones' deck-position error with sharing (cooperative) vs without (isolated -> stuck).

  python scripts/eval_heterogeneous_perception.py
"""

from __future__ import annotations

import numpy as np


def main(seed: int = 0) -> None:
    import mujoco

    from drone_landing.sim.platforms import ShipDeckMotion, sea_state
    from drone_landing_swarm.consensus import ConsensusDeckEstimator
    from drone_landing_swarm.vision import SwarmVision, VisionConfig
    from drone_landing_swarm.world import SwarmMujocoWorld

    n = 5
    camera_ids = {0, 1}                       # only drones 0,1 carry a camera; 2,3,4 are camera-less
    world = SwarmMujocoWorld(n, spawn_radius=2.0, spawn_alt=2.0)
    d0 = ShipDeckMotion(sea_state("calm")).reset(np.random.default_rng(seed))
    world.reset(d0, np.random.default_rng(seed))
    dp, _ = world.deck_state()
    vision = SwarmVision(world, VisionConfig(period=1), camera_ids=camera_ids)

    # camera drones hover over the deck (they SEE the pad); camera-less drones sit out at the holding ring
    offsets = {0: (0.2, 0.0, 1.8), 1: (-0.2, 0.3, 2.0),
               2: (2.0, 0.0, 2.0), 3: (0.0, 2.0, 2.0), 4: (-2.0, -0.4, 2.2)}
    for i, (dx, dy, h) in offsets.items():
        world.data.qpos[world.qadr[i]:world.qadr[i] + 3] = [dp[0] + dx, dp[1] + dy, dp[2] + h]
        world.data.qpos[world.qadr[i] + 3:world.qadr[i] + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(world.model, world.data)
    own = {i: world.drone_pos(i) for i in range(n)}
    prior = np.array([dp[0] + 3.0, dp[1] + 3.0, dp[2]])     # rough (wrong) shared prior on the deck

    def run(cooperative: bool):
        filt = ConsensusDeckEstimator(n, 0.05); filt.reset(prior)
        err = {i: [] for i in range(n)}
        for _ in range(120):
            fixes = vision.sense(own, float(dp[2]), set()); vision._k = 0
            stds = {i: 0.05 if fixes.get(i) is not None else 1.0 for i in range(n)}
            nbrs = {i: [j for j in range(n) if j != i] if cooperative else [] for i in range(n)}
            fused = filt.step(fixes, stds, nbrs)
            for i in range(n):
                err[i].append(float(np.linalg.norm(fused[i][:2] - dp[:2])))
        return {i: float(np.mean(err[i][-20:])) for i in range(n)}

    coop, iso = run(True), run(False)
    vision.close()
    print("Heterogeneous cooperative perception -- camera drones {0,1} guide camera-less {2,3,4}:")
    print("  drone   camera?    isolated err    cooperative err")
    for i in range(n):
        tag = "CAMERA" if i in camera_ids else "none"
        print(f"   {i}      {tag:7s}    {iso[i]:6.2f} m        {coop[i]:6.2f} m")
    cl = [i for i in range(n) if i not in camera_ids]
    print(f"\n  CAMERA-LESS drones: isolated {np.mean([iso[i] for i in cl]):.2f} m  ->  "
          f"cooperative {np.mean([coop[i] for i in cl]):.2f} m  "
          f"(recovered the deck purely from the camera drones' shared fixes)")


if __name__ == "__main__":
    main()
