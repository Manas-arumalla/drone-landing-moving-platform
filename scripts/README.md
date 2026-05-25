# Scripts

The primary interface is the **`drone`** and **`swarm`** command-line tools (see the
[main README](../README.md)). The scripts here are generators, reproducible evaluations, and training
entry points.

## Media & assets
- `make_demo_gifs.py` — render the closed-loop demo GIFs into `media/`.
- `gen_aruco_deck.py` — generate the ArUco landing-pad texture.
- `gen_seakeeping_data.py` — synthesize the 6-DOF seakeeping deck-motion CSVs from the wave spectrum.

## Benchmarks & ablations
- `benchmark.py` — the controller × scenario × disturbance matrix written to `docs/BENCHMARK.md`.
- `eval_*.py` — per-capability A/B ablations backing the numbers in `docs/RESULTS.md`
  (green-deck, air-wake/DOB, reachability shield, tube-MPC, sense-and-avoid, contingency,
  cooperative/active/heterogeneous perception, consensus, platforms, seakeeping).
- `check_world.py` — quick smoke test that a world loads, hovers, and renders.
- `validate_landing.py` — standalone landing validation.

## Training
- `train_ppo.py`, `train_mujoco_ppo.py` — reinforcement-learning entry points (also exposed via `drone train`).

## `legacy/`
Earlier one-off run/watch/trace/render utilities, superseded by the unified `drone` / `swarm` CLI. Kept for
reference; prefer the CLI.
