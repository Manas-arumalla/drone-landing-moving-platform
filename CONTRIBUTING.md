# Contributing

Thanks for your interest in this project! It is primarily a research / portfolio
codebase, but issues, suggestions, and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/Manas-arumalla/drone-landing-moving-platform.git
cd drone-landing-moving-platform
python -m pip install -e ".[vision,mpc,rl,viz,dev]"   # editable install + dev tools
python -m pytest -q                                    # run the test suite (should be all green)
```

This registers the `drone` and `swarm` command-line entry points. See the
[README](README.md) for the full CLI.

## Ground rules

This project follows a strict **no-cheats realism** principle (see
[docs/REALISM_CHARTER.md](docs/REALISM_CHARTER.md)): the deployable stack — controllers,
estimators, and the swarm coordination layer — must run on **onboard, sensor-derived
state only**. Ground truth from the simulator may be used for physics, contact, and
metrics, but **never inside a control or coordination decision**. Sensor *models*
(IMU, rangefinder, optical flow, camera) that return truth plus realistic noise/bias
are legitimate measurements; reading the simulator's exact pose into a controller is not.

Please preserve this when contributing.

## Conventions

- **Style:** `ruff` for linting/formatting; type hints throughout. Run `ruff check .` before a PR.
- **Tests:** add or update tests under `tests/` for any behavioral change; keep the suite green.
- **Additive changes:** new capabilities should be opt-in (a flag / subclass / module) and must
  not regress the validated baselines. Re-run the relevant scenario benchmarks and note the numbers.
- **Honesty:** results in the docs are reported as measured, including negative results and known
  limitations. Keep it that way — document what *doesn't* work alongside what does.

## Reporting results

When a change affects landing performance, include the before/after numbers
(success rate, touchdown error, impact velocity) from `drone run <scenario> --episodes N`
or `scripts/benchmark.py` in your PR description.
