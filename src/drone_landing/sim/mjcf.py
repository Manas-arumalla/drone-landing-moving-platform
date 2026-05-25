"""Locate and load the project's MuJoCo world files.

World MJCF files live in ``assets/mujoco/worlds`` and reference shared meshes in
``assets/mujoco/meshes``. This module resolves those paths independent of the current working
directory so worlds load the same way from scripts, tests, and the CLI.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the directory containing ``assets`` and ``src``)."""
    # src/drone_landing/sim/mjcf.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


WORLDS_DIR = repo_root() / "assets" / "mujoco" / "worlds"


def world_path(name: str) -> Path:
    """Resolve a world name (with or without ``.xml``) to an absolute path."""
    candidate = WORLDS_DIR / name
    if candidate.suffix != ".xml":
        candidate = candidate.with_suffix(".xml")
    if not candidate.exists():
        available = ", ".join(sorted(p.name for p in WORLDS_DIR.glob("*.xml"))) or "(none)"
        raise FileNotFoundError(f"World '{name}' not found in {WORLDS_DIR}. Available: {available}")
    return candidate


def load_model(name: str = "x2_landing_ground"):
    """Compile and return an ``mujoco.MjModel`` for the named world."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(world_path(name)))
