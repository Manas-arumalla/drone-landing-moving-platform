"""Enable ``python -m drone_landing_swarm ...`` as an alias for the ``swarm`` CLI."""

from drone_landing_swarm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
