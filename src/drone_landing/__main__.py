"""Enable ``python -m drone_landing ...`` as an alias for the ``dl`` CLI."""

from drone_landing.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
