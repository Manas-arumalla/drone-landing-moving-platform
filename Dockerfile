# One-command reproduction: clean machine -> full test suite green -> runnable demos.
#
#   docker build -t drone-landing .
#   docker run --rm drone-landing                                  # run the test suite
#   docker run --rm drone-landing drone run ground --episodes 2    # headless demo episodes
#   docker run --rm drone-landing drone list                       # scenarios / controllers / presets
#
# Rendering is software (OSMesa) so everything works headless without a GPU.

FROM python:3.10-slim

# MuJoCo headless rendering (OSMesa software GL + Mesa EGL fallback) and the
# shared libraries opencv's Linux wheels link against.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        libosmesa6 libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch: several GB smaller than the default CUDA wheels and all the
# bundled policies run fine on CPU. Installed first so the extras below see it satisfied.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Only what the package needs at runtime + tests (media/ and .git are not copied).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY assets ./assets
COPY runs ./runs
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs

RUN pip install -e ".[vision,mpc,rl,viz,dev]"

# Default command proves the install: the full pytest suite, headless.
CMD ["pytest", "-q"]
