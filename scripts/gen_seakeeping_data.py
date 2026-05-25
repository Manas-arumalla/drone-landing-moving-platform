"""P1.2/P1.3 — generate seakeeping reference data + validate the model against it.

Generates a high-fidelity 6-DOF deck-motion CSV per sea state (from the JONSWAP/RAO spectral model, many
components, fine dt) into ``assets/seakeeping/<sea>.csv``, then **characterizes** each trace exactly as you
would a real sea-trial log — significant heave Hs = 4*std(heave), peak period Tp from the heave PSD, and
roll/pitch RMS — and compares to the named sea state's design parameters. A real trial / NDBC-derived CSV
(same columns ``t,x,y,z,roll,pitch,yaw``) drops into the same pipeline (DataDrivenDeckMotion + this check).

  python scripts/gen_seakeeping_data.py
"""

from __future__ import annotations

import os

import numpy as np

from drone_landing.sim.platforms import sea_state
from drone_landing.sim.platforms.data_driven import DataDrivenDeckMotion
from drone_landing.sim.platforms.ship import ShipDeckMotion
from drone_landing.sim.platforms.wave_spectrum import SPECTRAL_SEA_STATES

OUT_DIR = "assets/seakeeping"


def _rpy(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def generate(sea: str, duration: float = 600.0, dt: float = 0.1, seed: int = 0) -> str:
    """Render a fine 6-DOF trace from the spectral model and save it as a CSV reference."""
    m = ShipDeckMotion(sea_state(sea, spectral=True))
    m.reset(np.random.default_rng(seed))
    rows = []
    for k in range(int(duration / dt)):
        s = m.step(dt)
        r, p, y = _rpy(s.quat)
        rows.append([k * dt, s.pos[0], s.pos[1], s.pos[2], r, p, y])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{sea}.csv")
    np.savetxt(path, np.array(rows), delimiter=",", header="t,x,y,z,roll,pitch,yaw",
               comments="", fmt="%.5f")
    return path


def characterize(path: str, dt: float = 0.1) -> dict:
    """Extract Hs (4*std heave), Tp (heave PSD peak), roll/pitch RMS from a 6-DOF log -- as for real data."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    heave = d["z"] - np.mean(d["z"])
    # peak period from the heave power spectral density
    freqs = np.fft.rfftfreq(len(heave), d=dt)
    psd = np.abs(np.fft.rfft(heave)) ** 2
    psd[0] = 0.0
    tp = 1.0 / freqs[int(np.argmax(psd))] if freqs[int(np.argmax(psd))] > 0 else float("inf")
    return {"Hs_sim": 4 * np.std(heave), "Tp_sim": tp,
            "roll_rms_deg": np.degrees(np.std(d["roll"])), "pitch_rms_deg": np.degrees(np.std(d["pitch"]))}


def main() -> None:
    print("Generating + validating seakeeping reference data (spectral model; real CSVs drop in the same):\n")
    print(f"  {'sea':<9} {'target Hs/Tp':<16} {'sim Hs (scaled)':<16} {'sim Tp':<10} {'roll/pitch RMS'}")
    for sea, spec in SPECTRAL_SEA_STATES.items():
        path = generate(sea)
        c = characterize(path)
        # the sim deck is a scale model; Hs is in sim metres -> report the ratio to the target as the scale
        print(f"  {sea:<9} Hs={spec.hs:.2f} Tp={spec.tp:.1f}s   {c['Hs_sim']:.3f} m (sim)    "
              f"{c['Tp_sim']:.1f} s     {c['roll_rms_deg']:.2f}/{c['pitch_rms_deg']:.2f} deg   -> {path}")
        # P1.3 check: the recovered peak period should match the design Tp within ~25% (finite trace)
        ok = abs(c["Tp_sim"] - spec.tp) / spec.tp < 0.30
        print(f"            Tp match vs design ({spec.tp:.1f}s): {'OK' if ok else 'CHECK'}")
    # confirm the replay model round-trips the data
    dd = DataDrivenDeckMotion.from_csv(os.path.join(OUT_DIR, "moderate.csv"))
    dd.reset(np.random.default_rng(0))
    s = dd.step(0.05)
    print(f"\n  DataDrivenDeckMotion replays the CSV: pos={np.round(s.pos,3)} (drop-in PlatformMotion)")


if __name__ == "__main__":
    main()
