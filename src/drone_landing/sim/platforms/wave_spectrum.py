"""Maritime fidelity (B1): real wave spectra (JONSWAP / Pierson-Moskowitz) + response-amplitude
operators (RAOs) driving the ship deck.

The validated :class:`ShipDeckMotion` synthesizes each DOF as ``sum_i a_i sin(w_i t + phi_i)`` — which is
*exactly* the random-phase synthesis of a motion spectrum. This module is the principled **generator** of
those ``(amplitude, period)`` components: a named sea state -> a wave spectrum ``S(w)`` -> per-DOF ship
**response** spectra via RAOs -> discretized components. The proven motion engine is reused unchanged, so
the spectral model is a drop-in richer parameterization, not a rewrite of the simulation core.

Definitions used (standard naval-architecture forms):

* **Pierson-Moskowitz** (fully developed sea), in Hs/Tp form:
  ``S_pm(w) = (5/16) Hs^2 (w_p^4 / w^5) exp(-5/4 (w_p/w)^4)``,  ``w_p = 2*pi/Tp``.
* **JONSWAP** (fetch-limited, peakier): ``S_j(w) = S_pm(w) * (1 - 0.287 ln gamma) * gamma^r``,
  ``r = exp(-(w - w_p)^2 / (2 sigma^2 w_p^2))``, ``sigma = 0.07 (w<=w_p) else 0.09``, ``gamma ~ 3.3``.
* **Spectral moments** ``m_n = integral w^n S(w) dw``; significant height ``Hs = 4 sqrt(m0)``;
  random-phase synthesis amplitude per bin ``a_i = sqrt(2 S(w_i) dw_i)``.
* **RAO**: the ship's response spectrum in a DOF is ``|RAO(w)|^2 S(w)``; the component amplitude is then
  ``a_i = sqrt(2 |RAO(w_i)|^2 S(w_i) dw_i)`` (metres for heave/sway, radians for roll/pitch).

A single ``model_scale`` maps real-world significant motion to the tabletop sim deck (the named presets are
calibrated so this reproduces the magnitude of the validated sum-of-sinusoids, keeping the ship-landing
result intact — only the spectral *content* gets richer/realistic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.sim.platforms.ship import ShipMotionConfig, WaveComponent

G = 9.81


# --------------------------------------------------------------------- wave spectra

def pierson_moskowitz(omega: np.ndarray, hs: float, tp: float) -> np.ndarray:
    """Pierson-Moskowitz spectral density S(omega) [m^2 s] for significant height ``hs`` and peak
    period ``tp``."""
    omega = np.asarray(omega, dtype=float)
    wp = 2 * np.pi / tp
    with np.errstate(divide="ignore", over="ignore"):
        s = (5.0 / 16.0) * hs**2 * wp**4 / omega**5 * np.exp(-1.25 * (wp / omega) ** 4)
    return np.where(omega > 0, s, 0.0)


def jonswap(omega: np.ndarray, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP spectral density S(omega) [m^2 s]; ``gamma=1`` recovers Pierson-Moskowitz."""
    omega = np.asarray(omega, dtype=float)
    wp = 2 * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp) ** 2) / (2 * sigma**2 * wp**2))
    norm = 1.0 - 0.287 * np.log(gamma)
    return pierson_moskowitz(omega, hs, tp) * norm * gamma**r


def spectral_moment(omega: np.ndarray, s: np.ndarray, n: int = 0) -> float:
    """n-th spectral moment m_n = integral omega^n S(omega) d(omega) (trapezoidal)."""
    # numpy>=2 renamed trapz -> trapezoid AND removed the old name; guard both.
    # Note: getattr(np, "trapezoid", np.trapz) is not safe -- Python evaluates the
    # default eagerly, and np.trapz no longer exists on numpy>=2, so it raises before
    # the getattr fallback can fire.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(omega**n * s, omega))


def significant_height(omega: np.ndarray, s: np.ndarray) -> float:
    """Significant wave height Hs = 4 sqrt(m0) recovered from a (possibly RAO-shaped) spectrum."""
    return 4.0 * np.sqrt(max(spectral_moment(omega, s, 0), 0.0))


# --------------------------------------------------------------------- RAOs (dimensionless / dimensional)

@dataclass(frozen=True)
class RAOConfig:
    """Simplified, physically-shaped response-amplitude operators (per unit wave amplitude).

    Heave/sway follow long waves (RAO -> 1 at low frequency) and roll off at short waves; pitch and roll
    are second-order resonances about their natural periods (roll lightly damped -> peaky, the real cause
    of bad roll). Gains are per-metre-of-wave-amplitude: ``*_gain`` in rad/m for the angular DOFs.
    """
    heave_cut: float = 1.2     # rad/s heave low-pass corner
    sway_gain: float = 0.6     # sway is a fraction of heave response
    sway_cut: float = 1.0
    pitch_tn: float = 6.0      # s   pitch natural period
    pitch_zeta: float = 0.5
    pitch_gain: float = 0.03   # rad per m wave amplitude at resonance
    roll_tn: float = 7.5       # s   roll natural period (near the seaway -> resonance)
    roll_zeta: float = 0.12    # lightly damped -> sharp roll peak
    roll_gain: float = 0.10    # rad per m wave amplitude at resonance


def _lowpass(omega: np.ndarray, corner: float, order: int = 4) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + (omega / corner) ** (2 * order))


def _resonance(omega: np.ndarray, tn: float, zeta: float) -> np.ndarray:
    """Magnitude of a 2nd-order band-pass resonance peaking at natural frequency wn = 2pi/tn."""
    wn = 2 * np.pi / tn
    rr = omega / wn
    return rr**2 / np.sqrt((1 - rr**2) ** 2 + (2 * zeta * rr) ** 2)


def rao(dof: str, omega: np.ndarray, cfg: RAOConfig) -> np.ndarray:
    if dof == "heave":
        return _lowpass(omega, cfg.heave_cut)
    if dof == "sway":
        return cfg.sway_gain * _lowpass(omega, cfg.sway_cut)
    if dof == "pitch":
        peak = _resonance(2 * np.pi / cfg.pitch_tn * np.ones(1), cfg.pitch_tn, cfg.pitch_zeta)[0]
        return cfg.pitch_gain * _resonance(omega, cfg.pitch_tn, cfg.pitch_zeta) / peak
    if dof == "roll":
        peak = _resonance(2 * np.pi / cfg.roll_tn * np.ones(1), cfg.roll_tn, cfg.roll_zeta)[0]
        return cfg.roll_gain * _resonance(omega, cfg.roll_tn, cfg.roll_zeta) / peak
    raise ValueError(f"unknown DOF '{dof}'")


# --------------------------------------------------------------------- spectrum -> components

def discretize_components(dof: str, hs: float, tp: float, *, gamma: float, rao_cfg: RAOConfig,
                          target_rms: float, n_components: int = 24,
                          omega_range: tuple[float, float] = (0.2, 3.0)) -> tuple[WaveComponent, ...]:
    """Turn a (DOF response) spectrum into ``n_components`` random-phase sinusoid components.

    The RAO-shaped spectrum ``|RAO(w)|^2 S(w)`` sets the **spectral shape** (which frequencies dominate —
    the physically-correct part); the overall magnitude is then **normalized so the DOF's RMS equals
    ``target_rms``** (taken from the validated sum-of-sinusoids baseline, so the ship-landing result is
    preserved). Equal-frequency bins; component amplitude ``a_i = sqrt(2 |RAO|^2 S dw)`` before scaling.
    """
    w = np.linspace(omega_range[0], omega_range[1], n_components)
    dw = (omega_range[1] - omega_range[0]) / (n_components - 1)
    s = jonswap(w, hs, tp, gamma)
    h = rao(dof, w, rao_cfg)
    amps = np.sqrt(2.0 * (h**2) * s * dw)
    raw_rms = np.sqrt(np.sum(amps**2) / 2.0)             # RMS of sum of sinusoids = sqrt(sum a^2 / 2)
    if raw_rms > 1e-12:
        amps = amps * (target_rms / raw_rms)             # calibrate magnitude to the validated baseline
    return tuple(WaveComponent(float(a), float(2 * np.pi / wi)) for a, wi in zip(amps, w) if a > 1e-5)


@dataclass(frozen=True)
class SeaSpectrum:
    """A named sea state: real-world wave params (``hs``/``tp``/``gamma``, literature-matched) + the
    per-DOF target RMS the sim deck is calibrated to (matching the validated sum-of-sinusoids model, so
    spectral *shape* is physical while magnitude preserves the ship-landing result)."""
    name: str
    hs: float            # m   significant wave height (real)
    tp: float            # s   peak period (real)
    gamma: float         # JONSWAP peak factor
    heave_rms: float     # m     target sim heave RMS
    roll_rms: float      # rad   target sim roll RMS
    pitch_rms: float     # rad   target sim pitch RMS
    sway_rms: float      # m     target sim sway RMS
    deck_z: float = 0.30
    forward_speed: float = 0.4
    heading: float = 0.0
    rao: RAOConfig = RAOConfig()


# Named spectral sea states. Hs/Tp follow WMO sea-state bands (SS3 calm, SS4 moderate, SS6 rough); per-DOF
# target RMS calibrated to the validated sum-of-sinusoids model so the ship-landing result is preserved,
# while the spectral content (frequency distribution + RAO resonances) becomes physically correct.
SPECTRAL_SEA_STATES = {
    "calm": SeaSpectrum("calm", hs=0.6, tp=5.5, gamma=2.0,
                        heave_rms=0.014, roll_rms=np.deg2rad(0.71), pitch_rms=np.deg2rad(0.50),
                        sway_rms=0.014),
    "moderate": SeaSpectrum("moderate", hs=1.88, tp=8.8, gamma=3.3,
                            heave_rms=0.092, roll_rms=np.deg2rad(4.48), pitch_rms=np.deg2rad(3.00),
                            sway_rms=0.071),
    "rough": SeaSpectrum("rough", hs=5.0, tp=12.4, gamma=3.3,
                         heave_rms=0.171, roll_rms=np.deg2rad(8.98), pitch_rms=np.deg2rad(5.21),
                         sway_rms=0.127),
}


def spectral_sea_state(name: str = "moderate") -> ShipMotionConfig:
    """Build a :class:`ShipMotionConfig` whose per-DOF components are drawn from a JONSWAP spectrum
    shaped by RAOs — a spectrally-correct, literature-matched replacement for the hand-tuned presets."""
    name = name.lower()
    if name not in SPECTRAL_SEA_STATES:
        raise ValueError(f"unknown spectral sea state '{name}' (use {list(SPECTRAL_SEA_STATES)})")
    spec = SPECTRAL_SEA_STATES[name]
    kw = dict(gamma=spec.gamma, rao_cfg=spec.rao)
    return ShipMotionConfig(
        deck_z=spec.deck_z, forward_speed=spec.forward_speed, heading=spec.heading,
        heave=discretize_components("heave", spec.hs, spec.tp, target_rms=spec.heave_rms, **kw),
        roll=discretize_components("roll", spec.hs, spec.tp, target_rms=spec.roll_rms, **kw),
        pitch=discretize_components("pitch", spec.hs, spec.tp, target_rms=spec.pitch_rms, **kw),
        sway=discretize_components("sway", spec.hs, spec.tp, target_rms=spec.sway_rms, **kw),
    )
