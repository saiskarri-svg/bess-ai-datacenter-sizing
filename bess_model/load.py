"""
Load profiles for an AI training campus.

Two time scales matter for a battery:
  * daily (15-minute steps): cooling load follows the weather, which decides
    how far the facility goes above its grid connection limit.
  * seconds (1-second steps): synchronized training jobs swing the whole
    cluster between full power and low power every few seconds, which the
    grid may not tolerate.
"""

import numpy as np
import pandas as pd

from .config import ClusterInputs, TrainingSwingInputs


def peak_it_mw(c: ClusterInputs) -> float:
    """Peak IT load in MW: GPU racks plus networking and storage."""
    racks = np.ceil(c.num_gpus / c.gpus_per_rack)
    gpu_mw = racks * c.rack_kw / 1000
    return gpu_mw * (1 + c.non_gpu_it_fraction)


def peak_facility_mw(c: ClusterInputs) -> float:
    """Peak facility load: peak IT at the hottest-hour PUE."""
    return peak_it_mw(c) * (c.pue_avg + c.pue_daily_swing)


def daily_profile(c: ClusterInputs, step_min: int = 15) -> pd.DataFrame:
    """One design day of facility load.

    IT load is flat at average utilization (training clusters run near-constant).
    PUE follows a sine wave peaking at 3 pm, so cooling adds the most load
    in the afternoon.
    """
    hours = np.arange(0, 24, step_min / 60)
    pue = c.pue_avg + c.pue_daily_swing * np.sin((hours - 9) / 24 * 2 * np.pi)
    it = np.full_like(hours, peak_it_mw(c) * c.avg_utilization)
    return pd.DataFrame({"hour": hours, "it_mw": it, "pue": pue,
                         "facility_mw": it * pue})


def training_swing_profile(c: ClusterInputs, s: TrainingSwingInputs,
                           minutes: float = 30) -> pd.DataFrame:
    """Second-by-second facility load during a large synchronized training run."""
    t = np.arange(0, int(minutes * 60))
    step = s.compute_phase_s + s.sync_phase_s
    in_compute = (t % step) < s.compute_phase_s
    it_frac = np.where(in_compute, 1.0, s.low_load_fraction)

    ckpt_period = s.checkpoint_every_min * 60
    in_ckpt = (t % ckpt_period) >= (ckpt_period - s.checkpoint_duration_s)
    it_frac = np.where(in_ckpt, s.checkpoint_load_fraction, it_frac)

    it_mw = peak_it_mw(c) * it_frac
    # Cooling responds slowly (thermal mass), so it tracks the average IT load, not each swing.
    cooling_mw = peak_it_mw(c) * it_frac.mean() * (c.pue_avg - 1)
    return pd.DataFrame({"second": t, "it_mw": it_mw,
                         "facility_mw": it_mw + cooling_mw})
