"""
Battery sizing for three jobs a BESS can do at an AI data center.

1. Backup        - replace or supplement diesel generators for hours of outage cover.
2. Smoothing     - absorb second-scale GPU training swings so the grid sees a gentle ramp.
3. Grid cap      - run the campus above its interconnection limit ("bridge power"),
                   recharging when load dips below the limit.

Each job gives a power need (MW: how fast) and an energy need (MWh: how long).
The recommended system covers all enabled jobs, plus a degradation reserve.
"""

from dataclasses import dataclass

import numpy as np

from .config import ModelInputs
from .load import daily_profile, peak_facility_mw, peak_it_mw, training_swing_profile


@dataclass
class SizingResult:
    name: str
    power_mw: float
    energy_mwh: float
    note: str = ""

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw if self.power_mw else 0.0


def size_backup(m: ModelInputs) -> SizingResult:
    b = m.bess
    power = peak_facility_mw(m.cluster) * b.backup_load_fraction
    one_way_eff = np.sqrt(b.round_trip_efficiency)  # discharge side of the round trip
    energy = power * b.backup_hours / one_way_eff / b.usable_depth_of_discharge
    return SizingResult("Backup", power, energy,
                        f"{b.backup_hours:g} h at {b.backup_load_fraction:.0%} of peak facility load")


def ramp_limited_grid(load_mw: np.ndarray, max_ramp_mw_per_s: float,
                      window_s: int = 60) -> np.ndarray:
    """What the grid supplies under a simple battery controller.

    The grid follows the rolling average of the load (so the battery charges and
    discharges around zero instead of drifting), and that target may change by at
    most max_ramp each second.
    """
    load_mw = np.asarray(load_mw, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(load_mw)])
    idx = np.arange(1, len(load_mw) + 1)
    start = np.maximum(0, idx - window_s)
    target = (csum[idx] - csum[start]) / (idx - start)   # trailing rolling mean
    target[:window_s] = load_mw[:window_s].mean()          # warm start: no startup transient

    grid = np.empty_like(load_mw)
    grid[0] = target[0]
    for i in range(1, len(load_mw)):
        delta = np.clip(target[i] - grid[i - 1], -max_ramp_mw_per_s, max_ramp_mw_per_s)
        grid[i] = grid[i - 1] + delta
    return grid


def size_smoothing(m: ModelInputs):
    """Returns (SizingResult, profile DataFrame with grid and battery columns)."""
    prof = training_swing_profile(m.cluster, m.swing)
    peak = peak_facility_mw(m.cluster)
    max_ramp = peak * m.grid.max_ramp_pct_per_min / 100 / 60  # MW per second
    prof["grid_mw"] = ramp_limited_grid(prof["facility_mw"].to_numpy(), max_ramp)
    prof["bess_mw"] = prof["facility_mw"] - prof["grid_mw"]  # + discharging, - charging

    power = prof["bess_mw"].abs().max()
    soc_mwh = prof["bess_mw"].cumsum() / 3600           # 1-second steps -> MWh
    energy = (soc_mwh.max() - soc_mwh.min()) / m.bess.usable_depth_of_discharge
    raw_ramp = prof["facility_mw"].diff().abs().max()
    note = (f"Raw swing up to {raw_ramp:.0f} MW in one second; "
            f"grid limited to {max_ramp:.2f} MW/s")
    return SizingResult("Smoothing", power, energy, note), prof


def size_grid_cap(m: ModelInputs):
    """Returns (SizingResult, daily profile with battery columns, feasible flag)."""
    day = daily_profile(m.cluster)
    cap = m.grid.interconnection_limit_mw
    dt_h = day["hour"].diff().iloc[1]
    above = (day["facility_mw"] - cap).clip(lower=0)
    below = (cap - day["facility_mw"]).clip(lower=0)

    day["grid_mw"] = day["facility_mw"].clip(upper=cap)
    day["bess_mw"] = day["facility_mw"] - day["grid_mw"]

    power = above.max()
    energy_needed = above.sum() * dt_h / np.sqrt(m.bess.round_trip_efficiency)
    recharge_available = below.sum() * dt_h * np.sqrt(m.bess.round_trip_efficiency)
    feasible = recharge_available >= energy_needed
    energy = energy_needed / m.bess.usable_depth_of_discharge

    if power == 0:
        note = "Facility stays under the interconnection limit; no bridge power needed"
    elif feasible:
        note = (f"Covers {power:.1f} MW above the {cap:g} MW limit; "
                f"recharges in off-peak hours")
    else:
        note = ("NOT FEASIBLE daily: average load exceeds the limit, so the battery "
                "cannot recharge. Needs on-site generation or a smaller cluster.")
    return SizingResult("Grid cap", power, energy, note), day, feasible


@dataclass
class SystemSize:
    components: list
    power_mw: float
    energy_mwh: float          # beginning-of-life nameplate energy
    grid_cap_feasible: bool

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw if self.power_mw else 0.0


def recommend_system(m: ModelInputs) -> SystemSize:
    b = m.bess
    comps, feasible = [], True
    backup = smooth = cap = None
    if b.use_backup:
        backup = size_backup(m); comps.append(backup)
    if b.use_smoothing:
        smooth, _ = size_smoothing(m); comps.append(smooth)
    if b.use_grid_cap:
        cap, _, feasible = size_grid_cap(m); comps.append(cap)

    # Smoothing and grid-cap run at the same time in normal operation, so they stack.
    # Backup is only used during an outage, when the other jobs stop, so it does not stack.
    daily_p = (smooth.power_mw if smooth else 0) + (cap.power_mw if cap else 0)
    daily_e = (smooth.energy_mwh if smooth else 0) + (cap.energy_mwh if cap else 0)
    power = max(backup.power_mw if backup else 0, daily_p)
    energy = max(backup.energy_mwh if backup else 0, daily_e) / b.end_of_life_capacity
    # Cells can only deliver so much power per MWh; a high-power, short job is power-limited.
    energy = max(energy, power / b.max_c_rate)
    return SystemSize(comps, power, energy, feasible)


def summary(m: ModelInputs) -> dict:
    return {"peak_it_mw": peak_it_mw(m.cluster),
            "peak_facility_mw": peak_facility_mw(m.cluster),
            "racks": int(np.ceil(m.cluster.num_gpus / m.cluster.gpus_per_rack))}
