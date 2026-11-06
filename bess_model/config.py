"""
Inputs for the AI data center BESS sizing model.

Every number here is an editable assumption. Defaults are illustrative
round numbers for a 2026 liquid-cooled GPU campus, NOT vendor quotes.
Replace them with project-specific data (vendor quotes, utility tariffs,
interconnection study results) before using the output for decisions.
"""

from dataclasses import dataclass


@dataclass
class ClusterInputs:
    """What is being powered: the GPU cluster and the building around it."""

    num_gpus: int = 50_000          # total GPUs in the cluster
    gpus_per_rack: int = 72         # rack-scale systems put 72 GPUs in one rack
    rack_kw: float = 132.0          # power per rack at full load (kW)
    non_gpu_it_fraction: float = 0.10  # networking + storage, as a share of GPU rack load
    pue_avg: float = 1.20           # Power Usage Effectiveness: facility power / IT power
    pue_daily_swing: float = 0.08   # PUE rises on hot afternoons (cooling works harder)
    avg_utilization: float = 0.85   # average share of peak IT power actually drawn


@dataclass
class GridInputs:
    """What the utility can deliver and how fast the load may change."""

    interconnection_limit_mw: float = 105.0  # max power the grid connection can supply (MW)
    max_ramp_pct_per_min: float = 10.0       # utility ramp limit, % of peak load per minute


@dataclass
class TrainingSwingInputs:
    """Fast power swings caused by synchronized GPU training jobs."""

    low_load_fraction: float = 0.40   # IT power during communication / sync phases
    compute_phase_s: float = 6.0      # seconds at full power per training step
    sync_phase_s: float = 2.0         # seconds at low power per training step
    checkpoint_every_min: float = 10.0  # minutes between checkpoints
    checkpoint_load_fraction: float = 0.30  # IT power while writing a checkpoint
    checkpoint_duration_s: float = 45.0


@dataclass
class BessInputs:
    """Battery technical assumptions and which jobs the battery must do."""

    backup_hours: float = 4.0            # hours of backup for the critical load
    backup_load_fraction: float = 1.0    # share of facility load the battery must back up
    round_trip_efficiency: float = 0.88  # AC-to-AC round-trip efficiency
    usable_depth_of_discharge: float = 0.95
    end_of_life_capacity: float = 0.85   # capacity floor before augmentation
    max_c_rate: float = 2.0              # max power / energy ratio (LFP ~0.25-2C; supercaps far higher)
    use_backup: bool = True
    use_smoothing: bool = True
    use_grid_cap: bool = True


@dataclass
class EconomicInputs:
    """Costs and value streams. All currency in USD."""

    energy_cost_per_kwh: float = 280.0      # installed battery energy cost ($/kWh)
    power_cost_per_kw: float = 150.0        # inverters/PCS + interconnection BOP ($/kW)
    om_pct_of_capex: float = 0.02           # annual O&M as share of capex
    diesel_cost_per_kw: float = 800.0       # installed diesel genset cost ($/kW)
    replace_diesel: bool = True             # does the battery displace diesel backup?
    demand_charge_per_kw_month: float = 15.0
    cycles_per_year: float = 250.0          # arbitrage cycles per year
    backup_energy_tradable: float = 0.25    # share of backup energy allowed to cycle daily
    charge_price_per_mwh: float = 35.0
    discharge_price_per_mwh: float = 90.0
    speed_to_power_months: float = 18.0     # months of operation gained before grid upgrade
    compute_revenue_per_it_mw_year: float = 8_000_000.0  # gross GPU cloud revenue per IT MW
    compute_margin: float = 0.30            # share of compute revenue that is profit
    # Without a battery, swings are hidden by making GPUs burn filler power so the
    # load stays flat near peak. The battery avoids buying that wasted energy.
    training_hours_per_year: float = 7000.0
    avg_energy_price_per_mwh: float = 60.0
    project_years: int = 15
    discount_rate: float = 0.08


@dataclass
class ModelInputs:
    cluster: ClusterInputs
    grid: GridInputs
    swing: TrainingSwingInputs
    bess: BessInputs
    econ: EconomicInputs

    @classmethod
    def default(cls) -> "ModelInputs":
        return cls(ClusterInputs(), GridInputs(), TrainingSwingInputs(),
                   BessInputs(), EconomicInputs())
