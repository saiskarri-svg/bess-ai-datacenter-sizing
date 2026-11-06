"""
Economics of the recommended battery: capex, value streams, NPV and LCOS.

Value streams counted:
  * Avoided diesel capex  - if the battery replaces generators for backup.
  * Demand charge savings - grid peak is held at the interconnection limit.
  * Energy arbitrage      - charge cheap, discharge expensive.
  * Speed to power        - extra IT capacity the campus can run before a grid upgrade.
  * Avoided filler power  - without a battery, GPUs burn extra power to keep the load
                            flat for the grid; the battery smooths it instead.
"""

from dataclasses import dataclass

import numpy as np

from .config import ModelInputs
from .sizing import SystemSize, size_backup, size_grid_cap, size_smoothing


@dataclass
class EconomicResult:
    capex: float
    avoided_diesel: float
    annual_demand_savings: float
    annual_arbitrage: float
    annual_avoided_filler: float
    speed_to_power_value: float
    annual_om: float
    npv: float
    lcos_per_mwh: float
    simple_payback_years: float
    cash_flows: list


def _pv(amount: float, rate: float, years: int) -> float:
    """Present value of a constant annual amount for `years` years."""
    return sum(amount / (1 + rate) ** y for y in range(1, years + 1))


def evaluate(m: ModelInputs, system: SystemSize) -> EconomicResult:
    e, b = m.econ, m.bess
    capex = system.energy_mwh * 1000 * e.energy_cost_per_kwh + system.power_mw * 1000 * e.power_cost_per_kw
    om = capex * e.om_pct_of_capex

    avoided_diesel = 0.0
    if b.use_backup and e.replace_diesel:
        avoided_diesel = size_backup(m).power_mw * 1000 * e.diesel_cost_per_kw

    cap_mw = 0.0
    if b.use_grid_cap:
        cap_result, _, feasible = size_grid_cap(m)
        cap_mw = cap_result.power_mw if feasible else 0.0
    demand_savings = cap_mw * 1000 * e.demand_charge_per_kw_month * 12

    # Arbitrage may only use capacity no other job has claimed. Daily jobs (smoothing,
    # grid cap) own their share; backup energy must stay charged for outages, except
    # the share the owner is willing to cycle (backup_energy_tradable).
    eff = np.sqrt(b.round_trip_efficiency)
    by_name = {c.name: c.energy_mwh for c in system.components}
    daily_e = by_name.get("Smoothing", 0.0) + by_name.get("Grid cap", 0.0)
    reserved_backup = by_name.get("Backup", 0.0) * (1 - e.backup_energy_tradable)
    free_mwh = max(0.0, system.energy_mwh * b.end_of_life_capacity - daily_e - reserved_backup)
    usable_free = free_mwh * b.usable_depth_of_discharge
    arb_discharged = usable_free * e.cycles_per_year * eff
    arb_charged = usable_free * e.cycles_per_year / eff
    arbitrage = arb_discharged * e.discharge_price_per_mwh - arb_charged * e.charge_price_per_mwh

    # Bridge-power energy delivered every day (its value is counted in speed to power).
    cap_daily_mwh = by_name.get("Grid cap", 0.0) * b.usable_depth_of_discharge * eff ** 2
    discharged = arb_discharged + cap_daily_mwh * 365
    charged = arb_charged + cap_daily_mwh * 365 / b.round_trip_efficiency

    avoided_filler = 0.0
    if b.use_smoothing:
        _, prof = size_smoothing(m)
        waste_mw = (prof["facility_mw"].max() - prof["facility_mw"]).mean()
        avoided_filler = waste_mw * e.training_hours_per_year * e.avg_energy_price_per_mwh

    # Extra IT MW the campus can run thanks to the battery, before the grid is upgraded.
    extra_it_mw = cap_mw / m.cluster.pue_avg
    speed_value = (extra_it_mw * e.compute_revenue_per_it_mw_year * e.compute_margin
                   * e.speed_to_power_months / 12)

    years = e.project_years
    flows = [-capex + avoided_diesel]
    for y in range(1, years + 1):
        flow = demand_savings + arbitrage + avoided_filler - om
        if y <= int(np.ceil(e.speed_to_power_months / 12)):
            months_this_year = min(12, e.speed_to_power_months - 12 * (y - 1))
            flow += speed_value * months_this_year / e.speed_to_power_months if e.speed_to_power_months else 0
        flows.append(flow)
    npv = sum(f / (1 + e.discount_rate) ** i for i, f in enumerate(flows))

    pv_costs = capex + _pv(om + charged * e.charge_price_per_mwh, e.discount_rate, years)
    pv_mwh = _pv(discharged, e.discount_rate, years)
    lcos = pv_costs / pv_mwh if pv_mwh else float("nan")

    cumulative, payback = flows[0], float("inf")
    for y, f in enumerate(flows[1:], start=1):
        if cumulative < 0 <= cumulative + f:
            payback = y - 1 + (-cumulative / f)
        cumulative += f
    if flows[0] >= 0:
        payback = 0.0

    return EconomicResult(capex, avoided_diesel, demand_savings, arbitrage, avoided_filler, speed_value,
                          om, npv, lcos, payback, flows)
