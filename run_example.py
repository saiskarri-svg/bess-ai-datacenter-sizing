"""
Run three scenarios for a 50,000-GPU AI training campus and save charts to docs/.

    python run_example.py
"""

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bess_model import (ModelInputs, evaluate, recommend_system, size_grid_cap,
                        size_smoothing, summary)

DOCS = Path(__file__).parent / "docs"
DOCS.mkdir(exist_ok=True)


def scenarios() -> dict:
    base = ModelInputs.default()
    keep_diesel = replace(base, bess=replace(base.bess, use_backup=False))
    cheaper = replace(base, econ=replace(base.econ, energy_cost_per_kwh=200.0))
    return {
        "A. Replace diesel (4 h backup) + bridge + smoothing": base,
        "B. Keep diesel; battery for bridge + smoothing": keep_diesel,
        "C. Scenario A at $200/kWh battery cost": cheaper,
    }


def fmt_m(x: float) -> str:
    return f"${x / 1e6:,.1f}M"


def print_report(name: str, m: ModelInputs) -> dict:
    s = summary(m)
    system = recommend_system(m)
    econ = evaluate(m, system)
    print(f"\n=== {name} ===")
    print(f"Cluster: {m.cluster.num_gpus:,} GPUs in {s['racks']} racks | "
          f"peak IT {s['peak_it_mw']:.1f} MW | peak facility {s['peak_facility_mw']:.1f} MW")
    for c in system.components:
        print(f"  {c.name:<10} {c.power_mw:7.1f} MW  {c.energy_mwh:7.1f} MWh   {c.note}")
    print(f"  Recommended system: {system.power_mw:.1f} MW / {system.energy_mwh:.1f} MWh "
          f"({system.duration_h:.2f} h), incl. degradation reserve")
    print(f"  Capex {fmt_m(econ.capex)} | avoided diesel {fmt_m(econ.avoided_diesel)} | "
          f"speed-to-power value {fmt_m(econ.speed_to_power_value)}")
    print(f"  Annual: demand savings {fmt_m(econ.annual_demand_savings)}, "
          f"arbitrage {fmt_m(econ.annual_arbitrage)}, avoided filler power {fmt_m(econ.annual_avoided_filler)}, O&M {fmt_m(econ.annual_om)}")
    payback = "never" if econ.simple_payback_years == float("inf") else f"{econ.simple_payback_years:.1f} yrs"
    print(f"  NPV {fmt_m(econ.npv)} | LCOS ${econ.lcos_per_mwh:,.0f}/MWh | payback {payback}")
    return {"name": name, "system": system, "econ": econ}


def save_charts(m: ModelInputs, results: list) -> None:
    _, prof = size_smoothing(m)
    window = prof[prof["second"] < 120]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window["second"], window["facility_mw"], label="Cluster load (no battery)", lw=1)
    ax.plot(window["second"], window["grid_mw"], label="Grid sees (with battery)", lw=2)
    ax.set(xlabel="Seconds", ylabel="MW", title="GPU training swings: battery absorbs what the grid cannot")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(DOCS / "smoothing.png", dpi=150); plt.close(fig)

    _, day, _ = size_grid_cap(m)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(day["hour"], day["facility_mw"], label="Facility load", lw=2)
    ax.axhline(m.grid.interconnection_limit_mw, color="red", ls="--", label="Interconnection limit")
    ax.fill_between(day["hour"], m.grid.interconnection_limit_mw, day["facility_mw"],
                    where=day["facility_mw"] > m.grid.interconnection_limit_mw,
                    alpha=0.3, label="Battery discharges")
    ax.set(xlabel="Hour of day", ylabel="MW", title="Bridge power: running above the grid limit on hot afternoons")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(DOCS / "grid_cap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [r["name"].split(".")[0] for r in results]
    ax.bar(labels, [r["econ"].npv / 1e6 for r in results],
           color=["#c0392b" if r["econ"].npv < 0 else "#2e86c1" for r in results])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="NPV ($M)", title="Project NPV by scenario (15 years, 8% discount rate)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(DOCS / "npv_by_scenario.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    runs = scenarios()
    results = [print_report(name, m) for name, m in runs.items()]
    save_charts(next(iter(runs.values())), results)
    print(f"\nCharts saved to {DOCS}/")
