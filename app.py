"""
Interactive BESS sizing tool for AI data centers.

    streamlit run app.py
"""

import plotly.graph_objects as go
import streamlit as st

from bess_model import (BessInputs, ClusterInputs, EconomicInputs, GridInputs, ModelInputs,
                        TrainingSwingInputs, evaluate, recommend_system, size_grid_cap,
                        size_smoothing, summary)

st.set_page_config(page_title="AI Data Center BESS Sizing", layout="wide")
st.title("Battery Storage Sizing for AI Data Centers")
st.caption("Size a battery for backup, GPU load smoothing and bridge power, then test the economics. "
           "Defaults are illustrative assumptions, not vendor quotes.")

d = ModelInputs.default()
sb = st.sidebar
sb.header("GPU cluster")
num_gpus = sb.number_input("GPUs", 1_000, 1_000_000, d.cluster.num_gpus, step=5_000)
rack_kw = sb.number_input("kW per rack", 10.0, 300.0, d.cluster.rack_kw)
gpus_per_rack = sb.number_input("GPUs per rack", 4, 144, d.cluster.gpus_per_rack)
pue = sb.slider("Average PUE", 1.05, 1.6, d.cluster.pue_avg, 0.01)
pue_swing = sb.slider("Daily PUE swing (hot afternoons)", 0.0, 0.2, d.cluster.pue_daily_swing, 0.01)
util = sb.slider("Average IT utilization", 0.5, 1.0, d.cluster.avg_utilization, 0.01)

sb.header("Grid")
limit = sb.number_input("Interconnection limit (MW)", 1.0, 5_000.0, d.grid.interconnection_limit_mw)
ramp = sb.slider("Utility ramp limit (% of peak per minute)", 1.0, 50.0, d.grid.max_ramp_pct_per_min)

sb.header("Battery jobs")
use_backup = sb.checkbox("Backup (replace / supplement diesel)", d.bess.use_backup)
backup_h = sb.slider("Backup hours", 0.25, 8.0, d.bess.backup_hours, 0.25, disabled=not use_backup)
use_smooth = sb.checkbox("Smooth GPU training swings", d.bess.use_smoothing)
low_frac = sb.slider("Load during sync phase (% of peak)", 0.1, 0.9, d.swing.low_load_fraction, 0.05,
                     disabled=not use_smooth)
use_cap = sb.checkbox("Bridge power above grid limit", d.bess.use_grid_cap)

sb.header("Economics")
e_cost = sb.number_input("Battery energy cost ($/kWh)", 50.0, 1_000.0, d.econ.energy_cost_per_kwh)
p_cost = sb.number_input("Power equipment cost ($/kW)", 0.0, 1_000.0, d.econ.power_cost_per_kw)
replace_diesel = sb.checkbox("Battery replaces diesel gensets", d.econ.replace_diesel, disabled=not use_backup)
diesel = sb.number_input("Diesel genset cost ($/kW)", 0.0, 3_000.0, d.econ.diesel_cost_per_kw)
s2p = sb.slider("Months before grid upgrade (speed to power)", 0, 60, int(d.econ.speed_to_power_months))
rev = sb.number_input("Compute revenue per IT MW-year ($M)", 0.0, 50.0,
                      d.econ.compute_revenue_per_it_mw_year / 1e6) * 1e6

m = ModelInputs(
    ClusterInputs(num_gpus=int(num_gpus), gpus_per_rack=int(gpus_per_rack), rack_kw=rack_kw,
                  pue_avg=pue, pue_daily_swing=pue_swing, avg_utilization=util),
    GridInputs(interconnection_limit_mw=limit, max_ramp_pct_per_min=ramp),
    TrainingSwingInputs(low_load_fraction=low_frac),
    BessInputs(backup_hours=backup_h, use_backup=use_backup, use_smoothing=use_smooth,
               use_grid_cap=use_cap),
    EconomicInputs(energy_cost_per_kwh=e_cost, power_cost_per_kw=p_cost, replace_diesel=replace_diesel,
                   diesel_cost_per_kw=diesel, speed_to_power_months=s2p,
                   compute_revenue_per_it_mw_year=rev),
)

if not (use_backup or use_smooth or use_cap):
    st.warning("Select at least one battery job in the sidebar.")
    st.stop()

s = summary(m)
system = recommend_system(m)
econ = evaluate(m, system)

def card(col, label, value, sub):
    col.metric(label, value)
    col.caption(sub)


npv = econ.npv / 1e6
payback = "Never pays back" if econ.simple_payback_years == float("inf") \
    else f"{econ.simple_payback_years:.1f} year payback"
c1, c2, c3, c4 = st.columns(4)
card(c1, "Peak facility load", f"{s['peak_facility_mw']:.0f} MW", f"{s['racks']:,} racks")
card(c2, "Recommended battery", f"{system.energy_mwh:,.0f} MWh",
     f"{system.power_mw:.0f} MW, {system.duration_h:.1f} h duration")
card(c3, "Net capex", f"${(econ.capex - econ.avoided_diesel) / 1e6:,.0f}M",
     f"after ${econ.avoided_diesel / 1e6:,.0f}M avoided diesel")
card(c4, "NPV (15 yr)", f"{'-' if npv < 0 else ''}${abs(npv):,.0f}M", payback)

if not system.grid_cap_feasible:
    st.error("Bridge power is not feasible: average load is above the grid limit, so the battery can "
             "never fully recharge. A battery shifts energy; it cannot create it. Add on-site generation, "
             "phase the cluster, or raise the limit.")

st.subheader("Sizing by job")
st.dataframe([{"Job": c.name, "Power (MW)": round(c.power_mw, 1), "Energy (MWh)": round(c.energy_mwh, 1),
               "Duration (h)": round(c.duration_h, 2), "Basis": c.note} for c in system.components],
             hide_index=True, use_container_width=True)
st.caption("Backup only runs during outages, so it does not stack with daily jobs. "
           f"System energy includes a reserve for degradation to {m.bess.end_of_life_capacity:.0%} capacity.")

left, right = st.columns(2)
if use_smooth:
    _, prof = size_smoothing(m)
    w = prof[prof["second"] < 120]
    fig = go.Figure([go.Scatter(x=w["second"], y=w["facility_mw"], name="Cluster load", line=dict(width=1)),
                     go.Scatter(x=w["second"], y=w["grid_mw"], name="Grid sees", line=dict(width=3))])
    fig.update_layout(title="GPU training swings (first 2 minutes)", xaxis_title="Seconds",
                      yaxis_title="MW", height=360, margin=dict(t=50, b=40))
    left.plotly_chart(fig, use_container_width=True)
if use_cap:
    _, day, _ = size_grid_cap(m)
    fig = go.Figure([go.Scatter(x=day["hour"], y=day["facility_mw"], name="Facility load", line=dict(width=3)),
                     go.Scatter(x=day["hour"], y=[limit] * len(day), name="Grid limit",
                                line=dict(dash="dash", color="red"))])
    fig.update_layout(title="Design day vs. interconnection limit", xaxis_title="Hour",
                      yaxis_title="MW", height=360, margin=dict(t=50, b=40))
    right.plotly_chart(fig, use_container_width=True)

st.subheader("Value streams")
st.dataframe([
    {"Item": "Battery capex", "Value ($M)": -econ.capex / 1e6, "Timing": "Year 0"},
    {"Item": "Avoided diesel gensets", "Value ($M)": econ.avoided_diesel / 1e6, "Timing": "Year 0"},
    {"Item": "Speed to power (extra compute margin)", "Value ($M)": econ.speed_to_power_value / 1e6,
     "Timing": f"First {s2p} months"},
    {"Item": "Avoided filler power", "Value ($M)": econ.annual_avoided_filler / 1e6, "Timing": "Per year"},
    {"Item": "Demand charge savings", "Value ($M)": econ.annual_demand_savings / 1e6, "Timing": "Per year"},
    {"Item": "Energy arbitrage", "Value ($M)": econ.annual_arbitrage / 1e6, "Timing": "Per year"},
    {"Item": "O&M", "Value ($M)": -econ.annual_om / 1e6, "Timing": "Per year"},
], hide_index=True, use_container_width=True, column_config={"Value ($M)": st.column_config.NumberColumn(format="%.1f")})
