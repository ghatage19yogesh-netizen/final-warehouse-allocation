"""Allocation Planner — Streamlit demo.

Three screens from the wireframe: Set up & run -> Plan -> Warnings.
Runs on synthetic data with an open-source solver (HiGHS via SciPy).
"""

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import demo_data as dd
from engine import pre_run_checks, run_allocation

st.set_page_config(page_title="Allocation Planner (demo)", layout="wide")

BLUE = "#2a78d6"
ICON = {"ok": "✅", "warn": "⚠️", "error": "⛔"}
LEVEL_ICON = {"High": "🔴", "Medium": "🟠", "Info": "ℹ️"}


def month_options(n: int = 6) -> list[str]:
    today = date.today()
    out = []
    for i in range(1, n + 1):
        m = (today.month - 1 + i) % 12 + 1
        y = today.year + (today.month - 1 + i) // 12
        out.append(date(y, m, 1).strftime("%B %Y"))
    return out


st.title("Allocation Planner")
st.caption("Demo with made-up data. Same steps as the allocation engine: rules for special products, "
           "then a cost-minimising solver for the rest.")

tab_setup, tab_plan, tab_warn = st.tabs(["1 · Set up & run", "2 · Plan", "3 · Warnings"])

# ============================================================== 1. SET UP & RUN
with tab_setup:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("New monthly run")
        c1, c2 = st.columns(2)
        month = c1.selectbox("Month", month_options())
        scenario = c2.selectbox("Scenario", list(dd.SCENARIOS),
                                help="Peak season cuts warehouse space by 30% to show what happens when space runs out.")

        st.markdown("**Warehouse limits** (share of the optimizer's units)")
        limits = st.data_editor(
            dd.default_limits(), key="limits", hide_index=True, width="stretch",
            disabled=["Warehouse"],
            column_config={
                "Min %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%.0f%%"),
                "Max %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%.0f%%"),
            },
        )

        with st.expander("Settings"):
            s1, s2, s3 = st.columns(3)
            penalty = s1.number_input("Overflow penalty ($ per unit)", 1.0, 500.0, 25.0, 5.0,
                                      help="Cost of placing a unit beyond a warehouse's space when space is short.")
            fill = s2.radio("Fill missing shipping costs with", ["Warehouse average", "Minimum cost (current code)"],
                            help="The current code uses the minimum, which makes unknown lanes look cheapest.")
            rule = s3.radio("Slow sellers go to", ["Cheapest warehouse", "Random warehouse (current code)"])

        with st.expander("Demo input data"):
            st.markdown("**Warehouse space** (editable)")
            capacity = st.data_editor(dd.capacity_table(scenario), key=f"cap_{scenario}", hide_index=True,
                                      disabled=["warehouse", "storage"], width="stretch")
            demand = dd.demand_table()
            st.markdown("**Demand by zone**")
            st.dataframe(demand.pivot(index="product", columns="zone", values="units")
                         .reindex(columns=dd.ZONES).fillna(0).astype(int), width="stretch")
            st.markdown("**Products**")
            st.dataframe(dd.PRODUCTS.drop(columns=["size_factor"]), hide_index=True, width="stretch")

    costs = dd.cost_table()
    checks = pre_run_checks(limits, costs, demand, capacity, dd.PRODUCTS)
    blocked = any(c["level"] == "error" for c in checks)

    with right:
        with st.container(border=True):
            st.markdown("**Checks before running**")
            for c in checks:
                st.markdown(f"{ICON[c['level']]} {c['text']}")
        run = st.button("Run allocation", type="primary", disabled=blocked, width="stretch")
        st.caption("Fix every ⛔ before running." if blocked else "Usually takes a second on demo data.")

        if run:
            with st.spinner("Placing products and solving…"):
                result = run_allocation(
                    dd.PRODUCTS, demand, costs, capacity, limits, penalty=penalty,
                    missing_cost_fill="average" if fill.startswith("Warehouse") else "min",
                    slow_seller_rule="cheapest" if rule.startswith("Cheapest") else "random",
                )
            st.session_state["result"] = result
            st.session_state["run_meta"] = {"month": month, "scenario": scenario}
            st.session_state.pop("approved", None)
            if result["status"] == "optimal":
                st.success("Plan ready — open **2 · Plan**.")
            else:
                st.error("No plan fits these rules — see **2 · Plan**.")

result = st.session_state.get("result")
meta = st.session_state.get("run_meta", {})

# ============================================================== 2. PLAN
with tab_plan:
    if not result:
        st.info("Run the allocation on **1 · Set up & run** to see the plan.")
    elif result["status"] != "optimal":
        st.error(f"No plan found ({result['status']}). Usually the share limits or space are too tight. "
                 f"Solver message: {result['message']}")
    else:
        head_l, head_r = st.columns([3, 2])
        head_l.subheader(f"Plan for {meta['month']}")
        head_l.caption(f"Scenario: {meta['scenario']}")
        alloc = result["alloc"]
        export = alloc.assign(units=alloc["units"].round(1), cost=alloc["cost"].round(2))
        b1, b2 = head_r.columns(2)
        b1.download_button("Export CSV", export.to_csv(index=False), f"allocation_{meta['month'].replace(' ', '_')}.csv",
                           "text/csv", width="stretch")
        if b2.button("Approve", type="primary", width="stretch"):
            st.session_state["approved"] = True

        if st.session_state.get("approved"):
            st.success("✅ Plan approved.")
        elif result["overflow"]:
            st.warning("⚠️ Best plan found, but some warehouses are over their space limit. See **3 · Warnings**.")
        else:
            st.success("✅ Best plan found. Every zone gets its demand within space and share limits.")

        warns = result["warnings"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total shipping cost", f"${result['shipping_cost']:,.0f}",
                  help="Excludes overflow penalties, which are a planning signal, not real cost.")
        m2.metric("Units placed", f"{result['units']:,.0f}")
        m3.metric("Warnings", f"{len(warns)}", help="See 3 · Warnings")

        t_col, c_col = st.columns([3, 2], gap="large")
        with t_col:
            st.markdown("**Space used by warehouse and storage type**")
            s = result["summary"].copy()
            s["used %"] = (s["used"] * 100).round(0)
            st.dataframe(
                s[["warehouse", "storage", "allocated", "capacity", "used %", "overflow"]],
                hide_index=True, width="stretch",
                column_config={
                    "warehouse": "Warehouse", "storage": "Storage",
                    "allocated": st.column_config.NumberColumn("Units", format="%.0f"),
                    "capacity": st.column_config.NumberColumn("Space", format="%.0f"),
                    "used %": st.column_config.ProgressColumn("Space used", min_value=0, max_value=100, format="%.0f%%"),
                    "overflow": st.column_config.NumberColumn("Over limit", format="%.0f"),
                },
            )
            st.markdown("**Share of optimizer units vs limits**")
            sh = result["shares"].copy()
            for col in ["share", "min", "max"]:
                sh[col] = (sh[col] * 100).round(1)
            st.dataframe(sh, hide_index=True, width="stretch", column_config={
                "warehouse": "Warehouse",
                "share": st.column_config.NumberColumn("Share", format="%.1f%%"),
                "min": st.column_config.NumberColumn("Min", format="%.0f%%"),
                "max": st.column_config.NumberColumn("Max", format="%.0f%%"),
            })

        with c_col:
            st.markdown("**Shipping cost by warehouse**")
            by_wh = alloc.groupby("warehouse", as_index=False)["cost"].sum()
            chart = (
                alt.Chart(by_wh)
                .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48)
                .encode(
                    x=alt.X("warehouse:N", title=None, sort=dd.WAREHOUSES, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("cost:Q", title="Shipping cost ($)", axis=alt.Axis(format="$,.0f", grid=True, gridOpacity=0.3)),
                    tooltip=[alt.Tooltip("warehouse:N", title="Warehouse"),
                             alt.Tooltip("cost:Q", title="Cost", format="$,.0f")],
                )
                .properties(height=300)
                .configure_view(stroke=None)
            )
            st.altair_chart(chart, width="stretch")

        with st.expander("All allocations"):
            pick = st.multiselect("Filter products", sorted(alloc["product"].unique()))
            view = export if not pick else export[export["product"].isin(pick)]
            st.dataframe(view.drop(columns=["filled_cost"]), hide_index=True, width="stretch")

# ============================================================== 3. WARNINGS
with tab_warn:
    if not result or result["status"] != "optimal":
        st.info("Warnings appear here after a successful run.")
    else:
        warns = result["warnings"]
        st.subheader(f"Things to check · {len(warns)}")
        if not warns:
            st.success("Nothing to check.")
        for i, w in enumerate(warns):
            with st.container(border=True):
                st.markdown(f"{LEVEL_ICON[w['level']]} **{w['level']}** · **{w['title']}**")
                st.write(w["detail"])
                if w["rows"] is not None and len(w["rows"]):
                    with st.expander("Show rows"):
                        st.dataframe(w["rows"], hide_index=True, width="stretch")
