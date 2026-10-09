"""Simplified allocation engine.

Follows the same steps as the production pipeline:
  1. sort products (optimizer / slow seller / no demand)
  2. place special products by rules
  3. reduce warehouse space by what the rules used
  4. feasibility check -> soft (penalised) space limits where demand > space
  5. linear program: minimise shipping cost + overflow penalty
  6. combine everything into one plan and a list of warnings

Solved with SciPy's HiGHS (open source) instead of Gurobi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

MIN_COST = 0.5          # what the production code clips missing costs to
MIN_CAP_BUFFER = 0.95   # min share can't exceed 95% of a warehouse's space


# --------------------------------------------------------------------------- checks
def pre_run_checks(limits: pd.DataFrame, costs: pd.DataFrame, demand: pd.DataFrame,
                   capacity: pd.DataFrame, products: pd.DataFrame) -> list[dict]:
    """Checks shown before the run. level: ok / warn / error (error blocks the run)."""
    checks = [{"level": "ok", "text": "Demo input data loaded"}]

    lim = limits.fillna(-1)
    bad_range = lim[(lim["Min %"] < 0) | (lim["Max %"] < 0) | (lim["Max %"] > 100)]
    if len(bad_range):
        checks.append({"level": "error", "text": f"Fill in min and max (0–100%) for {', '.join(bad_range['Warehouse'])}"})
    if lim["Max %"].sum() < 100:
        checks.append({"level": "error", "text": f"Max shares add up to {lim['Max %'].sum():.0f}% (need at least 100%)"})
    else:
        checks.append({"level": "ok", "text": "Max shares add up to at least 100%"})
    over = lim[lim["Min %"] > lim["Max %"]]
    if len(over):
        checks.append({"level": "error", "text": f"Min is above max for {', '.join(over['Warehouse'])}"})
    else:
        checks.append({"level": "ok", "text": "Each min is at or below its max"})
    if lim["Min %"].sum() > 100:
        checks.append({"level": "error", "text": f"Min shares add up to {lim['Min %'].sum():.0f}% (must be 100% or less)"})
    else:
        checks.append({"level": "ok", "text": "Min shares add up to 100% or less"})

    n_missing = int(costs["cost"].isna().sum())
    if n_missing:
        checks.append({"level": "warn", "text": f"{n_missing} shipping cost lane(s) missing — will be filled"})

    storage = products.set_index("product")["storage"]
    d = demand.assign(storage=demand["product"].map(storage)).groupby("storage")["units"].sum()
    cap = capacity.groupby("storage")["capacity"].sum()
    for k, units in d.items():
        if units > cap.get(k, 0):
            checks.append({"level": "warn", "text": f"{k} demand ({units:,.0f}) is more than total {k} space ({cap.get(k, 0):,.0f}) — limit becomes soft"})
    return checks


# --------------------------------------------------------------------------- engine
def run_allocation(products: pd.DataFrame, demand: pd.DataFrame, costs: pd.DataFrame,
                   capacity: pd.DataFrame, limits: pd.DataFrame, *, penalty: float = 25.0,
                   missing_cost_fill: str = "average", slow_seller_rule: str = "cheapest",
                   seed: int | None = None) -> dict:
    warehouses = list(limits["Warehouse"])
    zones = sorted(demand["zone"].unique(), key=list(demand["zone"].unique()).index)
    prod = products.set_index("product")
    storage_of = prod["storage"].to_dict()

    # ---- fill missing costs
    costs = costs.copy()
    costs["filled"] = costs["cost"].isna()
    if missing_cost_fill == "average":
        wh_avg = costs.groupby("warehouse")["cost"].transform("mean")
        costs["cost"] = costs["cost"].fillna(wh_avg)
    else:
        costs["cost"] = costs["cost"].fillna(MIN_COST)
    costs["cost"] = costs["cost"].clip(lower=MIN_COST)
    cost = costs.set_index(["warehouse", "zone", "product"])["cost"].to_dict()
    filled = set(costs.loc[costs["filled"], ["warehouse", "zone", "product"]].itertuples(index=False, name=None))

    # ---- 1. sort products
    total_demand = demand.groupby("product")["units"].sum()
    groups = {"ok": [], "low_vel": [], "no_demand": []}
    for p, row in prod.iterrows():
        if total_demand.get(p, 0) <= 0:
            groups["no_demand"].append(p)
        elif row["velocity"] == "low":
            groups["low_vel"].append(p)
        else:
            groups["ok"].append(p)

    remaining = capacity.set_index(["warehouse", "storage"])["capacity"].astype(float).to_dict()
    rule_rows: list[dict] = []
    rng = np.random.default_rng(seed)

    def allowed(p: str) -> list[str]:
        r = prod.loc[p]
        wh = r["allowed"] if isinstance(r["allowed"], list) else warehouses
        return [w for w in wh if w not in r["forbidden"]]

    # ---- 2. rule-based placements
    for p in groups["low_vel"]:
        k = storage_of[p]
        need = total_demand[p]
        dz = demand[demand["product"] == p]
        options = []
        for w in allowed(p):
            if remaining.get((w, k), 0) >= need:
                total = sum(cost[(w, z, p)] * u for z, u in zip(dz["zone"], dz["units"]))
                options.append((total, w))
        if not options:
            options = [(0.0, w) for w in allowed(p)]
        if slow_seller_rule == "random":
            chosen = options[rng.integers(len(options))][1]
        else:
            chosen = min(options)[1]
        for z, u in zip(dz["zone"], dz["units"]):
            rule_rows.append({"warehouse": chosen, "zone": z, "product": p, "storage": k, "units": u,
                              "unit_cost": cost[(chosen, z, p)], "method": "Slow seller → 1 warehouse",
                              "filled_cost": (chosen, z, p) in filled})
        remaining[(chosen, k)] = remaining.get((chosen, k), 0) - need

    for p in groups["no_demand"]:
        k = storage_of[p]
        qty = float(prod.loc[p, "order_qty"])
        targets = [w for w in allowed(p) if remaining.get((w, k), 0) > 0] or allowed(p)
        share = qty / len(targets)
        for w in targets:
            rule_rows.append({"warehouse": w, "zone": "—", "product": p, "storage": k, "units": share,
                              "unit_cost": np.nan, "method": "No demand → even split", "filled_cost": False})
            remaining[(w, k)] = remaining.get((w, k), 0) - share

    # ---- 4. feasibility: soft limits where demand > space
    ok = groups["ok"]
    opt_demand = demand[demand["product"].isin(ok)]
    D = float(opt_demand["units"].sum())
    by_storage = opt_demand.assign(storage=opt_demand["product"].map(storage_of)).groupby("storage")["units"].sum()
    relaxed = set()
    for k, units in by_storage.items():
        space = sum(max(remaining.get((w, k), 0), 0) for w in warehouses)
        if units > space:
            relaxed |= {(w, k) for w in warehouses if remaining.get((w, k), 0) > 0}

    # ---- share limits in units
    lim = limits.set_index("Warehouse")
    max_u = {w: lim.loc[w, "Max %"] / 100 * D for w in warehouses}
    min_u = {}
    for w in warehouses:
        space_w = sum(max(remaining.get((w, k), 0), 0) for k in by_storage.index)
        min_u[w] = min(lim.loc[w, "Min %"] / 100 * D, MIN_CAP_BUFFER * space_w)

    # ---- 5. linear program
    d_jp = opt_demand.set_index(["zone", "product"])["units"].to_dict()
    var = [(w, z, p) for w in warehouses for z in zones for p in ok]
    vidx = {v: n for n, v in enumerate(var)}
    pen = sorted(relaxed)
    pidx = {v: len(var) + n for n, v in enumerate(pen)}
    n = len(var) + len(pen)

    c = np.zeros(n)
    bounds = []
    allowed_ok = {p: set(allowed(p)) for p in ok}
    for (w, z, p), i in vidx.items():
        c[i] = cost[(w, z, p)]
        bounds.append((0, 0) if w not in allowed_ok[p] else (0, None))
    for v, i in pidx.items():
        c[i] = penalty
        bounds.append((0, None))

    rows, rhs = [], []
    for z in zones:                                  # demand: sum_i x >= d
        for p in ok:
            dval = d_jp.get((z, p), 0.0)
            if dval > 0:
                r = np.zeros(n)
                for w in warehouses:
                    r[vidx[(w, z, p)]] = -1
                rows.append(r); rhs.append(-dval)
    for w in warehouses:                             # share: min <= sum x <= max
        r = np.zeros(n)
        for z in zones:
            for p in ok:
                r[vidx[(w, z, p)]] = 1
        rows.append(r.copy()); rhs.append(max_u[w])
        rows.append(-r); rhs.append(-min_u[w])
    for w in warehouses:                             # space per storage type
        for k in by_storage.index:
            r = np.zeros(n)
            for z in zones:
                for p in ok:
                    if storage_of[p] == k:
                        r[vidx[(w, z, p)]] = 1
            if (w, k) in pidx:
                r[pidx[(w, k)]] = -1
            rows.append(r); rhs.append(max(remaining.get((w, k), 0), 0))

    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")

    out = {"status": {0: "optimal", 2: "infeasible"}.get(res.status, "error"),
           "message": res.message, "rule_rows": pd.DataFrame(rule_rows), "groups": groups,
           "relaxed": pen, "filled_lanes": sorted(filled), "D": D,
           "min_units": min_u, "max_units": max_u}
    if res.status != 0:
        return out

    x = res.x
    opt_rows = []
    for (w, z, p), i in vidx.items():
        if x[i] > 1e-6:
            opt_rows.append({"warehouse": w, "zone": z, "product": p, "storage": storage_of[p],
                             "units": round(x[i], 2), "unit_cost": cost[(w, z, p)],
                             "method": "Optimizer", "filled_cost": (w, z, p) in filled})
    overflow = {v: round(x[i], 2) for v, i in pidx.items() if x[i] > 1e-6}

    alloc = pd.concat([pd.DataFrame(opt_rows), pd.DataFrame(rule_rows)], ignore_index=True)
    alloc["cost"] = (alloc["units"] * alloc["unit_cost"]).fillna(0.0)

    cap = capacity.set_index(["warehouse", "storage"])["capacity"]
    summary = (alloc.groupby(["warehouse", "storage"])["units"].sum().reindex(cap.index, fill_value=0)
               .to_frame("allocated").join(cap).reset_index())
    summary["used"] = np.where(summary["capacity"] > 0, summary["allocated"] / summary["capacity"], 0.0)
    summary["overflow"] = [overflow.get((w, k), 0.0) for w, k in zip(summary["warehouse"], summary["storage"])]

    opt_units = alloc[alloc["method"] == "Optimizer"].groupby("warehouse")["units"].sum().reindex(warehouses, fill_value=0)
    shares = pd.DataFrame({
        "warehouse": warehouses,
        "share": (opt_units / D).values if D else 0,
        "min": [lim.loc[w, "Min %"] / 100 for w in warehouses],
        "max": [lim.loc[w, "Max %"] / 100 for w in warehouses],
    })

    out.update({"alloc": alloc, "summary": summary, "shares": shares, "overflow": overflow,
                "total_cost": float(alloc["cost"].sum()) + penalty * sum(overflow.values()),
                "shipping_cost": float(alloc["cost"].sum()),
                "units": float(alloc["units"].sum())})
    out["warnings"] = build_warnings(out, penalty)
    return out


# --------------------------------------------------------------------------- warnings
def build_warnings(r: dict, penalty: float) -> list[dict]:
    w = []
    alloc = r["alloc"]
    if r["overflow"]:
        rows = pd.DataFrame([{"warehouse": a, "storage": b, "overflow units": u} for (a, b), u in r["overflow"].items()])
        w.append({"level": "High", "title": f"{len(rows)} warehouse space limit(s) exceeded",
                  "detail": f"Demand was higher than space, so {sum(r['overflow'].values()):,.0f} extra units were placed at a ${penalty:,.0f}/unit penalty.",
                  "rows": rows})
    if r["relaxed"] and not r["overflow"]:
        w.append({"level": "Medium", "title": "Space limits were made soft but not used",
                  "detail": "Total demand looked higher than total space, but the solver fit everything.", "rows": None})
    if r["filled_lanes"]:
        lanes = pd.DataFrame(r["filled_lanes"], columns=["warehouse", "zone", "product"])
        used = alloc[alloc["filled_cost"]]
        lvl = "High" if len(used) else "Medium"
        detail = (f"{len(lanes)} lane(s) had no cost data and were filled with an estimate. "
                  + (f"{used['units'].sum():,.0f} units ship on these lanes — check the estimate." if len(used)
                     else "No units ship on these lanes."))
        w.append({"level": lvl, "title": "Shipping costs were missing", "detail": detail,
                  "rows": used[["warehouse", "zone", "product", "units", "unit_cost"]] if len(used) else lanes})
    binding = r["shares"][(r["shares"]["share"] >= r["shares"]["max"] - 1e-4) | (r["shares"]["share"] <= r["shares"]["min"] + 1e-4)]
    if len(binding):
        w.append({"level": "Info", "title": f"{len(binding)} warehouse(s) at a share limit",
                  "detail": "These limits changed the plan; loosening them could lower cost.",
                  "rows": binding.assign(**{c: binding[c].map('{:.0%}'.format) for c in ["share", "min", "max"]})})
    rules = r["rule_rows"]
    if len(rules):
        g = rules.groupby(["product", "method", "warehouse"])["units"].sum().reset_index()
        w.append({"level": "Info", "title": f"{g['product'].nunique()} product(s) placed by business rules",
                  "detail": "Slow sellers and products with no demand skip the optimizer.", "rows": g})
    return w
