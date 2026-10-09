"""Synthetic demo data for the Allocation Planner.

Everything here is made up for illustration. It has the same shape as the
production inputs (warehouses, delivery zones, product types, storage types,
lane costs, demand and capacity), so the app walks through the same steps.
"""

import numpy as np
import pandas as pd

WAREHOUSES = ["WH-East", "WH-Central", "WH-West"]
ZONES = ["Northeast", "Southeast", "Midwest", "West"]
STORAGE_TYPES = ["FLA", "HFG"]  # FLA = folded / flat, HFG = hanging

# Base shipping cost per unit, warehouse -> zone (closer = cheaper).
BASE_LANE_COST = {
    "WH-East": {"Northeast": 4.1, "Southeast": 5.0, "Midwest": 6.2, "West": 9.5},
    "WH-Central": {"Northeast": 6.0, "Southeast": 5.6, "Midwest": 4.3, "West": 7.4},
    "WH-West": {"Northeast": 9.8, "Southeast": 8.6, "Midwest": 6.9, "West": 4.0},
}

# product, storage, velocity, size factor, allowed warehouses (medium velocity),
# forbidden warehouses (business exception), order qty (used only when no demand)
PRODUCTS = pd.DataFrame(
    [
        ("Tops-V101-high", "FLA", "high", 1.0, None, [], 0),
        ("Denim-V204-high", "FLA", "high", 1.2, None, [], 0),
        ("HomeDecor-V960-high", "FLA", "high", 1.5, None, [], 0),
        ("Suits-V850-high", "HFG", "high", 1.4, None, ["WH-West"], 0),
        ("Dresses-V310-medium", "HFG", "medium", 1.1, ["WH-East", "WH-West"], [], 0),
        ("Coats-V415-medium", "HFG", "medium", 1.6, ["WH-East", "WH-Central"], [], 0),
        ("Scarves-V520-low", "FLA", "low", 0.6, None, [], 0),
        ("Belts-V630-low", "FLA", "low", 0.7, None, [], 0),
        ("Socks-V740-high", "FLA", "high", 0.5, None, [], 400),
    ],
    columns=["product", "storage", "velocity", "size_factor", "allowed", "forbidden", "order_qty"],
)

# Monthly unit demand by zone (Northeast, Southeast, Midwest, West).
_DEMAND = {
    "Tops-V101-high": [2400, 1800, 1500, 1700],
    "Denim-V204-high": [1500, 1300, 1100, 1600],
    "HomeDecor-V960-high": [700, 600, 500, 400],
    "Suits-V850-high": [500, 350, 400, 450],
    "Dresses-V310-medium": [900, 700, 500, 800],
    "Coats-V415-medium": [600, 300, 700, 500],
    "Scarves-V520-low": [60, 40, 50, 30],
    "Belts-V630-low": [40, 30, 30, 50],
    # Socks-V740-high has no demand history on purpose.
}

# One lane has no cost data, to show how missing costs are filled.
MISSING_LANES = [("WH-West", "Southeast", "HomeDecor-V960-high")]

BASE_CAPACITY = {
    ("WH-East", "FLA"): 9000, ("WH-East", "HFG"): 3000,
    ("WH-Central", "FLA"): 7000, ("WH-Central", "HFG"): 3500,
    ("WH-West", "FLA"): 6000, ("WH-West", "HFG"): 2500,
}

SCENARIOS = {
    "Normal month": 1.0,
    "Peak season (tight space)": 0.7,
}


def demand_table() -> pd.DataFrame:
    rows = []
    for product, units in _DEMAND.items():
        for zone, u in zip(ZONES, units):
            rows.append({"zone": zone, "product": product, "units": float(u)})
    return pd.DataFrame(rows)


def cost_table() -> pd.DataFrame:
    rows = []
    sizes = PRODUCTS.set_index("product")["size_factor"]
    for wh in WAREHOUSES:
        for zone in ZONES:
            for product in PRODUCTS["product"]:
                cost = round(BASE_LANE_COST[wh][zone] * sizes[product], 2)
                if (wh, zone, product) in MISSING_LANES:
                    cost = np.nan
                rows.append({"warehouse": wh, "zone": zone, "product": product, "cost": cost})
    return pd.DataFrame(rows)


def capacity_table(scenario: str) -> pd.DataFrame:
    factor = SCENARIOS[scenario]
    rows = [
        {"warehouse": wh, "storage": k, "capacity": int(round(cap * factor))}
        for (wh, k), cap in BASE_CAPACITY.items()
    ]
    return pd.DataFrame(rows)


def default_limits() -> pd.DataFrame:
    return pd.DataFrame(
        {"Warehouse": WAREHOUSES, "Min %": [20.0, 20.0, 15.0], "Max %": [50.0, 45.0, 40.0]}
    )
