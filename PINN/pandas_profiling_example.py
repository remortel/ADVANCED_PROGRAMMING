"""
Extensive example of pandas operations with cProfile.
Generates a large synthetic sales dataset and performs:
  - groupby()
  - pivot_table()
  - apply()
  - map()
  - query()
Each operation is profiled individually and combined into one SnakeViz-ready file.
"""

import pandas as pd
import numpy as np
from cProfile import Profile
from pstats import Stats
import os

# ──────────────────────────────────────────────
# 1. GENERATE AND SAVE A LARGE CSV DATA FILE
# ──────────────────────────────────────────────

DATA_FILE = "sales_data.csv"
N_ROWS = 500_000

def generate_data():
    """Generate a synthetic sales CSV with 500,000 rows."""
    print(f"Generating {N_ROWS:,} rows of synthetic sales data...")
    rng = np.random.default_rng(42)

    regions     = ["North", "South", "East", "West", "Central"]
    categories  = ["Electronics", "Clothing", "Food", "Furniture", "Sports"]
    products    = [f"Product_{i}" for i in range(1, 51)]       # 50 products
    salespeople = [f"Rep_{i}"     for i in range(1, 101)]      # 100 sales reps

    df = pd.DataFrame({
        "date":        pd.date_range("2020-01-01", periods=N_ROWS, freq="1min"),
        "region":      rng.choice(regions,     size=N_ROWS),
        "category":    rng.choice(categories,  size=N_ROWS),
        "product":     rng.choice(products,    size=N_ROWS),
        "salesperson": rng.choice(salespeople, size=N_ROWS),
        "units_sold":  rng.integers(1, 100,    size=N_ROWS),
        "unit_price":  rng.uniform(5.0, 500.0, size=N_ROWS).round(2),
        "discount_pct":rng.uniform(0.0, 0.30,  size=N_ROWS).round(3),
        "returned":    rng.choice([True, False],size=N_ROWS, p=[0.05, 0.95]),
    })

    df.to_csv(DATA_FILE, index=False)
    print(f"Saved '{DATA_FILE}' ({os.path.getsize(DATA_FILE) / 1e6:.1f} MB)\n")


# ──────────────────────────────────────────────
# 2. READ THE DATA FILE
# ──────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Read the CSV, parse dates, and add a revenue column."""
    print("Loading data from CSV...")
    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
        dtype={
            "region":      "category",
            "category":    "category",
            "product":     "category",
            "salesperson": "category",
            "returned":    "bool",
        },
    )
    # Derived column used throughout
    df["revenue"] = (df["units_sold"] * df["unit_price"] * (1 - df["discount_pct"])).round(2)
    print(f"Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory\n")
    return df


# ──────────────────────────────────────────────
# 3. PANDAS OPERATIONS
# ──────────────────────────────────────────────

def op_groupby(df: pd.DataFrame):
    """
    groupby() – aggregate revenue and units by region and category.
    Shows total revenue, mean unit price, and transaction count.
    """
    result = (
        df.groupby(["region", "category"], observed=True)
        .agg(
            total_revenue =("revenue",    "sum"),
            mean_price    =("unit_price", "mean"),
            transactions  =("revenue",    "count"),
            total_units   =("units_sold", "sum"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
    )
    print("── groupby() ──────────────────────────────")
    print(result.head(10))
    print()
    return result


def op_pivot_table(df: pd.DataFrame):
    """
    pivot_table() – cross-tabulate total revenue by region (rows)
    and category (columns), with grand totals.
    """
    result = pd.pivot_table(
        df,
        values="revenue",
        index="region",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
        observed=True,
    ).round(2)
    print("── pivot_table() ──────────────────────────")
    print(result)
    print()
    return result


def op_apply(df: pd.DataFrame):
    """
    apply() – use a custom function to compute a 'performance tier'
    for each row based on revenue thresholds, then summarise by tier.
    """
    def revenue_tier(row):
        rev = row["revenue"]
        if rev >= 40_000:
            return "Platinum"
        elif rev >= 20_000:
            return "Gold"
        elif rev >= 5_000:
            return "Silver"
        else:
            return "Bronze"

    # apply() along axis=1 (row-wise)
    df = df.copy()
    df["tier"] = df.apply(revenue_tier, axis=1)

    summary = df.groupby("tier")["revenue"].agg(["count", "sum", "mean"]).round(2)
    print("── apply() ────────────────────────────────")
    print(summary)
    print()
    return df


def op_map(df: pd.DataFrame):
    """
    map() – replace region abbreviations using a lookup dictionary,
    and convert the boolean 'returned' flag to a readable label.
    """
    region_full = {
        "North":   "Northern Territory",
        "South":   "Southern Territory",
        "East":    "Eastern Territory",
        "West":    "Western Territory",
        "Central": "Central Territory",
    }
    returned_label = {True: "Returned", False: "Completed"}

    df = df.copy()
    df["region_full"]     = df["region"].map(region_full)
    df["order_status"]    = df["returned"].map(returned_label)

    print("── map() ──────────────────────────────────")
    print(df[["region", "region_full", "returned", "order_status"]].head(8).to_string(index=False))
    print()
    return df


def op_query(df: pd.DataFrame):
    """
    query() – filter rows using a readable string expression:
      • Electronics or Furniture category
      • revenue > 10,000
      • not returned
      • in the East or West region
    """
    result = df.query(
        "category in ['Electronics', 'Furniture'] "
        "and revenue > 10_000 "
        "and returned == False "
        "and region in ['East', 'West']"
    )

    print("── query() ────────────────────────────────")
    print(f"Rows matching filter: {len(result):,}")
    print(result[["date", "region", "category", "product", "revenue"]].head(8).to_string(index=False))
    print()
    return result


# ──────────────────────────────────────────────
# 4. PROFILE EACH OPERATION AND COMBINE
# ──────────────────────────────────────────────

def main():
    # Generate data file if it doesn't exist yet
    if not os.path.exists(DATA_FILE):
        generate_data()

    df = load_data()

    profiler = Profile()

    print("Profiling pandas operations...\n")

    profiler.runcall(op_groupby,      df)
    profiler.runcall(op_pivot_table,  df)
    profiler.runcall(op_apply,        df)
    profiler.runcall(op_map,          df)
    profiler.runcall(op_query,        df)

    # ── Print summary to console ──
    print("=" * 55)
    print("PROFILER SUMMARY (top 20 by cumulative time)")
    print("=" * 55)
    stats = Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    # ── Dump combined stats for SnakeViz ──
    PROF_FILE = "pandas_ops.prof"
    stats.dump_stats(PROF_FILE)
    print(f"\nProfile data saved to '{PROF_FILE}'")
    print(f"Visualise with:  snakeviz {PROF_FILE}")


if __name__ == "__main__":
    main()
