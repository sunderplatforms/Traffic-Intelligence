"""
Generic Junction Label Detector
==================================
Systematically checks every junction label in the data for the same issue
found manually with "LA Boundary" (an artifact merging unrelated physical
locations) and verified NOT present for "M6" (a genuine single junction).

Run:
    python3 check_generic_junctions.py
"""

import math
from itertools import combinations

import pandas as pd

DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
DISTANCE_THRESHOLD_KM = 3.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def canonical_road_name(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


df = pd.read_csv(DATA_PATH, low_memory=False)

has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
subset = df[has_both].copy()

canonical_lookup = {}
for raw_name in subset["road_name"].unique():
    canonical_lookup.setdefault(canonical_road_name(raw_name), raw_name)
subset["road_name_canonical"] = subset["road_name"].apply(
    lambda n: canonical_lookup[canonical_road_name(n)]
)

long_rows = []
for _, row in subset.drop_duplicates(
    subset=["count_point_id", "start_junction_road_name", "end_junction_road_name"]
).iterrows():
    for label in (row["start_junction_road_name"], row["end_junction_road_name"]):
        long_rows.append({
            "junction_label": str(label).strip(),
            "count_point_id": row["count_point_id"],
            "road_name": row["road_name_canonical"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        })

long_df = pd.DataFrame(long_rows).drop_duplicates()

results = []
for label, group in long_df.groupby("junction_label"):
    points = group[["latitude", "longitude"]].drop_duplicates().values.tolist()
    n_roads = group["road_name"].nunique()
    n_count_points = group["count_point_id"].nunique()

    if len(points) > 1:
        max_dist = max(
            haversine_km(a[0], a[1], b[0], b[1]) for a, b in combinations(points, 2)
        )
    else:
        max_dist = 0.0

    results.append({
        "junction_label": label,
        "n_distinct_roads": n_roads,
        "n_count_points": n_count_points,
        "max_spread_km": round(max_dist, 2),
        "roads_involved": sorted(group["road_name"].unique().tolist()),
    })

results_df = pd.DataFrame(results).sort_values("max_spread_km", ascending=False)

flagged = results_df[
    (results_df["max_spread_km"] > DISTANCE_THRESHOLD_KM) & (results_df["n_distinct_roads"] > 1)
]

print(f"Checked {len(results_df)} distinct junction labels.\n")
print(f"=== Flagged as likely generic/ambiguous (spread > {DISTANCE_THRESHOLD_KM}km, "
      f"multiple roads) ===")
if flagged.empty:
    print("None found beyond what's already been fixed.")
else:
    print(flagged.to_string(index=False))

print(f"\n=== Top 15 labels by geographic spread (for manual review) ===")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("outputs/junction_label_spread_check.csv", index=False)
print("\nFull results saved to outputs/junction_label_spread_check.csv")