"""
Junction Traffic Predictor — Birmingham Major Road Network
==============================================================
Extends junction_network.py: instead of historical average traffic per
link, this predicts traffic using a trained Random Forest model for any
YEAR and HOUR you specify, then aggregates predictions across every link
touching a chosen junction.

Scope and honest limitations:
  - This predicts TOTAL volume on each link connected to a junction — it
    does not split traffic into "arriving from the north" vs "leaving to
    the south" for that specific junction, because the data only gives
    coordinates for each LINK, not for each junction endpoint separately.
    A true turning-movement breakdown would need that extra geometry.
  - `flow_direction` (inbound/outbound/lateral) is included as an
    approximate directional signal, but it's relative to Birmingham's
    CENTRE, not to the specific junction being queried.
  - The model is trained once on the full dataset with hyperparameters
    already found via tuning in the main pipeline (traffic_flow_prediction_v3.py).
    Re-running that tuning here would be redundant and slow.

Run:
    python3 junction_predictor.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor
import networkx as nx

DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Best hyperparameters found via RandomizedSearchCV in traffic_flow_prediction_v3.py
# (re-tuning here would be redundant — reuse the already-validated result)
BEST_RF_PARAMS = {
    "n_estimators": 400,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 20,
}

BIRMINGHAM_CENTER_LAT = 52.4796
BIRMINGHAM_CENTER_LON = -1.9026
CARDINAL_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
GENERIC_JUNCTION_LABELS = {"la boundary"}


def bearing_to_cardinal(lat_center, lon_center, lat_point, lon_point):
    d_lat = lat_point - lat_center
    d_lon = lon_point - lon_center
    angle = np.degrees(np.arctan2(d_lon, d_lat)) % 360
    if angle >= 315 or angle < 45:
        return "N"
    elif angle < 135:
        return "E"
    elif angle < 225:
        return "S"
    else:
        return "W"


def classify_flow(direction, bearing_cardinal):
    if direction == CARDINAL_OPPOSITE.get(bearing_cardinal):
        return "inbound"
    elif direction == bearing_cardinal:
        return "outbound"
    else:
        return "lateral"


def canonical_road_name(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def disambiguate_junction(junction_name, road_name, known_road_names_canonical):
    label = str(junction_name).strip()
    is_generic = (
        canonical_road_name(label) in known_road_names_canonical
        or label.lower() in GENERIC_JUNCTION_LABELS
    )
    if is_generic:
        return f"{label} (via {road_name})"
    return label


# ---------------------------------------------------------------------------
# 1. Load data and engineer features (same as traffic_flow_prediction_v3.py)
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
df["count_point_id"] = df["count_point_id"].astype(str)

df["bearing_cardinal"] = [
    bearing_to_cardinal(BIRMINGHAM_CENTER_LAT, BIRMINGHAM_CENTER_LON, lat, lon)
    for lat, lon in zip(df["latitude"], df["longitude"])
]
df["flow_direction"] = [
    classify_flow(d, b) for d, b in zip(df["direction_of_travel"], df["bearing_cardinal"])
]

# ---------------------------------------------------------------------------
# 2. Train the model on the FULL dataset (this is a prediction tool, not an
#    evaluation exercise — evaluation/tuning already done in the main pipeline)
# ---------------------------------------------------------------------------
target = "all_motor_vehicles"
features = [
    "year", "hour", "count_point_id", "road_type", "road_name",
    "direction_of_travel", "latitude", "longitude", "is_peak_hour",
    "flow_direction",
]

X = df[features]
y = df[target]

HIGH_CARDINALITY = ["count_point_id", "road_name"]
LOW_CARDINALITY_CATEGORICAL = ["road_type", "direction_of_travel", "flow_direction"]
numeric_features = [c for c in features if c not in HIGH_CARDINALITY + LOW_CARDINALITY_CATEGORICAL]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("onehot", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), LOW_CARDINALITY_CATEGORICAL),
    ("target", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)),
    ]), HIGH_CARDINALITY),
])

print("Fitting preprocessor and model on full dataset...")
X_processed = np.asarray(preprocessor.fit_transform(X, y))

model = RandomForestRegressor(**BEST_RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_processed, y)
print("Model trained.")

# ---------------------------------------------------------------------------
# 3. Build the junction graph (same cleaning as junction_network.py)
# ---------------------------------------------------------------------------
has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
subset = df[has_both].copy()

canonical_lookup = {}
for raw_name in subset["road_name"].unique():
    canonical_lookup.setdefault(canonical_road_name(raw_name), raw_name)
subset["road_name_canonical"] = subset["road_name"].apply(
    lambda n: canonical_lookup[canonical_road_name(n)]
)
known_road_names_canonical = set(subset["road_name_canonical"].unique())
subset["start_junction_clean"] = [
    disambiguate_junction(j, r, known_road_names_canonical)
    for j, r in zip(subset["start_junction_road_name"], subset["road_name_canonical"])
]
subset["end_junction_clean"] = [
    disambiguate_junction(j, r, known_road_names_canonical)
    for j, r in zip(subset["end_junction_road_name"], subset["road_name_canonical"])
]

# Metadata per (count_point_id, direction_of_travel): the static attributes
# needed to build a feature row for prediction at any year/hour.
link_meta = subset[[
    "count_point_id", "road_name_canonical", "road_type", "latitude", "longitude",
    "direction_of_travel", "flow_direction", "start_junction_clean", "end_junction_clean",
]].drop_duplicates()

G = nx.Graph()
for _, row in link_meta[["start_junction_clean", "end_junction_clean", "count_point_id"]].drop_duplicates().iterrows():
    G.add_edge(row["start_junction_clean"], row["end_junction_clean"],
               count_point_id=row["count_point_id"])

print(f"Graph built: {G.number_of_nodes()} junction nodes, {G.number_of_edges()} links")


# ---------------------------------------------------------------------------
# 4. Prediction function
# ---------------------------------------------------------------------------
def predict_junction_traffic(junction_name, year, hour, model, preprocessor, link_meta, graph):
    """
    Predict total traffic for every link touching `junction_name`, at the
    given year/hour, using the trained model. Returns (breakdown_df, total).
    """
    if junction_name not in graph:
        raise ValueError(f"'{junction_name}' not found in the junction graph. "
                          f"Check outputs/junction_totals.csv for valid names.")

    is_peak_hour = int(hour in [7, 8, 9, 16, 17, 18])

    connected_count_points = set()
    for neighbor in graph.neighbors(junction_name):
        cp = graph[junction_name][neighbor]["count_point_id"]
        connected_count_points.add(cp)

    rows = []
    for cp in connected_count_points:
        cp_rows = link_meta[link_meta["count_point_id"] == cp]
        for _, r in cp_rows.iterrows():
            rows.append({
                "year": year,
                "hour": hour,
                "count_point_id": cp,
                "road_type": r["road_type"],
                "road_name": r["road_name_canonical"],
                "direction_of_travel": r["direction_of_travel"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "is_peak_hour": is_peak_hour,
                "flow_direction": r["flow_direction"],
            })

    query_df = pd.DataFrame(rows)
    query_processed = np.asarray(preprocessor.transform(query_df))
    query_df["predicted_traffic"] = model.predict(query_processed)

    breakdown = query_df.groupby(
        ["count_point_id", "road_name", "direction_of_travel", "flow_direction"]
    )["predicted_traffic"].sum().reset_index().sort_values("predicted_traffic", ascending=False)

    total = query_df["predicted_traffic"].sum()
    return breakdown, total


# ---------------------------------------------------------------------------
# 5. Example queries
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    junction_totals = pd.read_csv(OUTPUT_DIR / "junction_totals.csv") \
        if (OUTPUT_DIR / "junction_totals.csv").exists() else None

    example_junctions = ["A38", "A4040", "A41"]
    example_junctions = [j for j in example_junctions if j in G.nodes()]
    if not example_junctions:
        example_junctions = list(G.nodes())[:3]

    for junction in example_junctions:
        for hour in [8, 17]:  # morning and evening peak
            breakdown, total = predict_junction_traffic(
                junction, year=2025, hour=hour,
                model=model, preprocessor=preprocessor,
                link_meta=link_meta, graph=G,
            )
            print(f"\n=== Predicted traffic at junction '{junction}', "
                  f"year=2025, hour={hour}:00 ===")
            print(breakdown)
            print(f"TOTAL predicted traffic: {total:.1f}")

    print("\nTo query any junction yourself, call:")
    print("  predict_junction_traffic('A38', year=2026, hour=8, "
          "model=model, preprocessor=preprocessor, link_meta=link_meta, graph=G)")