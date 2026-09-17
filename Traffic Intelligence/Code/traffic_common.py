"""
Shared utilities for the Traffic Intelligence pipeline
=========================================================
Centralises logic that used to be copy-pasted across
traffic_flow_prediction_v2.py, junction_network.py, junction_predictor.py
and junction_cli.py: file paths, the Birmingham-relative directional
feature, the shared encoding pipeline, and junction/road-name cleaning.

Having one copy matters here specifically because the junction-label
cleaning rules (see load_junction_subset below) have already needed several
rounds of fixes (LA Boundary, generic road-name labels, bare M6 junction
numbers, a raw-data typo) — with three separate copies, every fix had to be
applied three times and was easy to miss in one of them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file's location (Traffic Intelligence/
# Code/) rather than hardcoded to one machine/user, so the pipeline runs
# unchanged regardless of who checks it out or which directory it's run from.
# ---------------------------------------------------------------------------
TRAFFIC_INTELLIGENCE_DIR = Path(__file__).resolve().parents[1]   # .../Traffic Intelligence
PROJECT_ROOT = TRAFFIC_INTELLIGENCE_DIR.parent                    # .../FYP v.2

DATA_PATH = TRAFFIC_INTELLIGENCE_DIR / "dft_rawcount_local_authority_id_141.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Directional in/out-of-Birmingham feature
# ---------------------------------------------------------------------------
# direction_of_travel is only a compass direction (N/S/E/W) — it doesn't say
# whether that direction means "into Birmingham" or "out of it", since that
# depends on where the count point sits relative to the city centre. We
# compute the bearing from the city centre to each count point, compare it
# to the recorded direction of travel, and classify each observation as:
#   - inbound  : travelling roughly opposite the centre->point bearing
#   - outbound : travelling roughly along the centre->point bearing
#   - lateral  : travelling roughly perpendicular to that bearing
BIRMINGHAM_CENTER_LAT = 52.4796
BIRMINGHAM_CENTER_LON = -1.9026
CARDINAL_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def bearing_to_cardinal(lat_center, lon_center, lat_point, lon_point):
    """Nearest compass cardinal (N/E/S/W) from the centre to a point."""
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


def add_engineered_features(df):
    """Adds is_morning_peak / is_evening_peak / is_peak_hour and
    flow_direction (inbound/outbound/lateral relative to Birmingham's
    centre) to df in place, and returns it for convenience."""
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([16, 17, 18]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["bearing_cardinal"] = [
        bearing_to_cardinal(BIRMINGHAM_CENTER_LAT, BIRMINGHAM_CENTER_LON, lat, lon)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    df["flow_direction"] = [
        classify_flow(d, b) for d, b in zip(df["direction_of_travel"], df["bearing_cardinal"])
    ]
    return df


# ---------------------------------------------------------------------------
# Modelling feature lists + shared preprocessing pipeline
# ---------------------------------------------------------------------------
TARGET = "all_motor_vehicles"
FEATURES = [
    "year", "hour", "count_point_id", "road_type", "road_name",
    "direction_of_travel", "latitude", "longitude", "is_peak_hour",
    "flow_direction",
]
# High-cardinality columns -> target encoding; low-cardinality -> one-hot.
# One-hot encoding count_point_id/road_name (~600 dummy columns) made Linear
# Regression extrapolate wildly on unseen locations under grouped CV — see
# README Key Finding 2.
HIGH_CARDINALITY = ["count_point_id", "road_name"]
LOW_CARDINALITY_CATEGORICAL = ["road_type", "direction_of_travel", "flow_direction"]
NUMERIC_FEATURES = [c for c in FEATURES if c not in HIGH_CARDINALITY + LOW_CARDINALITY_CATEGORICAL]


def build_preprocessor(random_state=RANDOM_STATE):
    """The shared ColumnTransformer used by every script that trains a
    model: target encoding for high-cardinality location/road identifiers,
    one-hot for low-cardinality categoricals, median imputation for
    numerics. sklearn's TargetEncoder cross-fits internally, so
    fit_transform does not leak the target into itself."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("onehot", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), LOW_CARDINALITY_CATEGORICAL),
        ("target", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", TargetEncoder(target_type="continuous", random_state=random_state)),
        ]), HIGH_CARDINALITY),
    ])


def train_full_model(df, rf_params):
    """Fits the shared preprocessor + a Random Forest with the given
    hyperparameters on the FULL dataset. Used by the junction prediction
    tools, which reuse hyperparameters already tuned in
    traffic_flow_prediction_v2.py rather than re-tuning (this is a
    prediction tool, not an evaluation exercise)."""
    X = df[FEATURES]
    y = df[TARGET]
    preprocessor = build_preprocessor()
    X_processed = np.asarray(preprocessor.fit_transform(X, y))
    model = RandomForestRegressor(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_processed, y)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Junction / road-name cleaning
# ---------------------------------------------------------------------------
# See README "Extension: Junction-Level Traffic Prediction" for the
# data-quality issues these fix: inconsistent road-name variants (A38M vs
# A38(M)), generic junction labels that are really "this road crosses X
# somewhere" rather than one point (LA Boundary, bare road names, bare M6
# junction numbers), and a one-off spelling typo in the raw data.
GENERIC_JUNCTION_LABELS = {"la boundary"}

JUNCTION_LABEL_CORRECTIONS = {
    "A456/A457 rooundabout": "A456/A457 roundabout",
}

LINK_META_COLUMNS = [
    "count_point_id", "road_name_canonical", "road_type", "latitude", "longitude",
    "direction_of_travel", "flow_direction", "start_junction_clean", "end_junction_clean",
]


def canonical_road_name(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def build_road_name_lookup(road_names):
    """Maps every road_name spelling to a single canonical value, merging
    variants like 'A38M' / 'A38(M)' that would otherwise fragment one
    physical road into separate graph identities."""
    lookup = {}
    for raw_name in pd.unique(road_names):
        lookup.setdefault(canonical_road_name(raw_name), raw_name)
    return lookup


def is_generic_junction_label(label, known_road_names_canonical):
    label = str(label).strip()
    return (
        canonical_road_name(label) in known_road_names_canonical
        or label.lower() in GENERIC_JUNCTION_LABELS
        or label.isdigit()
    )


def disambiguate_junction(junction_name, road_name, known_road_names_canonical):
    """Qualifies a junction label with the road referencing it whenever the
    label alone is ambiguous (a bare road name, a catch-all like 'LA
    Boundary', or a bare number such as an M6 junction number) — otherwise
    unrelated physical junctions collapse into one graph node."""
    label = str(junction_name).strip()
    if is_generic_junction_label(label, known_road_names_canonical):
        return f"{label} (via {road_name})"
    return label


def load_junction_subset(df):
    """Returns the subset of rows with both junction endpoints populated,
    with road-name variants merged, the known 'rooundabout' typo corrected,
    and start/end junction labels disambiguated. `df` must already have
    add_engineered_features applied if flow_direction/is_peak_hour are
    needed downstream (e.g. for link_meta)."""
    has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
    subset = df[has_both].copy()

    subset["start_junction_road_name"] = subset["start_junction_road_name"].replace(JUNCTION_LABEL_CORRECTIONS)
    subset["end_junction_road_name"] = subset["end_junction_road_name"].replace(JUNCTION_LABEL_CORRECTIONS)

    lookup = build_road_name_lookup(subset["road_name"])
    subset["road_name_canonical"] = subset["road_name"].apply(lambda n: lookup[canonical_road_name(n)])
    known_road_names_canonical = set(subset["road_name_canonical"].unique())

    subset["start_junction_clean"] = [
        disambiguate_junction(j, r, known_road_names_canonical)
        for j, r in zip(subset["start_junction_road_name"], subset["road_name_canonical"])
    ]
    subset["end_junction_clean"] = [
        disambiguate_junction(j, r, known_road_names_canonical)
        for j, r in zip(subset["end_junction_road_name"], subset["road_name_canonical"])
    ]
    return subset


def build_link_meta(subset):
    """The static per-(count_point_id, direction) attributes needed to
    build a feature row for prediction at any year/hour."""
    return subset[LINK_META_COLUMNS].drop_duplicates()


def build_junction_graph(subset):
    """Builds a MultiGraph (junctions as nodes, links as edges) from a
    cleaned junction subset. A MultiGraph — not a plain Graph — matters
    because distinct roads can share the same two junction endpoints (e.g.
    both A38 and A4400 link 'A456/A457 roundabout' to 'B4100'); a plain
    Graph would silently drop one of the two links when both edges are
    added between the same node pair."""
    graph = nx.MultiGraph()
    link_rows = subset[["start_junction_clean", "end_junction_clean", "count_point_id"]].drop_duplicates()
    for _, row in link_rows.iterrows():
        graph.add_edge(
            row["start_junction_clean"], row["end_junction_clean"],
            count_point_id=row["count_point_id"],
        )
    return graph


def predict_junction_traffic(junction_name, year, hour, model, preprocessor, link_meta, graph):
    """
    Predict total traffic for every link touching `junction_name`, at the
    given year/hour, using the trained model. Returns (breakdown_df, total).
    """
    if junction_name not in graph:
        raise ValueError(f"'{junction_name}' not found in the junction graph. "
                          f"Check outputs/junction_totals.csv for valid names.")

    is_peak_hour = int(hour in [7, 8, 9, 16, 17, 18])

    connected_count_points = {
        data["count_point_id"] for _, _, data in graph.edges(junction_name, data=True)
    }

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
