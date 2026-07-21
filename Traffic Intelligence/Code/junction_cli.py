"""
Junction Traffic Predictor — Interactive CLI
================================================
An interactive command-line front-end for junction_predictor.py.

Trains the model once on startup, then lets you repeatedly query:
  - a junction name (from the network built out of the junction-defined
    subset of the data — 170 links across 23 major roads)
  - a year
  - an hour (0-23)

...and prints the predicted traffic breakdown for every link touching
that junction, using the trained Random Forest model.

Run:
    python3 junction_cli.py
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
        or label.isdigit()
    )
    if is_generic:
        return f"{label} (via {road_name})"
    return label


def build_model_and_graph():
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

    target = "all_motor_vehicles"
    features = [
        "year", "hour", "count_point_id", "road_type", "road_name",
        "direction_of_travel", "latitude", "longitude", "is_peak_hour",
        "flow_direction",
    ]
    X = df[features]
    y = df[target]

    high_cardinality = ["count_point_id", "road_name"]
    low_cardinality_categorical = ["road_type", "direction_of_travel", "flow_direction"]
    numeric_features = [c for c in features if c not in high_cardinality + low_cardinality_categorical]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("onehot", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), low_cardinality_categorical),
        ("target", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)),
        ]), high_cardinality),
    ])

    print("Training model on full dataset (this takes a minute or two)...")
    X_processed = np.asarray(preprocessor.fit_transform(X, y))
    model = RandomForestRegressor(**BEST_RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_processed, y)
    print("Model ready.")

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

    link_meta = subset[[
        "count_point_id", "road_name_canonical", "road_type", "latitude", "longitude",
        "direction_of_travel", "flow_direction", "start_junction_clean", "end_junction_clean",
    ]].drop_duplicates()

    graph = nx.Graph()
    for _, row in link_meta[["start_junction_clean", "end_junction_clean", "count_point_id"]].drop_duplicates().iterrows():
        graph.add_edge(row["start_junction_clean"], row["end_junction_clean"],
                        count_point_id=row["count_point_id"])

    print(f"Graph ready: {graph.number_of_nodes()} junctions, {graph.number_of_edges()} links.\n")
    return model, preprocessor, link_meta, graph


def predict_junction_traffic(junction_name, year, hour, model, preprocessor, link_meta, graph):
    if junction_name not in graph:
        raise ValueError(f"'{junction_name}' not found in the junction graph.")

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
                "year": year, "hour": hour, "count_point_id": cp,
                "road_type": r["road_type"], "road_name": r["road_name_canonical"],
                "direction_of_travel": r["direction_of_travel"],
                "latitude": r["latitude"], "longitude": r["longitude"],
                "is_peak_hour": is_peak_hour, "flow_direction": r["flow_direction"],
            })

    query_df = pd.DataFrame(rows)
    query_processed = np.asarray(preprocessor.transform(query_df))
    query_df["predicted_traffic"] = model.predict(query_processed)

    breakdown = query_df.groupby(
        ["count_point_id", "road_name", "direction_of_travel", "flow_direction"]
    )["predicted_traffic"].sum().reset_index().sort_values("predicted_traffic", ascending=False)

    return breakdown, query_df["predicted_traffic"].sum()


def list_junctions(graph, filter_text=None):
    names = sorted(graph.nodes())
    if filter_text:
        names = [n for n in names if filter_text.lower() in n.lower()]
    return names


def print_help():
    print(
        "\nCommands:\n"
        "  <junction name>       predict traffic for that junction (you'll be asked for year/hour)\n"
        "  list                  show all available junction names\n"
        "  list <text>           show junction names containing <text>\n"
        "  help                  show this message\n"
        "  quit / exit           close the tool\n"
    )


def main():
    model, preprocessor, link_meta, graph = build_model_and_graph()
    print("Junction Traffic Predictor — Birmingham Major Road Network")
    print(f"{graph.number_of_nodes()} junctions available across the major-road subset of the data.")
    print_help()

    while True:
        try:
            command = input("junction/command> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if command.lower() == "help":
            print_help()
            continue
        if command.lower().startswith("list"):
            parts = command.split(maxsplit=1)
            filter_text = parts[1] if len(parts) > 1 else None
            names = list_junctions(graph, filter_text)
            if not names:
                print("No matching junctions found.")
            else:
                print(f"\n{len(names)} junction(s):")
                for n in names:
                    print(f"  {n}")
                print()
            continue

        junction_name = command
        if junction_name not in graph:
            close_matches = list_junctions(graph, junction_name)
            print(f"'{junction_name}' not found.")
            if close_matches:
                print("Did you mean one of these?")
                for n in close_matches[:10]:
                    print(f"  {n}")
            else:
                print("Try 'list' to see all available junction names.")
            continue

        try:
            year_input = input("  Year (e.g. 2025): ").strip()
            hour_input = input("  Hour, 0-23 (e.g. 8): ").strip()
            year = int(year_input)
            hour = int(hour_input)
            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 0 and 23.")
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

        breakdown, total = predict_junction_traffic(
            junction_name, year, hour, model, preprocessor, link_meta, graph
        )
        print(f"\n=== Predicted traffic at '{junction_name}', year={year}, hour={hour}:00 ===")
        print(breakdown.to_string(index=False))
        print(f"\nTOTAL predicted traffic across all connected links: {total:.1f}\n")


if __name__ == "__main__":
    main()