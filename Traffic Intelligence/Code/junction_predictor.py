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
    already found via tuning in the main pipeline (traffic_flow_prediction_v2.py).
    Re-running that tuning here would be redundant and slow.

Run:
    python3 junction_predictor.py
"""

import pandas as pd

from traffic_common import (
    DATA_PATH, OUTPUT_DIR, add_engineered_features, build_junction_graph,
    build_link_meta, load_junction_subset, predict_junction_traffic,
    train_full_model,
)

# Best hyperparameters found via RandomizedSearchCV in traffic_flow_prediction_v2.py
# (re-tuning here would be redundant — reuse the already-validated result)
BEST_RF_PARAMS = {
    "n_estimators": 400,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 20,
}

# ---------------------------------------------------------------------------
# 1. Load data and engineer features
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["count_point_id"] = df["count_point_id"].astype(str)
add_engineered_features(df)

# ---------------------------------------------------------------------------
# 2. Train the model on the FULL dataset (this is a prediction tool, not an
#    evaluation exercise — evaluation/tuning already done in the main pipeline)
# ---------------------------------------------------------------------------
print("Fitting preprocessor and model on full dataset...")
model, preprocessor = train_full_model(df, BEST_RF_PARAMS)
print("Model trained.")

# ---------------------------------------------------------------------------
# 3. Build the junction graph (same cleaning as junction_network.py)
# ---------------------------------------------------------------------------
subset = load_junction_subset(df)
link_meta = build_link_meta(subset)
G = build_junction_graph(subset)

print(f"Graph built: {G.number_of_nodes()} junction nodes, {G.number_of_edges()} links")

# ---------------------------------------------------------------------------
# 4. Example queries
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
