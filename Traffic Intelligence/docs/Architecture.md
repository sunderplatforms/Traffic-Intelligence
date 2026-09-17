Architecture

System Overview

Smart Birmingham Traffic Intelligence is a set of Python scripts that model and predict traffic flow across Birmingham's road network using Genetic Programming and conventional machine learning, and extend that model to a junction-level traffic graph.

This describes the architecture as actually implemented: standalone scripts run from the command line, sharing a common module, reading a local CSV and writing charts/CSVs to `outputs/`. There is no running service, API, or web dashboard — see "Not currently implemented" below.

Pipeline

```
Raw DfT CSV
      ↓
traffic_common.py            shared paths, feature engineering, encoding
      ↓                       pipeline, junction-graph cleaning
      ├── traffic_flow_prediction_v2.py
      │       Cleaning → Feature Engineering → Grouped Train/Test Split
      │       → Encoding → Baseline Models → Hyperparameter Tuning
      │       → Genetic Programming (parsimony sweep + robustness check)
      │       → Evaluation → Feature Importance (+ location ablation)
      │       → charts/CSVs in outputs/
      │
      └── Junction-Level Extension (subset of rows with junction endpoints)
              ├── check_generic_junctions.py   diagnostic: flags ambiguous
              │                                junction labels before the
              │                                graph is built
              ├── junction_network.py          builds the junction graph,
              │                                historical-average baseline
              ├── junction_predictor.py        same graph + a trained RF
              │                                model, predicts for any
              │                                year/hour
              └── junction_cli.py              interactive front-end over
                                                junction_predictor.py
```

Components

`traffic_common.py`

Shared module imported by every other script. Responsible for:

* Resolving `DATA_PATH` / `OUTPUT_DIR` relative to the repo, not a hardcoded machine path
* `add_engineered_features()` — peak-hour indicators and the Birmingham-relative `flow_direction` feature
* `build_preprocessor()` — the target-encoding / one-hot ColumnTransformer used by every model-training script
* Junction/road-name cleaning shared by the three junction scripts (`canonical_road_name`, `disambiguate_junction`, `load_junction_subset`, `build_junction_graph`) and the shared `predict_junction_traffic()` function

`traffic_flow_prediction_v2.py`

The main modelling pipeline: data cleaning, feature engineering, a grouped train/test split, encoding, baseline model comparison, hyperparameter tuning, the Genetic Programming symbolic regression run (including the parsimony sweep and multi-seed robustness check), evaluation (holdout + GroupKFold CV), and feature importance (impurity, permutation, and a location-excluded ablation). Writes all charts/CSVs to `outputs/`.

Junction-level extension

Four scripts building on the subset of rows with populated junction endpoints (see README → Extension: Junction-Level Traffic Prediction):

* `check_generic_junctions.py` — flags junction labels that are geographically ambiguous, so a data-quality issue is caught before the graph is built rather than discovered by inspecting the output
* `junction_network.py` — builds the junction graph and a historical-average baseline per link
* `junction_predictor.py` — the same graph, but predicting with the trained Random Forest model for any chosen year/hour instead of a historical average
* `junction_cli.py` — an interactive command-line front-end over `junction_predictor.py`

Diagnostic / one-off scripts

`check_junctions.py` and `check_m6_node.py` are small, one-off checks used while investigating the junction data (field coverage, and verifying the M6 case specifically) — not part of the regular pipeline.

⸻

Not currently implemented

An earlier planning pass sketched a layered platform (a data-ingestion/versioning layer, a prediction API, and a web dashboard) as a longer-term vision for this project. None of that exists yet — the current implementation is the script-based pipeline described above, run locally and read via its charts/CSVs and the interactive CLI. Turning the trained model into a served API and a dashboard is listed as genuine future work, not a component to imply already exists; see README → Future Work.
