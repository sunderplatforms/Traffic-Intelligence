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

import pandas as pd

from traffic_common import (
    DATA_PATH, add_engineered_features, build_junction_graph, build_link_meta,
    load_junction_subset, predict_junction_traffic, train_full_model,
)

BEST_RF_PARAMS = {
    "n_estimators": 400,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 20,
}


def build_model_and_graph():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["count_point_id"] = df["count_point_id"].astype(str)
    add_engineered_features(df)

    print("Training model on full dataset (this takes a minute or two)...")
    model, preprocessor = train_full_model(df, BEST_RF_PARAMS)
    print("Model ready.")

    subset = load_junction_subset(df)
    link_meta = build_link_meta(subset)
    graph = build_junction_graph(subset)

    print(f"Graph ready: {graph.number_of_nodes()} junctions, {graph.number_of_edges()} links.\n")
    return model, preprocessor, link_meta, graph


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
            year_input = input("  Year (e.g. 2026): ").strip()
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
