"""
Junction Flow Network — Birmingham Major Road Network
========================================================
Builds a graph of Birmingham's classified-road junctions from the subset
of DfT count points that have both start_junction_road_name and
end_junction_road_name populated (~32% of the full dataset, 170 distinct
links / 172 near-unique link definitions across 23 major roads).

Nodes  = named junctions (identified by the cross-road name in the data,
         e.g. "A4400/A38", "LA Boundary")
Edges  = links between junctions (one per count_point_id), weighted by
         mean historical all_motor_vehicles traffic on that link

This is a SCHEMATIC network, not a geographically precise map — junction
node positions are laid out with a force-directed algorithm, not real
coordinates, since the data identifies junctions by name rather than by
lat/long. This script establishes the baseline (historical average) flow
network; junction_predictor.py swaps in model *predictions* for a chosen
year/hour instead of historical averages.

The graph is a MultiGraph rather than a plain Graph because distinct roads
can share the same two junction endpoints (e.g. both A38 and A4400 link
"A456/A457 roundabout" to "B4100") — a plain Graph would silently drop one
of the two links. See traffic_common.build_junction_graph.

Run:
    pip install networkx
    python3 junction_network.py
"""

import matplotlib.pyplot as plt
import networkx as nx

from traffic_common import (
    DATA_PATH, OUTPUT_DIR, is_generic_junction_label, load_junction_subset,
)
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load and filter to the junction-defined subset (cleaning — road-name
#    variant merging, the "rooundabout" typo fix, and generic-label
#    disambiguation — happens inside load_junction_subset)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
print(f"Junction-defined subset: {has_both.sum()} rows, "
      f"{df.loc[has_both, 'count_point_id'].nunique()} distinct count points")

subset = load_junction_subset(df)

subset["link_id"] = (
    subset["road_name_canonical"] + " [" + subset["start_junction_clean"]
    + " -> " + subset["end_junction_clean"] + "]"
)

# ---------------------------------------------------------------------------
# 1b. Reporting: how much did the cleaning actually change?
# ---------------------------------------------------------------------------
n_variants_merged = subset["road_name"].nunique() - subset["road_name_canonical"].nunique()
if n_variants_merged > 0:
    merged_examples = (
        subset[["road_name", "road_name_canonical"]]
        .drop_duplicates()
        .groupby("road_name_canonical")["road_name"]
        .apply(list)
    )
    merged_examples = merged_examples[merged_examples.apply(len) > 1]
    print(f"\nMerged {n_variants_merged} inconsistent road name variant(s):")
    print(merged_examples)

known_road_names_canonical = set(subset["road_name_canonical"].unique())
start_generic = subset["start_junction_road_name"].apply(
    lambda label: is_generic_junction_label(label, known_road_names_canonical))
end_generic = subset["end_junction_road_name"].apply(
    lambda label: is_generic_junction_label(label, known_road_names_canonical))
n_generic_disambiguated = start_generic.sum() + end_generic.sum()

generic_labels_found = sorted(set(
    subset.loc[start_generic, "start_junction_road_name"].str.strip().tolist()
    + subset.loc[end_generic, "end_junction_road_name"].str.strip().tolist()
))
print(f"\nDisambiguated {n_generic_disambiguated} rows across {len(generic_labels_found)} "
      f"generic junction label(s) into road-specific nodes:")
print(generic_labels_found)

# ---------------------------------------------------------------------------
# 2. Aggregate to one row per link: mean historical traffic, both directions
#    combined (a link's total volume, not split by direction of travel)
# ---------------------------------------------------------------------------
link_summary = subset.groupby(
    ["link_id", "road_name_canonical", "start_junction_clean", "end_junction_clean"]
).agg(
    mean_traffic=("all_motor_vehicles", "mean"),
    total_observations=("all_motor_vehicles", "count"),
).reset_index().rename(columns={
    "road_name_canonical": "road_name",
    "start_junction_clean": "start_junction_road_name",
    "end_junction_clean": "end_junction_road_name",
})

print(f"\nBuilt {len(link_summary)} link summaries.")
print(link_summary.sort_values("mean_traffic", ascending=False).head(10))

link_summary.to_csv(OUTPUT_DIR / "junction_link_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Build the graph
# ---------------------------------------------------------------------------
G = nx.MultiGraph()

for _, row in link_summary.iterrows():
    G.add_edge(
        row["start_junction_road_name"], row["end_junction_road_name"],
        weight=row["mean_traffic"],
        road_name=row["road_name"],
        link_id=row["link_id"],
    )

print(f"\nGraph built: {G.number_of_nodes()} junction nodes, {G.number_of_edges()} links")

# ---------------------------------------------------------------------------
# 4. Per-junction total connected traffic (sum of all links touching a
#    node — weighted degree correctly sums across parallel edges too)
# ---------------------------------------------------------------------------
junction_totals = dict(G.degree(weight="weight"))

junction_df = pd.DataFrame(
    sorted(junction_totals.items(), key=lambda x: -x[1]),
    columns=["junction", "total_connected_traffic"],
)
print("\nTop 15 junctions by total connected traffic:")
print(junction_df.head(15))
junction_df.to_csv(OUTPUT_DIR / "junction_totals.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Visualize the network
# ---------------------------------------------------------------------------
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, k=0.6)

node_sizes = [max(junction_totals[n], 1) / 3 for n in G.nodes()]
edges_data = list(G.edges(data=True))
edge_weights = [d["weight"] for _, _, d in edges_data]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [1 + 6 * (w / max_weight) for w in edge_weights]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="tab:blue", alpha=0.7)
nx.draw_networkx_edges(
    G, pos, edgelist=[(u, v) for u, v, _ in edges_data],
    width=edge_widths, edge_color="tab:gray", alpha=0.6,
)
nx.draw_networkx_labels(G, pos, font_size=7)

plt.title("Birmingham Major Road Junction Network\n"
          "(node size = total connected traffic, edge width = link traffic — historical averages)")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "junction_network.png", dpi=300)
plt.close()

print(f"\nSaved network diagram to {OUTPUT_DIR / 'junction_network.png'}")
print("Saved link summary to outputs/junction_link_summary.csv")
print("Saved junction totals to outputs/junction_totals.csv")
