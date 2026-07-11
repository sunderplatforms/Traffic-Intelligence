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
network; a follow-up step can swap in model *predictions* for a chosen
year/hour instead of historical averages.

Run:
    pip install networkx
    python3 junction_network.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and filter to the junction-defined subset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
subset = df[has_both].copy()
print(f"Junction-defined subset: {len(subset)} rows, "
      f"{subset['count_point_id'].nunique()} distinct count points")

subset["link_id"] = (
    subset["road_name"] + " [" + subset["start_junction_road_name"]
    + " -> " + subset["end_junction_road_name"] + "]"
)

# ---------------------------------------------------------------------------
# 1b. Data cleaning: fix two real issues found by inspecting the built graph
# ---------------------------------------------------------------------------
# Issue A: the same road can appear under inconsistent name variants
# (e.g. "A38M" vs "A38(M)" for the Aston Expressway), which would otherwise
# fragment one physical road into two separate identities in the graph.
def canonical_road_name(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


canonical_lookup = {}
for raw_name in subset["road_name"].unique():
    canonical_lookup.setdefault(canonical_road_name(raw_name), raw_name)
subset["road_name_canonical"] = subset["road_name"].apply(
    lambda n: canonical_lookup[canonical_road_name(n)]
)

# Issue B: "LA Boundary" (and similar generic labels) is not a single
# physical location — it means "this road crosses the local authority
# boundary", which happens at a different physical point for every road.
# Treating it as one graph node falsely connects unrelated roads. Instead,
# qualify it with the road it belongs to, so each road's boundary crossing
# becomes its own distinct node.
# Issue B (generalized): a junction label is only meaningful if it names a
# SPECIFIC location. "LA Boundary" was the first example found — a generic
# label meaning "this road crosses the boundary somewhere" rather than one
# point. A systematic check (see check_generic_junctions.py) found this is
# actually pervasive: Birmingham's ring road (A4040) and several major
# through-roads (A38, A34, A41, etc.) are crossed by many different roads
# at DIFFERENT points, but DfT's junction label just names the crossing
# road ("meets the A4040") without saying where — so ANY label that is
# itself just a bare road name is ambiguous, not only "LA Boundary".
# Fix: qualify any such label with the road that is using it as an
# endpoint, so each real crossing becomes its own distinct node.
GENERIC_JUNCTION_LABELS = {"la boundary"}

known_road_names_canonical = set(subset["road_name_canonical"].unique())


def disambiguate_junction(junction_name, road_name):
    label = str(junction_name).strip()
    is_generic = (
        canonical_road_name(label) in known_road_names_canonical
        or label.lower() in GENERIC_JUNCTION_LABELS
    )
    if is_generic:
        return f"{label} (via {road_name})"
    return label


subset["start_junction_clean"] = [
    disambiguate_junction(j, r) for j, r in zip(subset["start_junction_road_name"],
                                                  subset["road_name_canonical"])
]
subset["end_junction_clean"] = [
    disambiguate_junction(j, r) for j, r in zip(subset["end_junction_road_name"],
                                                  subset["road_name_canonical"])
]

subset["link_id"] = (
    subset["road_name_canonical"] + " [" + subset["start_junction_clean"]
    + " -> " + subset["end_junction_clean"] + "]"
)

# ---------------------------------------------------------------------------
# 2. Aggregate to one row per link: mean historical traffic, both directions
#    combined (a link's total volume, not split by direction of travel)
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

def is_generic_label(label):
    label = str(label).strip()
    return (
        canonical_road_name(label) in known_road_names_canonical
        or label.lower() in GENERIC_JUNCTION_LABELS
    )


start_generic = subset["start_junction_road_name"].apply(is_generic_label)
end_generic = subset["end_junction_road_name"].apply(is_generic_label)
n_generic_disambiguated = start_generic.sum() + end_generic.sum()

generic_labels_found = sorted(set(
    subset.loc[start_generic, "start_junction_road_name"].str.strip().tolist()
    + subset.loc[end_generic, "end_junction_road_name"].str.strip().tolist()
))
print(f"\nDisambiguated {n_generic_disambiguated} rows across {len(generic_labels_found)} "
      f"generic junction label(s) into road-specific nodes:")
print(generic_labels_found)

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
G = nx.Graph()

for _, row in link_summary.iterrows():
    start_node = row["start_junction_road_name"]
    end_node = row["end_junction_road_name"]
    G.add_edge(
        start_node, end_node,
        weight=row["mean_traffic"],
        road_name=row["road_name"],
        link_id=row["link_id"],
    )

print(f"\nGraph built: {G.number_of_nodes()} junction nodes, {G.number_of_edges()} links")

# ---------------------------------------------------------------------------
# 4. Per-junction total connected traffic (sum of all links touching a node)
# ---------------------------------------------------------------------------
junction_totals = {}
for node in G.nodes():
    total = sum(G[node][neighbor]["weight"] for neighbor in G.neighbors(node))
    junction_totals[node] = total

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
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [1 + 6 * (w / max_weight) for w in edge_weights]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="tab:blue", alpha=0.7)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="tab:gray", alpha=0.6)
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