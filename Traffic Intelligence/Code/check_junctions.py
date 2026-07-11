import pandas as pd

DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
print(f"Rows with both junction fields populated: {has_both.sum()} / {len(df)} "
      f"({has_both.mean():.1%})")

subset = df[has_both]

print(f"\nUnique start_junction_road_name values: {subset['start_junction_road_name'].nunique()}")
print(f"Unique end_junction_road_name values: {subset['end_junction_road_name'].nunique()}")
print(f"Unique road_name values (in this subset): {subset['road_name'].nunique()}")

# A candidate "link" identity: road_name + start + end junction
subset = subset.copy()
subset["link_id"] = (
    subset["road_name"] + " [" + subset["start_junction_road_name"]
    + " -> " + subset["end_junction_road_name"] + "]"
)
print(f"\nUnique (road_name, start_junction, end_junction) combinations: {subset['link_id'].nunique()}")
print(f"Unique count_point_id values (in this subset): {subset['count_point_id'].nunique()}")

print("\nSample of link definitions:")
print(subset[["count_point_id", "road_name", "start_junction_road_name",
              "end_junction_road_name", "link_length_km"]].drop_duplicates().head(15))

# Check: does each count_point_id map to exactly one link definition?
per_point = subset.groupby("count_point_id")["link_id"].nunique()
print(f"\ncount_point_ids with more than one distinct link definition: "
      f"{(per_point > 1).sum()} / {len(per_point)}")
