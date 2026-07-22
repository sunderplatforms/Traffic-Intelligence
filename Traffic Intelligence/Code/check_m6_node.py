import pandas as pd

DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

has_both = df["start_junction_road_name"].notna() & df["end_junction_road_name"].notna()
subset = df[has_both].copy()

touches_m6 = (
    (subset["start_junction_road_name"].str.strip() == "M6")
    | (subset["end_junction_road_name"].str.strip() == "M6")
)
m6_rows = subset[touches_m6]

print(f"Rows where a junction endpoint is exactly 'M6': {len(m6_rows)}")
print("\nWhich roads reference plain 'M6' as a junction endpoint:")
print(m6_rows["road_name"].value_counts())

print("\nDistinct count points involved:")
print(m6_rows[["count_point_id", "road_name", "start_junction_road_name",
               "end_junction_road_name", "latitude", "longitude"]].drop_duplicates())
