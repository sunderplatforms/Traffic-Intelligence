Feature Analysis

Overview

Feature importance was computed two ways on the tuned Random Forest model: **impurity-based** (built into scikit-learn, but known to be biased toward high-cardinality features) and **permutation-based** (refits nothing, but re-measures RMSE after shuffling each feature — more reliable, though slower). Numbers below are read directly from `outputs/feature_importance_impurity.csv` and `outputs/feature_importance_permutation.csv`.

Because `count_point_id` and `road_name` are **target encoded** (one numeric column each, not one-hot dummies per road), "feature importance" here is importance of the encoded column, not of any individual road or location — see `Feature_Engineering.md`.

⸻

Full Model — Top Features

| Feature | Impurity | Permutation |
|---|---:|---:|
| count_point_id (target-encoded) | 40.9% | 65.1% |
| road_name (target-encoded) | 22.3% | 12.3% |
| road_type = Major (one-hot) | 11.3% | 3.1% |
| road_type = Minor (one-hot) | 9.5% | 2.4% |
| longitude | 6.7% | 2.3% |
| latitude | 3.4% | 2.4% |
| hour | 2.0% | 4.6% |

### Finding

Both methods agree: the target-encoded `count_point_id` column dominates, by an even larger margin under permutation importance (65.1%) than impurity (40.9%). This is expected — `count_point_id`'s target-encoded value is close to that location's historical average traffic, so it's an extremely strong (almost circular) predictor. On its own this doesn't tell us *why* traffic varies — see the ablation below.

⸻

Ablation: Random Forest Without `count_point_id`

Refitting without the location column (README Key Finding 4) tells a more precise story:

| Feature | Impurity | Permutation |
|---|---:|---:|
| road_name (target-encoded) | 38.5% | 58.6% |
| longitude | 15.6% | 15.4% |
| road_type = Minor (one-hot) | 14.3% | 7.6% |
| road_type = Major (one-hot) | 13.3% | 6.6% |
| latitude | 9.3% | 14.1% |
| year | 3.6% | 4.6% |
| hour | 2.2% | 4.6% |

Accuracy barely drops at all (R² 0.974 → 0.970, see `Model_Results.md`) once `count_point_id` is removed, and `road_name` immediately takes over as the dominant feature, followed by geographic coordinates. This means almost all of what `count_point_id` appeared to contribute was **recoverable from road identity and coordinates alone** — it was acting as a fine-grained proxy for information already present elsewhere, not contributing large amounts of genuinely new signal.

⸻

Road Identity and Road Type

Road identity (`road_name`) and classification (`road_type`) are consistently the strongest non-location predictors in both the full model and the ablation. This matches the EDA finding that a handful of roads — led by the M6 — carry disproportionately high traffic, and that major roads carry roughly 7x the traffic of minor roads on average (`EDA_Findings.md`).

⸻

Location Coordinates

`latitude` and `longitude` matter more once `count_point_id` is removed (their importance roughly doubles or more), consistent with them being a coarser, complementary source of the same geographic signal that `count_point_id` captures at finer granularity.

⸻

Time-Related Features

`hour` and `year` are real but comparatively minor contributors (a few percent each) in both impurity and permutation terms — time-of-day and long-term trend matter, but far less than *where* the count point is and *what kind of road* it's on.

⸻

Direction

`direction_of_travel` and the engineered `flow_direction` feature do not appear in the top features by either importance measure for this model — road identity, classification and location dominate. This doesn't mean direction is irrelevant: the README's junction-level extension finds a clear, real AM/PM directional asymmetry at specific junctions (e.g. the A41's A4540 link), it's just a smaller effect than location/road-identity at the level of a single global regression model.

⸻

Conclusion

Birmingham traffic flow is primarily driven by **where** a count point is and **what road** it's on — road identity and classification, then geography, then time. `count_point_id` alone looked dominant, but the ablation shows most of that was recoverable from road name and coordinates rather than being unique per-location signal. See README Key Finding 4 for the full discussion.
