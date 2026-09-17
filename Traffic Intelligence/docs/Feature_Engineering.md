Feature Engineering

Purpose

Feature engineering transforms raw traffic data into features that improve model performance while maintaining explainability.

Raw Features Used

Temporal

* year
* hour

Road Network

* road_name
* road_type
* direction_of_travel

Spatial

* count_point_id
* latitude
* longitude

Engineered Features

Peak Hour Indicators

* is_morning_peak — true for hour in {7, 8, 9}
* is_evening_peak — true for hour in {16, 17, 18}
* is_peak_hour — true for hour in {7, 8, 9, 16, 17, 18}. This is the one actually passed into the models; the morning/evening split exists for EDA (see `hourly_traffic_by_flow_direction.png`) rather than as a separate model input.

Directional flow_direction (Birmingham-relative)

`direction_of_travel` on its own is only a compass direction — it doesn't say whether that direction means "into Birmingham" or "out of it", since that depends on where the count point sits relative to the city centre. `flow_direction` combines direction with location:

1. Compute the bearing from Birmingham's centre (52.4796, -1.9026) to the count point, bucketed to the nearest compass cardinal (N/E/S/W).
2. Compare that bearing to the recorded `direction_of_travel`.
3. Classify as:
   - `inbound` — travelling roughly opposite the centre→point bearing (toward the centre)
   - `outbound` — travelling roughly along the centre→point bearing (away from the centre)
   - `lateral` — travelling roughly perpendicular to that bearing (a ring-road-type movement)

Implemented once in `traffic_common.add_engineered_features()` and shared by every script that trains or queries a model.

⸻

Encoding

* **road_type**, **direction_of_travel**, **flow_direction** — low cardinality (2–4 distinct values each) → one-hot encoded.
* **count_point_id** (~600 distinct values), **road_name** (~50 distinct values) → **target encoded**, not one-hot. An early version one-hot encoded these, producing 600+ columns; under grouped cross-validation, every dummy column is zero for a location never seen in training, which made Linear Regression extrapolate wildly (RMSE ≈ 3,600, R² ≈ −25.6). Target encoding (scikit-learn's `TargetEncoder`, which cross-fits internally so it doesn't leak the target into itself) fixed this and cut the feature space from 623 columns to 16. See README Key Finding 2.

⸻

Excluded Features (Data Leakage Prevention)

The following are excluded from modelling because they directly sum into `all_motor_vehicles`:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs
* hgvs_* columns
* two_wheeled_motor_vehicles

Including any of these would let the model reconstruct the target arithmetically instead of learning genuine traffic patterns.

⸻

Not implemented

An earlier planning pass considered a richer set of date-derived features (`day_of_week`, `month`, `quarter`, `season`, `is_weekend`) and a `road_pair` feature combining junction endpoints. These were not carried into the final pipeline — the junction endpoints are instead used directly by the separate junction-network extension (see README → Extension: Junction-Level Traffic Prediction), and `year`/`hour` already capture the temporal signal the model uses. Recorded here so this doc doesn't imply features that don't exist in `traffic_flow_prediction_v2.py`.
