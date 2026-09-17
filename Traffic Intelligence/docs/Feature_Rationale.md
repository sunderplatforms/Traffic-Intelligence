Feature Rationale

Purpose

This document explains why each feature is included in the traffic prediction model and how it relates to traffic behaviour in Birmingham.

The goal is to ensure all features are grounded in real-world traffic logic rather than arbitrary selection.

⸻

Core Idea

Traffic flow is influenced by four main factors:

1. Time
2. Location
3. Road network structure
4. Human behaviour patterns (commuting direction)

All engineered features in this project map to one of these categories.

⸻

1. Temporal Features

Hour of Day

Justification:

* Captures daily commuting patterns
* Reflects rush-hour congestion — see `EDA_Findings.md` for the measured AM/PM peaks

Year

Justification:

* Captures long-term growth or decline in traffic
* A comparatively minor contributor in feature importance (a few percent) relative to road/location features — see `Feature_Analysis.md`

Peak Hour Indicators (is_morning_peak, is_evening_peak, is_peak_hour)

Justification:

* Traffic is not linear across hours; peak hours behave differently from normal hours
* `is_peak_hour` is the one actually passed into the models

⸻

2. Spatial Features

Latitude / Longitude

Justification:

* Captures spatial clustering of traffic
* Becomes noticeably more important once `count_point_id` is excluded (ablation in `Feature_Analysis.md`) — the model falls back on coordinates as its geographic signal

count_point_id

Justification:

* Unique measurement location, target-encoded to its historical average traffic
* Dominates feature importance on its own (see `Feature_Analysis.md`), but the ablation there shows most of that signal is recoverable from `road_name` + coordinates — i.e. it's a strong but largely redundant proxy, not a uniquely informative feature

⸻

3. Road Network Features

road_name

Justification:

* Different roads carry very different volumes (the M6 vs. a residential side road)
* Target encoded (high cardinality — ~50 distinct values) rather than one-hot, for the same reason as count_point_id — see "Encoding strategy" below
* Becomes the single dominant feature once `count_point_id` is excluded

road_type

Justification:

* Differentiates major vs. minor roads — major roads carry roughly 7x the mean traffic of minor roads (`EDA_Findings.md`)
* Low cardinality (Major/Minor) → one-hot encoded

direction_of_travel

Justification:

* Raw compass direction alone is a weak signal on its own (traffic is fairly balanced N/E/S/W — see `EDA_Findings.md`)
* Its value comes from being combined with location into `flow_direction` (below), not from direction alone

flow_direction (engineered)

Justification:

* Combines `direction_of_travel` with the bearing from Birmingham's centre to the count point, to classify each observation as inbound / outbound / lateral
* Captures the project's core "traffic into and out of Birmingham" framing in a way raw compass direction cannot
* Low cardinality (3 values) → one-hot encoded

⸻

4. Encoding Strategy

Why target encoding for count_point_id / road_name, not one-hot:

One-hot encoding these high-cardinality columns (~600 and ~50 distinct values) blew the feature space up to 623 columns. Under grouped cross-validation (splitting by `count_point_id` so a location's data can't appear in both train and validation), every dummy column for a location unseen during training is zero — Linear Regression then extrapolates wildly on unseen locations (RMSE ≈ 3,600, R² ≈ −25.6). Target encoding replaces each category with a (cross-fitted, leakage-safe) estimate of its average target value, avoiding this failure mode entirely and cutting the feature space to 16 columns. See README Key Finding 2.

⸻

5. Excluded Features (Data Leakage Prevention)

The following are removed because they sum directly into `all_motor_vehicles`:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs / hgvs_* columns
* two_wheeled_motor_vehicles

Reason: including any of these would let the model reconstruct the target arithmetically rather than learning genuine traffic patterns.

⸻

6. Target Variable

all_motor_vehicles

Definition: total number of motor vehicles recorded at a traffic count point for a given hour and direction. Treated as a regression target.

⸻

Summary

This feature design aims for:

* Strong predictive power, honestly evaluated (grouped CV, not a leaky single split)
* No data leakage
* An encoding strategy that survives evaluation on genuinely unseen locations
* Alignment with real traffic systems and the project's directional "into/out of Birmingham" framing
