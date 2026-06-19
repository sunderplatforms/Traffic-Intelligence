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
4. Human behaviour patterns

All engineered features in this project map to one of these categories.

⸻

1. Temporal Features (Time-Based Behaviour)

Traffic changes predictably across time.

Hour of Day

The most important predictor of traffic volume.

Justification:

* Captures daily commuting patterns
* Reflects rush hour congestion

Expected pattern:

* High: 07:00–09:00, 16:00–18:00
* Low: late night hours

⸻

Year

Justification:

* Captures long-term growth or decline in traffic
* Accounts for infrastructure changes over time

⸻

Peak Hour Indicators

is_morning_peak
is_evening_peak
is_peak_hour

Justification:

* Traffic is not linear across hours
* Peak hours behave differently from normal hours

This improves model interpretability.

⸻

2. Spatial Features (Location-Based Behaviour)

Traffic varies heavily by geography.

Latitude / Longitude

Justification:

* Captures spatial clustering of traffic
* Allows model to learn regional traffic density

⸻

count_point_id

Justification:

* Unique measurement location
* Represents fixed monitoring stations

⸻

3. Road Network Features

Traffic is strongly dependent on road structure.

road_name

Justification:

* Different roads carry different volumes
* Major roads (A-roads) have higher flow

⸻

road_type

Justification:

* Differentiates between major and minor roads

⸻

road_pair (engineered)

Justification:

* Captures direction of flow between two junctions
* Represents traffic movement, not just location

⸻

direction_of_travel

Justification:

* Traffic differs depending on inbound/outbound flow

⸻

4. Road Geometry Features

link_length_km

Justification:

* Longer links may represent highways or major routes
* Correlates with traffic capacity

⸻

5. Excluded Features (Data Leakage Prevention)

The following are removed to prevent the model from cheating:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs
* all_vehicle category breakdowns

Reason:
These directly sum into all_motor_vehicles and would make prediction trivial.

⸻

6. Target Variable

all_motor_vehicles

Definition:
Total number of motor vehicles recorded at a traffic count point for a given hour and direction.

This is treated as a regression target.

⸻

Summary

This feature design ensures:

* Strong predictive power
* No data leakage
* Interpretability of results
* Alignment with real traffic systems