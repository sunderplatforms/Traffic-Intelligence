Feature Engineering

Purpose

Feature engineering transforms raw traffic data into features that improve model performance while maintaining explainability.

Raw Features

Temporal Features

* year
* count_date
* hour

Road Network Features

* road_name
* road_type
* start_junction_road_name
* end_junction_road_name
* direction_of_travel

Spatial Features

* count_point_id
* latitude
* longitude
* easting
* northing

Road Characteristics

* link_length_km

Engineered Features

Peak Hour Indicators

is_morning_peak

True when hour is between:

* 07:00
* 08:00
* 09:00

is_evening_peak

True when hour is between:

* 16:00
* 17:00
* 18:00

is_peak_hour

Combined rush-hour indicator.

Date Features

Derived from count_date:

* day_of_week
* month
* quarter
* season
* is_weekend

Road Features

road_pair

Combines:

* start_junction_road_name
* end_junction_road_name

Example:

A4400/A38_to_A4540

Excluded Features

The following features are excluded due to data leakage risk:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs
* hgvs_* columns
* two_wheeled_motor_vehicles

These variables directly contribute to all_motor_vehicles and would artificially inflate model performance.