Data Dictionary

Dataset Information

Dataset: DfT Birmingham Raw Traffic Counts

Source: Department for Transport Road Traffic Statistics

Project: Smart Birmingham Traffic Intelligence

Target Variable: all_motor_vehicles

Problem Type: Regression

Project Focus: Predicting hourly directional traffic flow at Birmingham traffic count points using explainable computational intelligence.

⸻

Column Definitions

Column	Description	Data Type	Feature Candidate	Notes
count_point_id	Unique identifier for traffic count location	Integer	Yes	Useful location feature
direction_of_travel	Direction vehicles are travelling (N, S, E, W etc.)	Categorical	Yes	Important predictor
year	Year of traffic count	Integer	Yes	Temporal feature
count_date	Date traffic was recorded	Date	Yes	Can generate month, season, weekday
hour	Hour of day traffic was recorded	Integer	Yes	Major predictor
region_id	Region identifier	Integer	No	Constant for Birmingham dataset
region_name	Region name	String	No	Constant value
local_authority_id	Local authority identifier	Integer	No	Constant value
local_authority_name	Local authority name	String	No	Constant value
road_name	Road identifier (e.g. A34)	Categorical	Yes	Strong location feature
road_type	Road classification	Categorical	Yes	Major, Minor, Motorway etc.
start_junction_road_name	Road at start junction	Categorical	Yes	Useful network feature
end_junction_road_name	Road at end junction	Categorical	Yes	Useful network feature
easting	British National Grid easting coordinate	Numeric	Yes	Spatial feature
northing	British National Grid northing coordinate	Numeric	Yes	Spatial feature
latitude	Geographic latitude	Numeric	Yes	Spatial feature
longitude	Geographic longitude	Numeric	Yes	Spatial feature
link_length_km	Road segment length in kilometres	Numeric	Yes	Road characteristic
link_length_miles	Road segment length in miles	Numeric	No	Duplicate of km value
pedal_cycles	Number of pedal cycles counted	Numeric	Consider	May indicate wider transport activity
two_wheeled_motor_vehicles	Motorcycle count	Numeric	No	Potential leakage
cars_and_taxis	Cars and taxis count	Numeric	No	Contributes directly to target
buses_and_coaches	Bus and coach count	Numeric	No	Contributes directly to target
lgvs	Light goods vehicles count	Numeric	No	Contributes directly to target
hgvs_2_rigid_axle	HGV count	Numeric	No	Contributes directly to target
hgvs_3_rigid_axle	HGV count	Numeric	No	Contributes directly to target
hgvs_4_or_more_rigid_axle	HGV count	Numeric	No	Contributes directly to target
hgvs_3_or_4_articulated_axle	HGV count	Numeric	No	Contributes directly to target
hgvs_5_articulated_axle	HGV count	No	Contributes directly to target	
hgvs_6_articulated_axle	HGV count	No	Contributes directly to target	
all_hgvs	Total heavy goods vehicles	Numeric	No	Contributes directly to target
all_motor_vehicles	Total motor vehicles recorded	Numeric	Target	Prediction target

⸻

Engineered Features

The following features will be generated during preprocessing:

Feature	Description
is_morning_peak	1 if hour is between 07:00 and 09:00
is_evening_peak	1 if hour is between 16:00 and 18:00
is_peak_hour	Combined rush hour indicator
day_of_week	Derived from count_date
month	Derived from count_date
season	Derived from count_date
road_pair	Combination of start and end junction roads
is_weekend	Indicates Saturday or Sunday

⸻

Modelling Strategy

Target Variable

all_motor_vehicles

Feature Categories

Temporal Features

* year
* hour
* day_of_week
* month
* season
* is_weekend

Spatial Features

* count_point_id
* latitude
* longitude
* easting
* northing

Road Network Features

* road_name
* road_type
* start_junction_road_name
* end_junction_road_name
* road_pair
* direction_of_travel
* link_length_km

Excluded Features

Vehicle category counts are excluded because they contribute directly to all_motor_vehicles and would introduce data leakage.

Excluded columns:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs
* hgvs_* columns
* two_wheeled_motor_vehicles

⸻

Target Prediction Statement

The model predicts the number of motor vehicles travelling past a Birmingham traffic count point during a given hour using temporal, spatial and road network characteristics.