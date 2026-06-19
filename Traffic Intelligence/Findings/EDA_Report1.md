Initial Findings

Dataset Size

The Birmingham Raw Traffic Counts dataset contains 72,948 traffic observations collected between 2000 and 2025.

The dataset includes:

* 72,948 records
* 32 features
* 562 unique traffic count points
* 50 unique roads

The size and temporal coverage of the dataset provide sufficient data to investigate long-term traffic patterns and train machine learning models.

⸻

Traffic Flow Distribution

The target variable for this project is all_motor_vehicles.

Summary statistics:

Metric	Value
Mean	536.87
Median	262
Standard Deviation	783.31
Minimum	0
Maximum	7061

The large difference between the mean and median indicates a right-skewed distribution. Most observations contain relatively low traffic volumes, while a smaller number of major roads carry significantly higher traffic volumes.

This suggests that road characteristics and location are likely to be important predictive features.

⸻

Road Traffic Analysis

The roads with the highest average traffic volumes are:

Road	Average Vehicles
M6	4061.96
A38(M)	3382.96
A38M	3290.93
A4400	1670.52
A38	1527.02
A45	1519.84
A4540	1514.09
A456	1272.50
A34	1198.04
A4041	1114.57

The motorway network carries substantially higher traffic volumes than most Birmingham roads. The M6 records the highest average traffic volume, exceeding 4,000 vehicles per observation.

This indicates that road identity and road classification are likely to be strong predictors of traffic flow.

⸻

Initial Modelling Implications

Several variables are expected to contribute strongly to prediction performance:

* hour
* direction_of_travel
* road_name
* road_type
* count_point_id
* latitude
* longitude
* link_length_km

Vehicle category breakdown columns will be excluded from modelling because they directly contribute to the target variable and would introduce data leakage.

Examples include:

* cars_and_taxis
* buses_and_coaches
* lgvs
* all_hgvs

Removing these variables ensures the model learns traffic patterns rather than reconstructing the target value through arithmetic relationships.