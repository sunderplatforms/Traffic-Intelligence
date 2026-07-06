Feature Analysis

Overview

Feature importance analysis was performed using the trained Random Forest regression model.

The purpose of this analysis was to identify which road, location, direction and time-related features had the greatest influence on predicted traffic flow in Birmingham.

Feature importance values show the relative contribution of each transformed input feature to the Random Forest model. A higher value indicates that the feature was more useful when the model made traffic-flow predictions.

⸻

Most Important Features

The most influential features included:

1. Road type
2. Road identity
3. Count point location
4. Year
5. Hour of day
6. Direction of travel
7. Peak-hour indicators

The strongest individual features were associated with whether a road was classified as a major or minor road, along with high-traffic routes such as the M6 and A38(M).

⸻

Road Type

Road type was one of the most important predictors of traffic flow.

The Random Forest model assigned high importance to both road_type_Major and road_type_Minor. This reflects the clear difference observed during exploratory data analysis:

* Major roads had substantially higher average traffic flow.
* Minor roads had lower average traffic flow.

This makes sense because major roads include strategic routes and key urban corridors that carry larger volumes of vehicles.

⸻

Road Identity

Specific road names were also important.

The M6 was one of the strongest road-level predictors. This is expected because the M6 carries much higher traffic volumes than many other roads in the Birmingham dataset.

Other important roads included:

* A38(M)
* A38
* A4540
* A4400
* A45

These roads are important transport corridors within or around Birmingham and are likely to experience consistently higher traffic levels.

⸻

Location and Count Points

count_point_id, latitude and longitude were important features.

These variables allow the model to distinguish between different monitoring locations. Traffic flow varies significantly between locations because roads differ in their role, capacity, surrounding land use and connection to the wider road network.

The importance of location features suggests that traffic prediction should consider where a count point is located, not only the time or direction of travel.

⸻

Time-Related Features

Year and hour were also influential.

The hour feature reflects variation across the day. Exploratory analysis showed that traffic levels were generally higher during morning and afternoon commuting periods.

The year feature may capture longer-term changes in traffic patterns, road use, transport behaviour and changes in the available count data over time.

Peak-hour indicator features contributed less than the raw hour value. This suggests that the model benefits from knowing the exact hour rather than only whether a time falls within a broad peak period.

⸻

Direction of Travel

Direction of travel was a useful but less influential feature compared with road type and road identity.

The model used directional features such as northbound, southbound, eastbound and westbound movement to improve predictions. This supports the project focus on modelling directional traffic flow into and out of Birmingham.

However, the lower importance compared with road type suggests that the characteristics of the road and count-point location have a greater overall effect on traffic volume.

⸻

Conclusion

The feature analysis shows that Birmingham traffic flow is primarily influenced by road classification, road identity and monitoring location.

Time of day, year and direction of travel also contribute to prediction accuracy, but their impact is smaller than the structural characteristics of the road network.

These findings support the use of an explainable traffic-intelligence system, as they identify the main factors associated with traffic flow across Birmingham.