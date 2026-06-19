# Exploratory Data Analysis Findings

## Dataset Overview

| Metric | Value |
|----------|----------|
| Rows | 72,948 |
| Columns | 32 |
| Years Covered | 2000–2025 |
| Count Points | 562 |
| Unique Roads | 50 |
| Target Variable | all_motor_vehicles |

The dataset contains Birmingham road traffic count observations collected by the Department for Transport. Each observation represents traffic recorded at a specific count point, direction and hour.

---

## Missing Data Analysis

The following fields contain substantial missing values:

| Column | Missing Values |
|----------|----------|
| start_junction_road_name | 49,680 |
| end_junction_road_name | 49,680 |
| link_length_km | 49,680 |
| link_length_miles | 49,680 |

These missing values are concentrated on minor roads and unclassified roads and do not represent random missing data.

---

## Traffic by Road Type

| Road Type | Mean Traffic |
|------------|------------:|
| Major | 1304 |
| Minor | 177 |

### Finding

Major roads carry significantly more traffic than minor roads.

The average traffic volume on major roads is more than seven times greater than on minor roads.

This suggests road classification is an important predictor of traffic flow.

---

## Traffic by Direction

| Direction | Mean Traffic |
|------------|------------:|
| West | 560 |
| East | 548 |
| North | 527 |
| South | 522 |

### Finding

Traffic flow is relatively balanced across directions.

Westbound traffic records the highest average flow, while southbound traffic records the lowest.

---

## Traffic by Hour

Peak traffic occurs during:

* 08:00
* 16:00
* 17:00

### Finding

The dataset clearly demonstrates morning and evening commuting behaviour.

This supports the inclusion of engineered peak-hour features.

---

## Highest Traffic Roads

| Road | Mean Traffic |
|--------|------------:|
| M6 | 4062 |
| A38(M) | 3383 |
| A38M | 3291 |
| A4400 | 1671 |
| A38 | 1527 |

### Finding

Motorways and major arterial routes dominate Birmingham traffic movement.

The M6 exhibits substantially higher traffic flow than all other roads.

---

## Machine Learning Baseline Results

Random Forest achieved:

| Metric | Value |
|----------|----------|
| MAE | 50.44 |
| RMSE | 117.02 |
| R² | 0.9779 |

### Finding

The model explains approximately 97.8% of the variance in traffic flow.

This indicates that the selected features contain strong predictive information.

---

## Feature Importance Findings

Top predictive features include:

1. Road Type
2. Road Name
3. Count Point Location
4. Year
5. Hour
6. Direction of Travel

### Finding

Traffic flow is primarily influenced by infrastructure characteristics and location rather than direction alone.

---

## Summary

The exploratory analysis demonstrates that:

* Major roads experience substantially higher traffic volumes.
* Traffic follows expected commuting patterns.
* Certain roads dominate Birmingham traffic movement.
* Road classification is a strong predictor of traffic flow.
* The dataset contains sufficient structure for predictive modelling.

These findings justify the use of machine learning and Genetic Programming for explainable traffic-flow prediction.