# Exploratory Data Analysis Report

## Smart Birmingham Traffic Intelligence

### Dataset

Source:
Department for Transport Road Traffic Statistics

Dataset:
Birmingham Raw Traffic Counts

Target Variable:
all_motor_vehicles

Problem Type:
Regression

---

# Dataset Overview

## Initial Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Records | TBD |
| Total Features | TBD |
| Date Range | TBD |
| Unique Count Points | TBD |
| Unique Roads | TBD |

---

# Data Quality Assessment

## Missing Values

Summary of missing values identified within the dataset.

### Findings

- Missing values in: TBD
- No missing values in target variable: TBD

### Actions

- Median imputation for numeric fields.
- Most frequent imputation for categorical fields.

---

## Duplicate Records

### Findings

TBD

### Actions

Duplicate records removed during preprocessing.

---

# Traffic Volume Analysis

## Distribution of Traffic Flow

Target variable:

all_motor_vehicles

### Findings

- Mean traffic volume: TBD
- Median traffic volume: TBD
- Maximum traffic volume: TBD
- Minimum traffic volume: TBD

### Interpretation

Understanding the distribution helps identify skewed traffic patterns and potential outliers.

---

# Traffic by Hour

## Objective

Understand daily traffic patterns.

### Findings

TBD

### Interpretation

Expected morning and evening peaks should be visible.

---

# Traffic by Direction

## Objective

Determine whether traffic differs by direction of travel.

### Findings

TBD

### Interpretation

Directional flow is expected to be an important predictive feature.

---

# Traffic by Road

## Objective

Identify roads carrying the highest traffic volumes.

### Findings

Top roads:

1. TBD
2. TBD
3. TBD
4. TBD
5. TBD

### Interpretation

Road identity is expected to be a significant predictor.

---

# Traffic Trends Over Time

## Objective

Understand long-term traffic changes.

### Findings

TBD

### Interpretation

Traffic patterns may reflect population growth, economic activity, and major events.

---

# Spatial Analysis

## Objective

Investigate geographical traffic patterns.

### Features Used

- Latitude
- Longitude
- Count Point

### Findings

TBD

### Interpretation

Traffic volume is expected to cluster around major transport corridors.

---

# Key Insights

## Insight 1

TBD

---

## Insight 2

TBD

---

## Insight 3

TBD

---

## Insight 4

TBD

---

# Impact on Feature Engineering

Based on the analysis, the following features will be retained:

- hour
- year
- direction_of_travel
- road_name
- road_type
- latitude
- longitude
- count_point_id
- link_length_km

Engineered features:

- is_peak_hour
- is_morning_peak
- is_evening_peak
- road_pair
- month
- season
- day_of_week

Excluded features:

- vehicle breakdown columns contributing directly to all_motor_vehicles

Reason:

Prevent data leakage.

---

# Conclusion

The Birmingham traffic dataset contains sufficient temporal, spatial, and road-network information to support explainable traffic-flow prediction using Genetic Programming and traditional machine learning techniques.

The next stage of the project is feature engineering and model development.