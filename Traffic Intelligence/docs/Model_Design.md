* Model Design

# Prediction Problem

The objective is to predict the value of
all_motor_vehicles using Birmingham
Department for Transport traffic count data.

Problem Type:
Regression

Target Variable:
all_motor_vehicles

Prediction Horizon:
Observed hourly traffic counts

# Selected Features

The following features were selected:

| Feature | Reason |
|----------|----------|
| year | captures long term traffic trends |
| hour | captures time of day effects |
| direction_of_travel | captures directional flow |
| road_type | major vs minor roads |
| road_name | identifies road characteristics |
| count_point_id | identifies traffic count location |
| latitude | geographic information |
| longitude | geographic information |
| is_morning_peak | engineered feature |
| is_evening_peak | engineered feature |
| is_peak_hour | engineered feature |

# Model Architecture

Data
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
Train Test Split
 ↓
Encoding
 ↓
Baseline Models
 ↓
Genetic Programming
 ↓
Evaluation

# Metrics 

MAE
RMSE
R²

