Model Results

Overview

Three machine learning models were trained and evaluated using the Birmingham Raw Traffic Counts dataset.

The target variable was:

all_motor_vehicles

The objective was to predict traffic flow using location, road, direction and temporal features.

⸻

Evaluation Metrics

The following metrics were used:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

⸻

Model Performance

Model	MAE	RMSE	R²
Random Forest	50.44	116.96	0.978
Gradient Boosting	161.75	272.84	0.880
Linear Regression	313.24	579.94	0.457

⸻

Findings

Random Forest

Random Forest produced the strongest performance across all evaluation metrics.

The model achieved an R² score of 0.978, meaning it explained approximately 97.8% of the variation in traffic flow.

The average prediction error was approximately 50 vehicles.

This suggests that traffic flow is influenced by complex non linear relationships between road type, road identity, location, direction of travel and time related features.

⸻

Gradient Boosting

Gradient Boosting also performed well, achieving an R² score of 0.880.

Although less accurate than Random Forest, it was still able to capture important traffic flow patterns within the dataset.

⸻

Linear Regression

Linear Regression produced the weakest performance.

The relatively low R² score of 0.457 indicates that traffic behaviour cannot be fully explained using simple linear relationships.

This result supports the use of more advanced computational intelligence techniques for traffic prediction.

⸻

Conclusion

The results demonstrate that Birmingham traffic flow contains complex patterns which are better captured by ensemble learning methods than traditional linear models.

Random Forest will be used as the primary benchmark model when evaluating the Genetic Programming symbolic regression model.