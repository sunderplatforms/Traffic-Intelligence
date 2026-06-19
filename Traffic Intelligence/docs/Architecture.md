Architecture

System Overview

Smart Birmingham Traffic Intelligence is an explainable traffic prediction platform that uses Genetic Programming and traditional machine learning techniques to model traffic flow across Birmingham road networks.

The platform processes Department for Transport traffic data, engineers predictive features, trains multiple regression models, and exposes predictions through visualisation and API layers.

High Level Architecture

City Data Layer
↓
RouteSenseEngine
↓
EvoTrafficCore
↓
FormulaLens
↓
FlowPredictAPI
↓
CityPulseDashboard
↓
Platform Layer

Components

City Data Layer

Responsible for collecting and storing Department for Transport Birmingham traffic datasets.

Responsibilities:

* Raw data ingestion
* Dataset validation
* Dataset versioning

RouteSenseEngine

Responsible for preparing data for analysis.

Responsibilities:

* Data cleaning
* Missing value handling
* Feature engineering
* Categorical encoding

EvoTrafficCore

Core modelling engine.

Responsibilities:

* Symbolic Regression
* Linear Regression
* Random Forest
* Gradient Boosting
* Model evaluation

FormulaLens

Provides explainability.

Responsibilities:

* Formula extraction
* Symbolic tree visualisation
* Model interpretation

FlowPredictAPI

Provides prediction services.

Responsibilities:

* Prediction endpoint
* Health checks
* Model serving

CityPulseDashboard

User interface for analysis.

Responsibilities:

* Traffic trends
* Traffic predictions
* Road comparisons
* Model performance visualisation

Platform Layer

Deployment and operations layer.

Responsibilities:

* Docker
* Kubernetes
* CI/CD
* Monitoring