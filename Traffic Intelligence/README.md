# Smart Birmingham Traffic Intelligence
## Explainable Traffic Flow Prediction using Genetic Programming

## Project Overview

This project explores how **Computational Intelligence** can be used to support **smart transportation** by modelling and predicting traffic flow into and out of Birmingham.

The main focus of the project is to use **Genetic Programming**, through **Symbolic Regression**, to evolve mathematical expressions that can predict traffic flow from road traffic data. Unlike some black-box machine learning models, symbolic regression can produce human-readable formulae, making the model easier to interpret and explain.

The project is inspired by research in Intelligent Transportation Systems, where traffic modelling and prediction are considered important parts of smart city infrastructure. Research has shown that Genetic Programming can be used for urban traffic modelling by treating traffic prediction as a symbolic regression problem. 
Birmingham is used as the case study location because public road traffic statistics are available from the Department for Transport. The Birmingham traffic statistics dataset includes long-term road traffic information, count points, vehicle class data, annual average daily flow, directional flow, and raw count data. 

---

## Project Aim

The aim of this project is to develop an explainable computational intelligence model that can predict traffic flow into and out of Birmingham using road traffic data and Genetic Programming.

---

## Project Objectives

The main objectives of this project are:

1. Collect and prepare Birmingham road traffic data.
2. Explore traffic patterns by road, year, vehicle type, and direction.
3. Build a Genetic Programming symbolic regression model to predict traffic flow.
4. Compare the Genetic Programming model with traditional machine learning models.
5. Evaluate model performance using regression metrics such as MAE, RMSE, and R².
6. Export and visualise the best evolved symbolic expression.
7. Discuss how the approach could support smart transportation and traffic planning.

---

## Background

Smart transportation is an important part of modern smart city development. Intelligent Transportation Systems aim to improve how transport networks are monitored, managed, and planned.

Research on Genetic Programming for urban traffic modelling describes intelligent transportation as a key part of smart cities and explains that traffic modelling and prediction are central to Intelligent Transportation Systems. 
Transport for West Midlands also highlights the importance of data collection, management, analysis, forecasting, and research for transport planning and operations across the West Midlands. Their Data Insight work includes transport data, operational data, traffic counts, vehicle classifications, congestion, road traffic collisions, modelling, forecasting, and nowcasting.

This project uses these ideas to create a simplified but realistic prototype of a traffic prediction system.

---

## Why Birmingham?

Birmingham is a suitable case study because it is a major UK city with significant traffic movement and publicly available transport data.

The Department for Transport road traffic statistics page for Birmingham states that **3,738 million vehicle miles** were travelled on roads in Birmingham in **2025**. The dataset covers the period **1993 to 2025** and includes **590 count points**. 

Available Birmingham traffic datasets include:

- Count points
- Traffic by vehicle class
- Annual average daily flow
- Annual average daily flow by direction
- Raw traffic counts

The directional flow dataset is especially useful for this project because it can support analysis of traffic travelling into and out of Birmingham.
---

## Computational Intelligence Approach

This project uses **Genetic Programming**, a type of evolutionary algorithm inspired by natural selection.

In this project, Genetic Programming is used for **Symbolic Regression**. Instead of simply training a model with fixed rules, symbolic regression evolves mathematical expressions over many generations.

The process works as follows:

1. Generate an initial population of random mathematical expressions.
2. Test each expression against the training data.
3. Score each expression using a fitness metric such as prediction error.
4. Select better-performing expressions.
5. Create new expressions using crossover and mutation.
6. Repeat the process over multiple generations.
7. Return the best evolved expression.

This makes the model useful not only for prediction, but also for explainability.

---

## Why Symbolic Regression?

Symbolic Regression is useful because it can discover relationships between input variables and a target variable in the form of a mathematical expression.

For example, instead of only producing a prediction such as:

```text
Predicted traffic flow = 42,000 vehicles