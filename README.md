# Digital Distraction vs Productivity Analyzer

## Overview
This project analyzes how digital habits such as screen time, social media usage, sleep, and study patterns affect student productivity. It uses a machine learning model to predict a productivity score and provide actionable recommendations.

## Problem Statement
Students often struggle with managing digital distractions, which negatively impacts their productivity. This project aims to model the relationship between lifestyle habits and productivity using AI/ML techniques.

## Objective
* Predict productivity score based on user habits
* Identify key factors affecting productivity
* Provide personalized improvement suggestions

## Features
* Synthetic dataset generation with realistic relationships
* Machine learning model using Random Forest Regressor
* Productivity score prediction (0–100)
* Classification into Low, Medium, High categories
* Rule-based recommendation system
* CLI-based user interaction

## Input Parameters
The model takes the following inputs:
* screen_time_hours
* social_media_time
* study_hours
* sleep_hours
* notifications_per_day
* exercise_minutes
* task_completion_rate
* break_frequency

## Output
* Productivity Score (0–100)
* Productivity Category:
  * Low (<40)
  * Medium (40–70)
  * High (>70)
* Personalized Recommendations

## Methodology
### 1. Data Generation
A synthetic dataset is created using statistical distributions:
* Normal distribution for continuous variables
* Poisson distribution for notifications
* Uniform distribution for task completion rate

A custom productivity function models:
* Positive effects (study, discipline, exercise)
* Negative effects (screen time, social media, notifications)
* Non-linear effects (sleep and breaks)

### 2. Model Selection
Random Forest Regressor is used because:
* Handles non-linear relationships
* Reduces overfitting via ensemble learning
* Works well with mixed feature types

### 3. Evaluation Metrics
* RMSE (Root Mean Squared Error)
* R² Score
* Cross-validation (5-fold)

### 4. Recommendation System
A rule-based system generates actionable suggestions based on input values.

## Installation
```bash
pip install numpy pandas scikit-learn
```

## Usage
Run the program:
```bash
python code.py
```
Enter input values when prompted.

## Sample Output
```
Productivity Score: 74.30
Category: High
Recommendations:
- (if applicable)
```

## Key Insights
* Task completion rate (discipline) significantly impacts productivity
* Excessive screen time and social media reduce productivity
* Both insufficient and excessive sleep negatively affect performance
* Balanced routines yield optimal productivity

## Limitations
* Uses synthetic data (not real-world collected)
* Assumes generalized user behavior
* Does not account for psychological or environmental factors

## Future Scope
* Integration with real user data (mobile usage tracking)
* Web or mobile application interface
* Advanced explainability using SHAP
* Personalized habit tracking over time

## Conclusion
This project demonstrates how machine learning can be applied to analyze behavioral patterns and improve productivity. It highlights the importance of balanced digital usage and disciplined routines.
