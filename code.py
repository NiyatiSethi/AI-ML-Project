import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# DATA GENERATION
def generate_data(N=700):
    np.random.seed(42)

    screen_time = np.clip(np.random.normal(6, 2, N), 1, 12)
    social_media = np.clip(np.random.normal(3, 1.5, N), 0, 8)
    study_hours = np.clip(np.random.normal(4, 2, N), 0, 10)
    sleep_hours = np.clip(np.random.normal(6.5, 1.5, N), 3, 10)
    notifications = np.random.poisson(80, N)
    exercise = np.clip(np.random.normal(30, 20, N), 0, 120)
    task_rate = np.random.uniform(0.4, 1.0, N)
    breaks = np.random.randint(1, 10, N)

    sleep_effect = -(sleep_hours - 7)**2
    break_effect = -(breaks - 5)**2
    noise = np.random.normal(0, 5, N)

    productivity = (
        6 * study_hours +
        35 * task_rate +
        0.5 * exercise -
        5 * screen_time -
        4 * social_media -
        0.08 * notifications +
        (-2 * (sleep_hours - 7)**2) +
        (-1.5 * (breaks - 5)**2) +
        noise
    )

    productivity = productivity + 30
    productivity = np.clip(productivity, 0, 100)

    df = pd.DataFrame({
        "screen_time_hours": screen_time,
        "social_media_time": social_media,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications,
        "exercise_minutes": exercise,
        "task_completion_rate": task_rate,
        "break_frequency": breaks,
        "productivity_score": productivity
    })

    return df

# PREPROCESSING
def preprocess(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.drop("productivity_score", axis=1)
    y = df["productivity_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns

# MODEL TRAINING
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,  
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# EVALUATION

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return y_pred


# CROSS VALIDATION

def cross_validate(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R²: {scores.mean():.2f}")


# FEATURE IMPORTANCE
def feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    print("\nFeature Importance:\n", importance.sort_values(ascending=False))


# RECOMMENDATION SYSTEM
def generate_recommendations(data):
    recs = []

    if data["screen_time_hours"] > 7:
        recs.append("Reduce screen time")

    if data["social_media_time"] > 3:
        recs.append("Limit social media usage")

    if data["study_hours"] < 3:
        recs.append("Increase study hours")

    if data["sleep_hours"] < 6:
        recs.append("Improve sleep duration")
    elif data["sleep_hours"] > 9:
        recs.append("Avoid oversleeping")

    if data["notifications_per_day"] > 100:
        recs.append("Reduce notifications")

    if data["exercise_minutes"] < 20:
        recs.append("Add daily exercise")

    if data["task_completion_rate"] < 0.6:
        recs.append("Improve task discipline")

    return recs[:3]


# PREDICTION
def predict(model, feature_names, user_input):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    score = model.predict(input_df)[0]
    if score < 40:
        category = "Low"
    elif score < 70:
        category = "Medium"
    else:
        category = "High"

    recs = generate_recommendations(user_input)

    print("\n--- RESULT ---")
    print(f"Productivity Score: {score:.2f}")
    print(f"Category: {category}")

    print("\nRecommendations:")
    for r in recs:
        print("-", r)

def main():
    df = generate_data()

    X_train, X_test, y_train, y_test, features = preprocess(df)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)
    cross_validate(model, X_train, y_train)
    feature_importance(model, features)

    print("\nEnter your data:")

    user_input = {}
    for feature in features:
        user_input[feature] = float(input(f"{feature}: "))

    predict(model, features, user_input)

main()