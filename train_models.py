"""
train_models.py
------------------------------------------------------
STEP 2 of 3.
This script trains 3 machine learning models, one for each feature,
and saves them into the models/ folder so app.py can load and use them.

Models used:
  1. Crop Recommendation  -> Random Forest Classifier
  2. Yield Prediction     -> Random Forest Regressor (predicts a number)
  3. Weather Insights     -> Logistic Regression (predicts Rain / No Rain)

Run this file AFTER generate_data.py:
    python train_models.py
------------------------------------------------------
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

os.makedirs("models", exist_ok=True)


# ======================================================
# 1. CROP RECOMMENDATION MODEL  (Random Forest Classifier)
# ======================================================
print("\n[1/3] Training Crop Recommendation model...")

df = pd.read_csv("data/crop_data.csv")
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, crop_model.predict(X_test))
print(f"    Crop model accuracy: {accuracy:.2f}")

joblib.dump(crop_model, "models/crop_model.pkl")


# ======================================================
# 2. YIELD PREDICTION MODEL  (Random Forest Regressor)
# ======================================================
print("\n[2/3] Training Yield Prediction model...")

df = pd.read_csv("data/yield_data.csv")
X = df[["rainfall", "fertilizer", "pesticide", "temperature"]]
y = df["yield_tons_per_hectare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

yield_model = RandomForestRegressor(n_estimators=200, random_state=42)
yield_model.fit(X_train, y_train)

r2 = r2_score(y_test, yield_model.predict(X_test))
print(f"    Yield model R2 score: {r2:.2f}  (closer to 1.0 is better)")

joblib.dump(yield_model, "models/yield_model.pkl")


# ======================================================
# 3. WEATHER INSIGHTS MODEL  (Logistic Regression)
# ======================================================
print("\n[3/3] Training Weather Insights model...")

df = pd.read_csv("data/weather_data.csv")
X = df[["humidity", "pressure", "wind_speed", "temperature", "cloud_cover"]]
y = df["rain_tomorrow"]

# Logistic Regression works better when features are scaled to a similar range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

weather_model = LogisticRegression(max_iter=1000)
weather_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, weather_model.predict(X_test))
print(f"    Weather model accuracy: {accuracy:.2f}")

joblib.dump(weather_model, "models/weather_model.pkl")
joblib.dump(scaler, "models/weather_scaler.pkl")   # save the scaler too, app.py needs it


print("\nAll 3 models trained and saved inside the models/ folder!")