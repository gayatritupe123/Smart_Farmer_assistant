"""
generate_data.py
------------------------------------------------------
STEP 1 of 3.
This script creates 3 simple CSV files inside the data/ folder.
These are FAKE (synthetic) but realistic datasets, so you don't need
to search for real data. Just run this file once.

Run this file first:
    python generate_data.py
------------------------------------------------------
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)          # keeps results same every time we run this
os.makedirs("data", exist_ok=True)


# ======================================================
# DATASET 1: CROP DATA  (used for Crop Recommendation)
# ------------------------------------------------------
# Each crop grows best in a certain range of Nitrogen(N), Phosphorus(P),
# Potassium(K), temperature, humidity, soil pH and rainfall.
# We create rows of data for each crop, staying inside its normal range.
# ======================================================

crop_ranges = {
    "rice":       {"N": (80,120), "P": (35,60), "K": (35,45), "temp": (22,32), "humidity": (75,90), "ph": (5.5,7.0), "rainfall": (180,300)},
    "maize":      {"N": (70,110), "P": (35,60), "K": (15,25), "temp": (18,27), "humidity": (50,70), "ph": (5.5,7.5), "rainfall": (60,120)},
    "wheat":      {"N": (90,130), "P": (40,60), "K": (35,55), "temp": (10,25), "humidity": (50,65), "ph": (6.0,7.5), "rainfall": (60,100)},
    "cotton":     {"N": (100,140),"P": (35,60), "K": (15,25), "temp": (22,32), "humidity": (60,80), "ph": (5.8,8.0), "rainfall": (60,110)},
    "banana":     {"N": (90,120), "P": (70,100),"K": (45,55), "temp": (24,32), "humidity": (70,90), "ph": (5.5,6.8), "rainfall": (90,160)},
    "mango":      {"N": (15,40),  "P": (15,40), "K": (25,35), "temp": (25,35), "humidity": (40,60), "ph": (5.5,7.5), "rainfall": (60,110)},
    "chickpea":   {"N": (20,50),  "P": (55,80), "K": (70,90), "temp": (15,25), "humidity": (15,30), "ph": (6.0,8.0), "rainfall": (60,100)},
    "watermelon": {"N": (85,110), "P": (10,25), "K": (45,55), "temp": (24,32), "humidity": (60,75), "ph": (5.8,6.8), "rainfall": (35,60)},
}

rows = []
for crop, r in crop_ranges.items():
    for _ in range(90):                       # 90 sample rows per crop
        rows.append([
            np.random.uniform(*r["N"]),
            np.random.uniform(*r["P"]),
            np.random.uniform(*r["K"]),
            np.random.uniform(*r["temp"]),
            np.random.uniform(*r["humidity"]),
            np.random.uniform(*r["ph"]),
            np.random.uniform(*r["rainfall"]),
            crop
        ])

crop_df = pd.DataFrame(rows, columns=["N","P","K","temperature","humidity","ph","rainfall","label"])
crop_df.to_csv("data/crop_data.csv", index=False)
print(f"[OK] data/crop_data.csv created -> {crop_df.shape[0]} rows")


# ======================================================
# DATASET 2: YIELD DATA  (used for Yield Prediction)
# ------------------------------------------------------
# We predict a NUMBER: expected yield in tons per hectare.
# It roughly depends on rainfall, fertilizer used, pesticide used
# and temperature. More rainfall/fertilizer generally = more yield,
# up to a point. Too much pesticide or extreme temperature hurts yield.
# ======================================================

rows = []
for _ in range(700):
    rainfall = np.random.uniform(40, 300)
    fertilizer = np.random.uniform(20, 200)
    pesticide = np.random.uniform(0, 40)
    temperature = np.random.uniform(15, 38)

    # formula that decides the "true" yield, plus random noise
    yield_value = (
        2.0
        + rainfall * 0.015
        + fertilizer * 0.02
        - pesticide * 0.01
        - abs(temperature - 27) * 0.08
        + np.random.normal(0, 0.5)
    )
    yield_value = max(0.5, yield_value)        # yield can't be negative

    rows.append([rainfall, fertilizer, pesticide, temperature, round(yield_value, 2)])

yield_df = pd.DataFrame(rows, columns=["rainfall","fertilizer","pesticide","temperature","yield_tons_per_hectare"])
yield_df.to_csv("data/yield_data.csv", index=False)
print(f"[OK] data/yield_data.csv created -> {yield_df.shape[0]} rows")


# ======================================================
# DATASET 3: WEATHER DATA  (used for Weather Insights)
# ------------------------------------------------------
# We predict whether it will RAIN TOMORROW (Yes/No) based on
# today's humidity, air pressure, wind speed, temperature and
# cloud cover.
# ======================================================

rows = []
for _ in range(700):
    humidity = np.random.uniform(20, 100)
    pressure = np.random.uniform(990, 1025)
    wind_speed = np.random.uniform(0, 40)
    temperature = np.random.uniform(10, 42)
    cloud_cover = np.random.uniform(0, 100)

    rain_score = humidity * 0.5 + cloud_cover * 0.4 - (pressure - 1000) * 0.6 + np.random.normal(0, 8)
    rain_tomorrow = 1 if rain_score > 45 else 0     # 1 = Yes, 0 = No

    rows.append([humidity, pressure, wind_speed, temperature, cloud_cover, rain_tomorrow])

weather_df = pd.DataFrame(rows, columns=["humidity","pressure","wind_speed","temperature","cloud_cover","rain_tomorrow"])
weather_df.to_csv("data/weather_data.csv", index=False)
print(f"[OK] data/weather_data.csv created -> {weather_df.shape[0]} rows")

print("\nAll 3 datasets created successfully inside the data/ folder!")