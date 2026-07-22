# 🌾 Smart Farmer Assistant (Simplified — 3 Features)

A simple ML project with **3 features**, **3 models**, **3 files** you need to run in order.

## Folder Structure

```
smart_farmer_assistant/
├── data/                  # created automatically by generate_data.py
│   ├── crop_data.csv
│   ├── yield_data.csv
│   └── weather_data.csv
├── models/                # created automatically by train_models.py
│   ├── crop_model.pkl
│   ├── yield_model.pkl
│   ├── weather_model.pkl
│   └── weather_scaler.pkl
├── generate_data.py       # Step 1: creates the datasets
├── train_models.py        # Step 2: trains the 3 models
├── app.py                 # Step 3: the web app (frontend)
└── requirements.txt
```

You only need to create 4 files yourself: `generate_data.py`, `train_models.py`,
`app.py`, and `requirements.txt`. The `data/` and `models/` folders + their files
get created automatically when you run the scripts.

## The 3 Features

| Feature | Model | What it predicts |
|---|---|---|
| 🌱 Crop Recommendation | Random Forest Classifier | Best crop name for given soil & climate |
| 📈 Yield Prediction | Random Forest Regressor | Expected yield (tons per hectare) — an actual number |
| ☁️ Weather Insights | Logistic Regression | Will it rain tomorrow? (Yes/No + probability) |

## How to Run in VS Code

**1. Open the project folder in VS Code**, then open a terminal (`` Ctrl + ` ``).

**2. (Recommended) create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install the required libraries:**
```bash
pip install -r requirements.txt
```

**4. Create the datasets** (run once):
```bash
python generate_data.py
```
You should see 3 lines confirming the CSV files were created.

**5. Train the models** (run once):
```bash
python train_models.py
```
You should see accuracy scores printed for each of the 3 models.

**6. Launch the app:**
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501` — that's your project running.

## Order of Execution (important!)

```
generate_data.py  --->  train_models.py  --->  app.py
   (makes CSVs)          (makes .pkl models)     (the website)
```

You must run them in this order the first time. After that, you only
need `streamlit run app.py` to reopen the app — the data and models are
already saved on disk.

## If Something Goes Wrong

- **"Models not found" error in the app** → you forgot to run `generate_data.py`
  and `train_models.py` first.
- **"streamlit: command not found"** → run `pip install -r requirements.txt` again,
  and make sure your virtual environment is activated.
- **Port already in use** → close other Streamlit apps, or run
  `streamlit run app.py --server.port 8502`.