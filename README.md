# Pearls AQI Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Type-ML%20Forecasting-6c63ff?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A serverless end-to-end ML forecasting platform predicting Air Quality Index for Islamabad, Lahore, and Karachi up to 72 hours in advance with explainable AI, automated hazard alerts, and a live Streamlit dashboard.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Features](#features)
- [Models Used](#models-used)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Overview

Pearls AQI Predictor is a production-grade, serverless Machine Learning platform that forecasts Air Quality Index (AQI) for three major Pakistani cities **Islamabad, Lahore, and Karachi** at **24h, 48h, and 72h** horizons. The system ingests live air quality and meteorological data, engineers 40+ features, trains multi-model forecasters, and serves pre-computed predictions through both a Streamlit dashboard and a FastAPI REST API with zero live inference cost during user traffic.

**Live Demo:** [pearls-aqi-predictor-pakistan.streamlit.app](https://pearls-aqi-predictor-pakistan.streamlit.app/)

---

## Project Workflow

| Step | Description |
|:---:|:---|
| 1 | Live AQI and weather data ingested hourly via AQICN, OpenWeatherMap, Open-Meteo APIs |
| 2 | 40+ features engineered time lags (1h, 3h, 6h, 12h, 24h), rolling stats, wind vectors, cyclical encodings |
| 3 | Features stored in Hopsworks Feature Store (local Parquet fallback) |
| 4 | Daily retraining of Ridge Regression and Random Forest on fresh data |
| 5 | LSTM trained separately for 72h sequential multi-step forecasting |
| 6 | Inference pipeline generates pre-computed +24h, +48h, +72h predictions hourly |
| 7 | SHAP values computed to explain feature contributions per forecast |
| 8 | Alerts pipeline flags forecasts crossing Unhealthy / Hazardous AQI thresholds |
| 9 | Predictions served via Streamlit dashboard and FastAPI REST endpoints |
| 10 | GitHub Actions automates hourly inference, daily training, weekly benchmarks |

---

## Features

| Feature | Description |
|:---|:---|
| Multi-Horizon Forecasting | AQI predictions at +24h, +48h, and +72h for 3 cities |
| Explainable AI (SHAP) | Per-forecast feature attribution breakdown |
| Hazardous Air Alerts | Auto-flagged when forecasts cross dangerous AQI thresholds |
| Pre-Computed Predictions | Zero live inference cost sub-second dashboard response |
| Serverless Architecture | Fully decoupled feature, training, and inference pipelines |
| REST API | FastAPI endpoints for third-party integration |
| Interactive Dashboard | AQI dials, trend charts, forecast cards, model diagnostics |
| CI/CD Automation | GitHub Actions for hourly, daily, and weekly pipeline runs |

---

## Models Used

| Model | Horizons | Library | Notes |
|:---|:---:|:---|:---|
| Ridge Regression | 24h, 48h, 72h | `sklearn.linear_model` | Best overall performer across all horizons |
| Random Forest | 24h, 48h, 72h | `sklearn.ensemble` | Strong ensemble baseline |
| Persistence Baseline | 24h, 48h, 72h | Custom | Naive last-value benchmark |
| LSTM | 72h | `tensorflow/keras` | Sequential deep learning for long-horizon |

> All models trained with StandardScaler feature scaling and time-based train/test splits.

---

## Results

| Model | Horizon (h) | RMSE | MAE | R² |
|:---|:---:|:---:|:---:|:---:|
| **Ridge** | **24** | **19.406** | **12.667** | **0.770** |
| Random Forest | 24 | 20.146 | 12.669 | 0.752 |
| Persistence | 24 | 24.347 | 14.295 | 0.637 |
| **Ridge** | **48** | **25.155** | **17.259** | **0.615** |
| Random Forest | 48 | 27.512 | 19.232 | 0.539 |
| Persistence | 48 | 29.692 | 18.756 | 0.463 |
| **Ridge** | **72** | **26.416** | **18.798** | **0.575** |
| Random Forest | 72 | 28.610 | 20.423 | 0.502 |
| Persistence | 72 | 31.377 | 20.939 | 0.401 |

> **Best Model: Ridge Regression** consistently lowest RMSE and highest R² across all three forecast horizons.

---

## Tech Stack

| Tool | Purpose |
|:---|:---|
| Python 3.10+ | Core language across all pipelines |
| Pandas, NumPy, SciPy | Feature engineering and time-series processing |
| Scikit-learn | Ridge Regression, Random Forest, StandardScaler |
| TensorFlow / Keras | LSTM sequential forecasting |
| SHAP | Explainable AI feature attribution |
| Hopsworks | Feature Store and Model Registry |
| FastAPI, Uvicorn | REST API backend |
| Streamlit, Plotly | Interactive web dashboard |
| GitHub Actions | CI/CD pipeline automation |
| AQICN, OpenWeatherMap, Open-Meteo | Live data sources |

---

## How to Run

**1. Clone the repository**
~~~bash
git clone https://github.com/Rabia605/pearls-aqi-predictor.git
cd pearls-aqi-predictor
~~~

**2. Set up environment variables**
~~~bash
cp .env.example .env
# Fill in AQICN_TOKEN, OPENWEATHER_API_KEY, HOPSWORKS_API_KEY
~~~

**3. Install dependencies**
~~~bash
pip install -r requirements.txt
~~~

**4. Run the feature pipeline**
~~~bash
python -m ml.pipelines.feature_pipeline
~~~

**5. Train the models**
~~~bash
python -m ml.pipelines.training_pipeline
~~~

**6. Run inference**
~~~bash
python -m ml.pipelines.inference_pipeline
~~~

**7. Launch the Streamlit dashboard**
~~~bash
streamlit run streamlit_app/app.py
~~~

**8. Launch the FastAPI backend**
~~~bash
uvicorn api.main:app --reload
~~~

---

## Project Structure

~~~
pearls-aqi-predictor/
│
├── ml/
│   ├── common/         (aqi.py, cities.py, config.py)
│   ├── clients/        (aqicn.py, openmeteo.py, openweather.py)
│   ├── features/       (engineering.py)
│   ├── training/       (dataset.py, models.py, sequences.py)
│   ├── storage/        (hopsworks_store.py, feature_store.py, registry.py)
│   ├── pipelines/      (feature, training, inference, alerts, explain, lstm)
│   └── analysis/       (eda.py, eda.ipynb)
│
├── api/
│   └── main.py
│
├── streamlit_app/
│   └── app.py
│
├── data/               (local Parquet fallbacks)
├── models/             (serialized model artifacts)
├── docs/               (architecture diagrams, reports)
├── .github/workflows/  (CI/CD automation)
│
├── cities.json
├── .env.example
├── requirements.txt
├── requirements-deep.txt
├── requirements-explain.txt
└── README.md
~~~

---

## Author

**Rabia Noreen**
*Software Engineer | Building with AI & ML*

---

<p align="center">If this project inspired you, hit that ⭐ button!</p>
