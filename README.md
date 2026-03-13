# 🍽️ Restaurant Rating Prediction App

An end-to-end machine learning project that predicts restaurant aggregate ratings based on key features — from data exploration and multi-model comparison to a deployed interactive Streamlit web application.

## 📌 Project Overview

This project covers the full ML pipeline:
- Exploratory data analysis on a real restaurant dataset
- Training and comparing 5 regression models with GridSearchCV
- Selecting the best model and deploying it as a live web app

Users can input restaurant details and instantly get a predicted rating category.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

## 🚀 Run the App
```bash
git clone https://github.com/Ulasaydo/Restaurants-Rating-Prediction-App.git
cd Restaurants-Rating-Prediction-App
pip install pandas scikit-learn streamlit seaborn matplotlib joblib
streamlit run app.py
```

## 🖥️ App Features

Input the following restaurant details to get an instant rating prediction:

| Input | Description |
|---|---|
| Average Cost for Two | Estimated cost in local currency |
| Table Booking | Whether the restaurant accepts reservations |
| Online Delivery | Whether the restaurant offers online ordering |
| Price Range | 1 (cheapest) to 4 (most expensive) |

The app returns one of five rating categories: **Poor / Average / Good / Very Good / Excellent**

## 📁 Project Structure
```
Restaurants-Rating-Prediction-App/
│
├── Data.csv           # Raw restaurant dataset
├── main.ipynb         # EDA, model training & comparison
├── app.py             # Streamlit web application
├── mlmodel.pkl        # Trained best model (Random Forest via GridSearchCV)
└── scaler.pkl         # Fitted StandardScaler
```

## 📈 Methodology

1. **EDA** — Analyzed average cost by city, cuisine votes, online delivery distribution, rating distributions, and feature correlations (pairplot)
2. **Preprocessing** — Label encoding for `Has Table booking` and `Has Online delivery`; StandardScaler for feature normalization; removed "Not rated" entries
3. **Features used** — `Average Cost for Two`, `Has Table booking`, `Has Online delivery`, `Price Range`
4. **Model Comparison with GridSearchCV** — Trained and tuned 5 models:

| Model | Tuned Parameters |
|---|---|
| Linear Regression | — |
| SVR | C, kernel, degree |
| Decision Tree | max_depth, min_samples_leaf, min_samples_split |
| **Random Forest** ✅ | max_depth, n_estimators |
| KNN | n_neighbors |
| AdaBoost | n_estimators, learning_rate |

5. **Best Model** — Random Forest Regressor (selected and serialized as `mlmodel.pkl`)
6. **Deployment** — Interactive Streamlit UI with real-time predictions

## 👤 Author

**Ulaş Göksu Aydoğdu** — [LinkedIn](https://linkedin.com/in/ulasaydo) · [GitHub](https://github.com/Ulasaydo)
