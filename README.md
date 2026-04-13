# 🏠 House Price Prediction API

## 📌 Overview
End-to-end Machine Learning project to predict house prices using **Lasso Regression**.  
The project covers the full ML lifecycle: data analysis, feature engineering, model training, evaluation, and deployment using FastAPI.

---

## 🎯 Problem Statement
Predict house sale prices based on various property features such as:
- living area
- overall quality
- number of rooms
- garage capacity
- construction year

This is a **supervised regression problem**.

---

## 📊 Dataset
Kaggle Housing Dataset:  
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

---

## ⚙️ Approach

### 1. Data Analysis (EDA)
- Distribution analysis (SalePrice)
- Outlier detection
- Correlation analysis
- Feature relationships (scatter plots, heatmap)

### 2. Feature Engineering
- Total square footage (`TotalSF`)
- House age (`HouseAge`)
- Remodel age (`RemodAge`)
- Total bathrooms (`TotalBath`)

### 3. Model Training
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression (final model)

### 4. Hyperparameter Tuning
- Alpha tuning for Lasso
- Selected best alpha = **0.001**

### 5. Model Evaluation
- RMSE: ~19,400  
- R² Score: ~0.92  
- Cross-validation mean R²: ~0.91  

---

## 🚀 API Deployment

The trained model is deployed using **FastAPI**.

### Run Locally

bash
uvicorn app.main:app --reload
Open the browser :  http://127.0.0.1:8000/docs
Example Input : 
{
  "MSSubClass": 60,
  "LotArea": 8450,
  "OverallQual": 7,
  "OverallCond": 5,
  "YearBuilt": 2003,
  "YearRemodAdd": 2003,
  "first_flr_sf": 856,
  "second_flr_sf": 854,
  "GrLivArea": 1710,
  "BsmtFullBath": 1,
  "BsmtHalfBath": 0,
  "FullBath": 2,
  "HalfBath": 1,
  "BedroomAbvGr": 3,
  "KitchenAbvGr": 1,
  "TotRmsAbvGrd": 8,
  "GarageCars": 2,
  "GarageArea": 548,
  "TotalBsmtSF": 856,
  "Fireplaces": 0,
  "YrSold": 2008,
  "MoSold": 2
}

Example output : 
{
  "predicted_price": 286244.69,
  "model": "lasso_full"
}
---

## key learning 
Importance of feature engineering consistency between training and inference
Handling categorical variables with one-hot encoding
Using Lasso for feature selection and regularization
Aligning input features using reindex()
Building and deploying ML models as APIs

## Project Structure
house-price-prediction/
│
├── app/
│   └── main.py
├── models/
│   ├── lasso_full_model.pkl
│   └── model_columns_full.pkl
├── notebooks/
├── data/
├── requirements.txt
└── README.md


## Tech Stack
Python
Pandas, NumPy
Scikit-learn
FastAPI
Uvicorn

## Author

Pramila Soni