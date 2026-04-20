# 📊 Telco Customer Churn Prediction

## 🔍 Overview
This project builds an end-to-end machine learning solution to predict customer churn using the Telco Customer Churn dataset.

The goal is to identify customers who are likely to leave (churn) so that businesses can take proactive retention actions.

---

## 🎯 Problem Statement
Customer churn is a critical challenge for subscription-based businesses.

The objective of this project is to:
- Analyze customer behavior
- Identify key drivers of churn
- Build a predictive model to detect high-risk customers early

---

## 📁 Dataset
- Source: Kaggle Telco Customer Churn Dataset  
- Records: 7,043 customers  
- Features: 21 attributes (demographics, services, billing, etc.)

---

## 🔎 Key Insights (EDA)

From exploratory data analysis:

- 📉 **Month-to-month contracts** have the highest churn (~43%)
- ⚡ **Fiber optic users** show higher churn (~42%)
- 💳 **Electronic check payments** have the highest churn (~45%)
- ⏳ **Low tenure customers (~18 months)** are more likely to churn
- 💰 **Higher monthly charges (~74)** correlate with churn

👉 High-risk profile:
> Month-to-month + Fiber + Electronic check

---

## 🛠️ Feature Engineering

- `AvgMonthlySpend = TotalCharges / tenure`
- `IsLongTerm = tenure > 24`

These features capture customer value and loyalty patterns.

---

## 🤖 Model

### Final Model:
- **Logistic Regression**
- `class_weight = balanced`
- `C = 10`
- `max_iter = 5000`

### Why Logistic Regression?
- Strong performance on recall
- Interpretable (feature importance)
- More reliable than Random Forest for this dataset

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| Recall | ~0.80 |
| ROC-AUC | ~0.86 |
| Threshold | 0.3 |

### 🎯 Why Recall?
In churn prediction:
> Missing a churner is more costly than a false alarm

So the model is optimized to **capture as many churners as possible**.

---

## ⚖️ Model Comparison

| Model | Recall |
|------|--------|
| Logistic Regression | ~0.80 ✅ |
| Random Forest | ~0.47 ❌ |

👉 Logistic Regression significantly outperformed Random Forest in detecting churn.

---

## 🧠 Model Interpretation

Top features influencing churn:

- InternetService_Fiber optic ↑
- TotalCharges ↑
- Streaming services ↑
- Electronic payment methods ↑

👉 Positive coefficients increase churn probability  
👉 Negative coefficients reduce churn risk  

---

## 🚀 API Deployment

The model is deployed using **FastAPI**.

### Run locally:
``bash
uvicorn app.main:app --reload

---

## End Points 
POST /predict

## Example Input:
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 500.0
}

## Example Output 
{
  "churn_probability": 0.86,
  "prediction": "Churn"
}

## Project Structure 

telco-churn-prediction/
│
├── app/
│   └── main.py              # FastAPI application
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── model_columns.pkl
│
├── notebooks/
│   └── Telco-Customer-Churn.ipynb
│
├── requirements.txt
└── README.md

## Business Value

This model enables businesses to:

Identify high-risk customers early
Design targeted retention campaigns
Reduce churn and increase customer lifetime value

## Key Learnings
Importance of EDA for business insights
Handling imbalanced datasets
Trade-off between precision vs recall
Using Pipeline to avoid data leakage
Building production-ready ML APIs

## Future Improvements
Try advanced models (XGBoost, LightGBM)
Add automated feature selection
Deploy to cloud (AWS / Azure)
Add real-time monitoring


## Conclusion

This project demonstrates how machine learning can go beyond prediction and act as a decision-support tool.
By combining data analysis, model optimization, and deployment, the solution provides actionable insights for customer retention and business growth.




