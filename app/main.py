from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model pipeline
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Churn Prediction API"}


@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Recreate feature engineering from training
    input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    input_df["IsLongTerm"] = (input_df["tenure"] > 24).astype(int)

    # Apply same encoding
    input_df = pd.get_dummies(input_df)

    # Align columns with training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    # Use tuned threshold
    prediction = "Churn" if prob > 0.3 else "No Churn"

    return {
        "churn_probability": round(float(prob), 3),
        "prediction": prediction
    }