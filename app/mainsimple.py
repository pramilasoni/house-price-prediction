from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# load trained model
with open("./models/lassofull_model.pkl", "rb") as f:
    model = pickle.load(f)

# load training columns
with open("./models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])

    # Feature Engineering (same as notebook)
    input_df["TotalSF"] = (
    input_df.get("TotalBsmtSF", 0)
    + input_df.get("1stFlrSF", 0)
    + input_df.get("2ndFlrSF", 0)
    )

    input_df["HouseAge"] = input_df.get("YrSold", 0) - input_df.get("YearBuilt", 0)

    input_df["RemodAge"] = input_df.get("YrSold", 0) - input_df.get("YearRemodAdd", 0)

    input_df["TotalBath"] = (
    input_df.get("FullBath", 0)
    + 0.5 * input_df.get("HalfBath", 0)
    + input_df.get("BsmtFullBath", 0)
    + 0.5 * input_df.get("BsmtHalfBath", 0)
    )

    # one-hot encode input
    input_df = pd.get_dummies(input_df, drop_first=True)

    # align with training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    print("Non-zero input columns:")
    print(input_df.loc[:, (input_df != 0).any(axis=0)].T)
    prediction_log = model.predict(input_df)[0]
    prediction = np.exp(prediction_log)

    return {"predicted_price": float(prediction)}