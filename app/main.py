from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="House Price Prediction API")

# Load model
with open("./models/lassofull_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("./models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


class HouseInput(BaseModel):
    MSSubClass: int
    LotArea: float
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    first_flr_sf: float
    second_flr_sf: float
    GrLivArea: float
    BsmtFullBath: int
    BsmtHalfBath: int = 0
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    TotRmsAbvGrd: int
    GarageCars: int
    GarageArea: float
    TotalBsmtSF: float
    Fireplaces: int
    YrSold: int
    MoSold: int


def prepare_input(data: HouseInput) -> pd.DataFrame:
    # Convert schema to dictionary
    row = data.dict()

    # Map API-friendly names back to training-style feature names
    row["1stFlrSF"] = row.pop("first_flr_sf")
    row["2ndFlrSF"] = row.pop("second_flr_sf")

    # Create DataFrame
    input_df = pd.DataFrame([row])

    # Feature engineering (same logic as notebook)
    input_df["TotalSF"] = (
        input_df["TotalBsmtSF"]
        + input_df["1stFlrSF"]
        + input_df["2ndFlrSF"]
    )

    input_df["HouseAge"] = input_df["YrSold"] - input_df["YearBuilt"]
    input_df["RemodAge"] = input_df["YrSold"] - input_df["YearRemodAdd"]

    input_df["TotalBath"] = (
        input_df["FullBath"]
        + 0.5 * input_df["HalfBath"]
        + input_df["BsmtFullBath"]
        + 0.5 * input_df["BsmtHalfBath"]
    )

    # Apply same encoding as training
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    return input_df


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseInput):
    input_df = prepare_input(data)

    prediction_log = model.predict(input_df)[0]
    prediction = np.exp(prediction_log)

    return {
        "predicted_price": round(float(prediction), 2),
        "model": "lasso_full"
    }