from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List

# Load the pre-trained model
with open('/kaggle/working/house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

# Define the input feature model
class HouseFeatures(BaseModel):
    features: List[float]

# Preprocessing function used in model training
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Load training data to get the list of columns to drop
    df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
    nulls = df_train.isnull().mean()
    drop = nulls[nulls > 0].index
    
    df.drop(drop, axis=1, inplace=True)
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = encoder.fit_transform(df[col])
    
    return df

@app.post("/predict")
def predict_price(house: HouseFeatures):
    # Convert input features to a pandas DataFrame
    features = pd.DataFrame([house.features], columns=[preprocess])
    
    # Apply the same preprocessing as in training
    preprocessed_features = preprocess(features)
    
    # Make prediction
    prediction = model.predict(preprocessed_features)
    
    return {"predicted_price": prediction[0]}

# To run the server, use `uvicorn main:app --reload` from the command line