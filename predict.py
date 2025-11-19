import pickle
import sklearn
import xgboost as xgb
import uvicorn
import pandas as pd
import numpy as np

from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Pydantic Model Definition (Using Literal for strict validation) ---
class NurseryRequest(BaseModel):
    """
    Pydantic model that strictly enforces input strings 
    to match the specified allowed categories using Literal types.
    """
    parents: Literal['usual', 'pretentious', 'great_pret']
    has_nurs: Literal['proper', 'less_proper', 'improper', 'critical', 'very_crit'] = Field(..., alias='has_nurs')
    form: Literal['complete', 'completed', 'incomplete', 'foster']
    children: Literal['1', '2', '3', 'more']
    housing: Literal['convenient', 'less_conv', 'critical']
    finance: Literal['convenient', 'inconv']
    social: Literal['nonprob', 'slightly_prob', 'problematic']
    health: Literal['recommended', 'priority', 'not_recom']
    
    # Allows Pydantic to accept 'has_nurs' in the input JSON
    class Config:
        populate_by_name = True
        # use_enum_values = True # Not needed with Literal, but good practice if using Enum

# --- FastAPI Application Setup ---
app = FastAPI(title="nursery-application-decision")

model_name = 'model.bin'

# Load the DictVectorizer (dv) and the trained XGBoost model (model)
with open(model_name, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

@app.post("/predict")
def predict(profile: NurseryRequest):
    """
    Predict the application decision based on the provided nursery profile.
    """
    
    # 1. Convert the Pydantic model instance into a dictionary 
    #    that the DictVectorizer expects.
    profile_dict = profile.model_dump(by_alias=True)
    
    # 2. Transform the dictionary into a sparse matrix
    features = list(dv.get_feature_names_out())
    # Note: dv.transform expects a list of dictionaries, even if it's just one sample
    X = dv.transform([profile_dict]) 
    
    # 3. Convert to DMatrix for XGBoost prediction
    dX = xgb.DMatrix(X, feature_names=features)
    
    # 4. Make prediction
    class_of_application = model.predict(dX)[0]
    
    # 5. Return the result as a standard Python float/int
    return float(class_of_application)

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=9696)
