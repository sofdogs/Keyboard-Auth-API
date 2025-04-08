from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

class RequestData(BaseModel):
    modelName: str
    data: dict

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Declare model and scaler variables at the top level
model = None
scaler = None


def load_and_preprocess_data(data, scaler=None):
    df = pd.DataFrame(data)
    
    # Convert the 'Key' column to numeric values using factorize
    df['Key'] = pd.factorize(df['Key'])[0]

    # Combine 'Key' and 'Delta' columns
    combined_data = pd.DataFrame({
        'Key': df['Key'],
        'Delta Time': df['Delta']
    })

    # Normalize the 'DeltaTime' column
    if scaler is None:
        scaler = StandardScaler()
        combined_data['Delta Time'] = scaler.fit_transform(combined_data['Delta Time'].values.reshape(-1, 1))
    else:
        combined_data['Delta Time'] = scaler.transform(combined_data['Delta Time'].values.reshape(-1, 1))

    return combined_data, scaler


def evaluate_anomaly_score(model, new_data, scaler):
    new_data, _ = load_and_preprocess_data(new_data, scaler)

    # Create features for the Isolation Forest
    new_X = new_data.values

    # Evaluate the model on the new dataset
    new_data_scores = model.decision_function(new_X)

    return new_data_scores


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def get_model(name: str = Form(...)) -> IsolationForest:
    model_path = f"{name}_model.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model with name '{name}' not found.")


@app.post("/train-model/")
def train_model(request_data: RequestData):
    global model
    global scaler

    name = request_data.modelName
    data = request_data.data


    df, scaler = load_and_preprocess_data(data, scaler)

    # Check if the model already exists
    model_path = f"{name}_model.joblib"
    try:
        existing_model = joblib.load(model_path)
        model = existing_model
        message = f"Model {name} updated"
    except FileNotFoundError:
        # If the file is not found, create a new model
        model = IsolationForest(n_estimators=500, max_samples=256, contamination=0.07333333333333333)
        message = f"Model {name} created" 

    # Continue training the model with the new data
    user_data = df.values
    model.fit(user_data)

    # Save the updated model
    joblib.dump(model, model_path)

    return {"message": message}


@app.post("/evaluate-data/")
def evaluate_data(request_data: RequestData):

    global model
    global scaler

    name = request_data.modelName
    data = request_data.data


    model_path = f"{name}_model.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model with name '{name}' not found.")
    
    # Evaluate the model on the new dataset
    new_data_scores = evaluate_anomaly_score(model, data, scaler)

    # Set the threshold to classify sequences as normal or anomalous
    threshold = 0.002
    anomalies = new_data_scores < threshold

    # Print the percentage of sequences classified as normal
    percentage_normal = (1 - np.mean(anomalies)) * 100

    # Return the anomaly score and percentage of normal sequences
    return {
        "anomaly_score": np.mean(anomalies),
        "percentage_normal": percentage_normal,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)