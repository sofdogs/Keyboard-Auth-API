import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import make_scorer, f1_score


def load_and_preprocess_data(file_path, scaler=None):
    df = pd.read_csv(file_path)

    # Extract relevant columns
    data = df[['Key', 'Delta Time (ms)']]

    # Convert the 'Key' column to numeric values (if it's not already)
    data['Key'] = pd.factorize(data['Key'])[0]

    # Normalize the 'DeltaTime' column
    if scaler is None:
        scaler = StandardScaler()
        data['Delta Time (ms)'] = scaler.fit_transform(data['Delta Time (ms)'].values.reshape(-1, 1))
    else:
        data['Delta Time (ms)'] = scaler.transform(data['Delta Time (ms)'].values.reshape(-1, 1))

    return data, scaler


def evaluate_anomaly_score(model, new_data, scaler, name):
    new_data, _ = load_and_preprocess_data(new_data, scaler)

    # Create features for the Isolation Forest
    new_X = new_data.values

    # Evaluate the model on the new dataset
    new_data_scores = model.decision_function(new_X)

    return new_data_scores, name


def train_model(user_data, model_path):

    isolation_forest = IsolationForest(n_estimators=500, max_samples=256, contamination=0.07333333333333333)

    # Train the model
    isolation_forest.fit(user_data)

    # Save the trained model to the specified path
    joblib.dump(isolation_forest, model_path)

    return isolation_forest


def load_model(model_path):
    # Load the trained model from the specified path
    return joblib.load(model_path)


def main():
    # Get the user's choice: train or evaluate
    choice = input("Enter 'train' to train the model or 'evaluate' to evaluate data: ")

    if choice == 'train':
        # Get the user's name
        user_name = input("Enter your name: ")

        # Construct file paths using the user's name
        user_file = f'{user_name}_keypress_data.csv'

        # Load and preprocess user data
        user_data, scaler = load_and_preprocess_data(user_file)
        user_data = user_data.values

        # Get the model name from the user
        model_name = input("Enter the model name: ")
        model_path = os.path.join("models", f"{model_name}_model.joblib")

        # Check if the model already exists
        if os.path.exists(model_path):
            print(f"The model '{model_name}' already exists. Loading the existing model.")
            isolation_forest = load_model(model_path)
        else:
            print(f"The model '{model_name}' does not exist. Training a new model.")
            isolation_forest = train_model(user_data, model_path)
    elif choice == 'evaluate':
        # Get the model name from the user
        model_name = input("Enter the model name: ")
        model_path = os.path.join("models", f"{model_name}_model.joblib")

        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"The model '{model_name}' does not exist. Please train the model first.")
            return
        else:
            print(f"Loading the model '{model_name}' for evaluation.")
            isolation_forest = load_model(model_path)
            
    else:
        print("Invalid choice. Please enter 'train' or 'evaluate'.")
        return


if __name__ == "__main__":
    main()