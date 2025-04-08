import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to build the neural network model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess data and train the model
def train_model(data, labels):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    input_shape = data_scaled.shape[1]

    model = build_model(input_shape)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=50, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on the test set: {accuracy}")

    return model, scaler

# Function to save the trained model and scaler to files
def save_model(model, scaler, model_filename='keystroke_model.h5', scaler_filename='scaler.pkl'):
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    # Save the scaler using joblib
    import joblib
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

# Function to load the trained model and scaler from files
def load_model(model_filename='keystroke_model.h5', scaler_filename='scaler.pkl'):
    loaded_model = tf.keras.models.load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Load the scaler using joblib
    import joblib
    loaded_scaler = joblib.load(scaler_filename)
    print(f"Scaler loaded from {scaler_filename}")

    return loaded_model, loaded_scaler

# Function to predict whether keystroke data belongs to the same person
def predict(model, scaler, data):
    data_scaled = scaler.transform(data)
    predictions = (model.predict(data_scaled) > 0.5).astype("int32")
    return predictions

# Example usage
if __name__ == "__main__":
    # Load data from CSV file
    data_path = 'Jacob_keypress_data.csv'
    df = pd.read_csv(data_path)

    # if User == 'Jacob', set label to 1, otherwise set label to 0
    labels = np.where(df['User'] == 'Jacob', 1, 0)
    
    data = df[['Delta Time (ms)']].values

    # Train the model
    trained_model, trained_scaler = train_model(data, labels)

    # Save the trained model and scaler
    save_model(trained_model, trained_scaler)

    # Optionally, load the model and scaler
    # loaded_model, loaded_scaler = load_model()

    # Example of predicting with the loaded model
    # new_data_path = 'new_keystroke_data.csv'
    # new_df = pd.read_csv(new_data_path)
    # new_data = new_df[['Delta Time (ms)']].values

    # predictions = predict(loaded_model, loaded_scaler, new_data)

   # print("Predictions for the new data:")
    # print(predictions)