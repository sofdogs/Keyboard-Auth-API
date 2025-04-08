import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout



# Load the dataset
df = pd.read_csv('Jacob_keypress_data.csv')

# Extract relevant columns
data = df[['Key', 'Delta Time (ms)']]

# Convert the 'Key' column to numeric values (if it's not already)
data['Key'] = pd.factorize(data['Key'])[0]

# Normalize the 'DeltaTime' column
scaler = StandardScaler()
data['Delta Time (ms)'] = scaler.fit_transform(data['Delta Time (ms)'].values.reshape(-1, 1))

# Define a function to create sequences of data
def create_sequences(data, sequence_length=50):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence.values)
    return np.array(sequences)

# Create sequences of data
sequences = create_sequences(data)

# Split the data into training and testing sets
train_size = int(len(sequences) * 0.8)
train_data, test_data = sequences[:train_size], sequences[train_size:]

# Build the one-class neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(50, 2)),  # 1st hidden layer
    Dense(64, activation='relu'),  # 2nd hidden layer
    Dense(64, activation='relu'),  # 3rd hidden layer
    Dense(64, activation='relu'),  # 4th hidden layer
    Dense(64, activation='relu'),  # 5th hidden layer
    Dense(64, activation='relu'),  # 6th hidden layer
    Dense(64, activation='relu'),  # 7th hidden layer
    Dense(64, activation='relu'),  # 8th hidden layer
    Dense(64, activation='relu'),  # 9th hidden layer
    Dense(64, activation='relu'),  # 10th hidden layer
    Flatten(),  # Flatten the output
    Dense(1, activation='sigmoid')  # Output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model on the training data
model.fit(train_data, np.ones(train_data.shape[0]), epochs=10, batch_size=32)

# Evaluate the model on the testing data
loss = model.evaluate(test_data, np.ones(test_data.shape[0]))
print(f'Test Loss: {loss}')





## NEW DATA SET

new_df = pd.read_csv('Mikayla_keypress_data.csv')  # Replace 'new_dataset.csv' with the name of your new dataset file

# Apply the same preprocessing steps
new_data = new_df[['Key', 'Delta Time (ms)']]
new_data['Key'] = pd.factorize(new_data['Key'])[0]
new_data['Delta Time (ms)'] = scaler.transform(new_data['Delta Time (ms)'].values.reshape(-1, 1))

# Create sequences for the new dataset
new_sequences = create_sequences(new_data)

# Evaluate the model on the new dataset
new_data_predictions = model.predict(new_sequences)

# Assuming higher values indicate normal behavior, you might consider setting a threshold
# to classify sequences as normal or anomalous. Adjust this threshold based on your data.

threshold = 0.00000005
anomalies = new_data_predictions < threshold

#print the average anomaly score
print(f'Mikayla anomaly score: {np.mean(new_data_predictions)}')
mikayla = np.mean(new_data_predictions)

# Print the percentage of sequences classified as normal
percentage_normal = (1 - np.mean(anomalies)) * 100
print(f'Percentage of sequences classified as normal: {percentage_normal}%')


## NEW DATA SET

new_df = pd.read_csv('Fred_keypress_data.csv')  # Replace 'new_dataset.csv' with the name of your new dataset file

# Apply the same preprocessing steps
new_data = new_df[['Key', 'Delta Time (ms)']]
new_data['Key'] = pd.factorize(new_data['Key'])[0]
new_data['Delta Time (ms)'] = scaler.transform(new_data['Delta Time (ms)'].values.reshape(-1, 1))

# Create sequences for the new dataset
new_sequences = create_sequences(new_data)

# Evaluate the model on the new dataset
new_data_predictions = model.predict(new_sequences)

# Assuming higher values indicate normal behavior, you might consider setting a threshold
# to classify sequences as normal or anomalous. Adjust this threshold based on your data.

threshold = 0.00000005
anomalies = new_data_predictions < threshold

#print the average anomaly score
print(f'Jacob anomaly score: {np.mean(new_data_predictions)}')
jacob = np.mean(new_data_predictions)

# Print the percentage of sequences classified as normal
percentage_normal = (1 - np.mean(anomalies)) * 100
print(f'Percentage of sequences classified as normal: {percentage_normal}%')

# Difference between the two scores
difference = jacob - mikayla

# Print who's more likely to be the user
if difference > 0:
    print("Jacob is more likely to be the user")
else:
    print("Mikayla is more likely to be the user")