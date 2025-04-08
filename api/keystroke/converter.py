import pandas as pd
import json

# Load CSV file into a DataFrame
csv_file_path = 'Sofia_keypress_data.csv'
df = pd.read_csv(csv_file_path, header=None)

# Rename columns
df.columns = ['Key', 'Delta Time (ms)', 'Duration (ms)']

# Create list of keys
keys = df['Key'].values


# Create list of delta times
delta_times = df['Delta Time (ms)'].values

# Create list of durations
durations = df['Duration (ms)'].values

# Convert DataFrame to JSON format
json_data = {
    "data": {
        "Key": keys.tolist(),
        "Delta": delta_times.tolist(),
        "Duration": durations.tolist()
    }
}

# Convert dictionary to JSON string
json_string = json.dumps(json_data, indent=4)

with open('Jacob2_keypress_data.json', 'w') as json_file:
    json_file.write(json_string)