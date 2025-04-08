import os
import time
import csv
from pynput import keyboard

# Function to handle keypress events
global delta_time_ms
global duration_ms
global prev_time_ms



# Initialize variables

delta_time_ms = 0
duration_ms = 0
prev_time_ms = 0
keys_being_pressed = {}

def on_key_press(key):
    global delta_time_ms
    global keys_being_pressed
    global prev_time_ms

    try:
        if key == keyboard.Key.shift:
            return 
        if key == keyboard.Key.esc:
            return False
        
        current_time_ms = time.time() * 1000  # Convert current time to milliseconds
        
        keys_being_pressed[key] = current_time_ms  # Add the key to the dictionary
        delta_time_ms = current_time_ms - prev_time_ms

        # If more than 2 seconds have passed, reset the time
        if delta_time_ms > 5000:
            delta_time_ms = 0

        prev_time_ms = current_time_ms


    except AttributeError:
        # Handle special keys that don't have a 'char' attribute
        pass


def on_key_release(key):
    if key == keyboard.Key.shift:
        return
    
    global duration_ms
    global keys_being_pressed
    global delta_time_ms
    global prev_time_ms

    if key in keys_being_pressed:
        press_time_ms = keys_being_pressed[key]
        release_time_ms = time.time() * 1000  # Convert current time to milliseconds
        duration_ms = release_time_ms - press_time_ms
        del keys_being_pressed[key]  # Remove the key from the dictionary
        print(f"Key: {key}, Time Since Previous Key (ms): {delta_time_ms:.2f}ms, Duration (ms): {duration_ms:.2f}ms")

        # Write the keypress information to a CSV file
        with open(f'{user_name}_keypress_data.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([key, delta_time_ms, duration_ms])




# Get the name of the user
user_name = input("Enter your name: ")

# Initialize the previous keypress time to the current time
prev_time_ms = time.time() * 1000

# Create a listener for keyboard events
with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
    listener.join()