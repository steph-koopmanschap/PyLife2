import random
import json
import datetime
import copy
import pygame
import numpy as np
from globals import APP

# Generates a random number between min and max,
# With a higher probability around the midpoint between min and max
def random_with_bias(min: float, max: float, bias_factor=1.0) -> float:
    # Exit early if min and max are 0
    if min == 0.0 and max == 0.0:
        return 0.0
    midpoint = (min + max) * 0.5
    spread = 3.0
    # Calculate standard deviation based on the distance from the midpoint
    # A bias factor of 1.0 results in normal distribution.
    # A bias factor of 0.5 results in flat distr
    # A bias factor of 0.1 results in an inverted normal distr
    std_dev = (max - midpoint) / (spread * bias_factor) 
    # Generate a random number from a normal distribution
    random_number = np.random.normal(loc=midpoint, scale=std_dev)
    # Clip the result to make sure it's within the specified range [min, max]
    return np.clip(random_number, min, max)

# Returns the distance between 2 points
def calc_distance(point1: list, point2: list) -> int:
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    # We use max(1, dist) because the min size is 1 pixel
    return max(1, distance)
    
def generate_exponential_numbers(min_value, max_value, total_numbers, desired_range=800):
    # Generate random numbers with an exponential distribution
    y_values = np.random.randint(min_value, max_value, total_numbers)
    numbers = np.array(np.exp(-y_values ** 2 / max_value ** 2) * max_value, dtype=int)
    
    # Map values to the desired range
    min_number = np.min(numbers)
    max_number = np.max(numbers)
    mapped_numbers = ((numbers - min_number) * desired_range / (max_number - min_number)).astype(int)
    
    # Sort and reverse the mapped numbers
    mapped_numbers = np.sort(mapped_numbers)[::-1]

    return mapped_numbers.tolist()

# Retrieve instance variables as a dictionary
def get_class_instance_vars(class_instance):
    instance_variables_dict = class_instance.__dict__
    return instance_variables_dict

# Load pre-defined organisms from .json file.
def load_organism_definitions() -> dict:
    filename = "organisms.json"
    print("Loading from file: ", filename)
    with open(filename, 'r') as json_file:
        organisms = json.load(json_file)
    print("File loaded.")
    return organisms

# Save pre-defined organisms to a .json file
def save_organism_definitions(organisms: dict):
    filename = "organisms.json"
    print("Saving to file: ", filename)
    with open(filename, 'w') as json_file:
        json.dump(organisms, json_file, indent=4)
    print("File saved.")

# Update statistics in tracker
def update_tracker(species, all_sprites):
    for key in APP['tracker']:
        specie = key[key.find("_") + 1:]
        if specie in species:
            APP['tracker']['total_organisms'] += APP['tracker'][key]
        elif key == "total_current_energy":
            for org in all_sprites:
                APP['tracker']['total_current_energy'] += org.current_state["current_energy"]
        elif key == "average_current_energy":
            APP['tracker']['average_current_energy'] = APP['tracker']['total_current_energy'] / len(all_sprites)

# Save statistics to a file
def log_tracker():
    tracker = APP['tracker']
    # Load the existing JSON data from the file
    with open(APP["current_log_file"], 'r') as file:
        data = json.load(file)
    # Create a new entry for tracker log file
    new_entry = copy.deepcopy( APP['tracker'])
    new_entry["time_stamp"] = datetime.datetime.now().strftime("%H:%M:%S")
    new_entry["pygame_tick"] = pygame.time.get_ticks()
    data["track_data"].append(new_entry)
    # Write the updated data back to the file
    with open(APP["current_log_file"], 'w') as file:
        json.dump(data, file, indent=4)
    
# OLD FUNCTION. Now replaced by "random_with_bias"
# Calculate a new parameter for a new organism based on previous min and max params of previous organism
def calc_new_paramater(min: float, max: float) -> float:
    midpoint = (min + max) * 0.5
    offset = midpoint * 0.135 # Offset is 13.5% of midpoint
    randomizer = round(random.uniform(-offset, offset), 2)
    new_param = midpoint + randomizer
    return new_param
