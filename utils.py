import random
import numpy as np

# Calculate a new parameter for a new organism based on previous min and max params of previous organism
def calc_new_paramater(min: float, max: float) -> float:
    midpoint = (min + max) * 0.5
    offset = midpoint * 0.135 # Offset is 13.5% of midpoint
    randomizer = round(random.uniform(-offset, offset), 2)
    new_param = midpoint + randomizer
    return new_param

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