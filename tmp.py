import random
import json
import numpy as np
from organisms import Organisms

def calc_new_paramater(min: float, max: float) -> float:
    midpoint = (min + max) * 0.5
    offset = midpoint * 0.135 # Offset is 13.5% of midpoint
    randomizer = round(random.uniform(-offset, offset), 2)
    new_param = midpoint + randomizer
    print("midpoint", midpoint)
    print("randomizer: ", randomizer)
    return new_param

# for i in range(10):
#     result = calc_new_paramater(0.0, 0.0)
#     print(result)
    
    
# Generates a random number between min and max,
# With a higher probability around the midpoint between min and max
def random_with_bias(min: float, max: float, bias_factor=0.7) -> float:
    midpoint = (min + max) * 0.5
    spread = 3.0
    # Calculate standard deviation based on the distance from the midpoint
    std_dev = (max - midpoint) / spread  
    # Generate a random number from a normal distribution
    random_number = np.random.normal(loc=midpoint, scale=std_dev)
    # Clip the result to make sure it's within the specified range [min, max]
    print("midpoint", midpoint)
    return np.clip(random_number, min, max)

# a = 10.0
# b = 30.0
# numbers = []
# for i in range(1000):
#     number = random_with_bias(a, b)
#     numbers.append(number)
#     print(number)

# Returns the distance between 2 points
def calc_distance(point1: list, point2: list) -> int:
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    # We use max(1, dist) because the min size is 1 pixel
    return max(1, distance)


def shortest_distance_to_reference(reference_point, input_points):
    distances = []
    for point in input_points:
        print("point: ", point)
        distances.append(calc_distance(reference_point, point))
    print("distances: ", distances)
    index_of_shortest_distance = np.argmin(distances)
    closest_point = input_points[index_of_shortest_distance]
    return closest_point

# points_list = [[1, 2], [3, 4], [5, 6], [7, 8]]
# reference_point = [4, 5]

# closest_point = shortest_distance_to_reference(reference_point, points_list)


# print(f"Closest Point: {closest_point}")

# Save pre-defined organisms to a .json file
def save_organism_definitions(organisms: dict):
    filename = "organisms.json"
    print("Saving to file: ", filename)
    with open(filename, 'w') as json_file:
        json.dump(organisms, json_file, indent=4)
    print("File saved.")

save_organism_definitions(Organisms)
