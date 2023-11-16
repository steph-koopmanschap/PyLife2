import random
import json
import numpy as np
import inspect

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

#save_organism_definitions(Organisms)

def getClassMethods():
    methods = [method for method in dir(Animal) if callable(getattr(Animal, method)) and not method.startswith("__")]

def functionOne(function):
    return function.__name__
    
    
class MyClass:
    def functionTwo(self):
        print(functionOne(self.functionTwo))
        
myclass = MyClass()
myclass.functionTwo()

# def checkBehaviorAllowed(func):
#     def wrapper(*args, **kwargs):
#         if func
#         print(f"Hello from {func.__name__}")
#         #return func(*args, **kwargs)
#     return wrapper



class Animal:
    def __init__(self, params):
        self.params = params
        self.available_methods = [method for method in dir(self) if "__" not in method]
        print("available_methods", self.available_methods)
        
    def is_behavior_allowed(self): #function_name
        if self.available_methods in self.params['behaviors']:
            return True
        return False
    
    def decide():
        pass

    def make_sound(self):
        print(f"{self.params['name']}: {self.params['sound']}")

    def fly(self):
        #if self.is_behavior_allowed(inspect.currentframe().f_code.co_name):
        if self.is_behavior_allowed():
            print(f"{self.params['name']}: I am flying")

    def walk(self):
        if self.is_behavior_allowed():
            print(f"{self.params['name']}: I am walking")

bird = {
    "name": 'bird', 
    "sound": "chirp",
    "behaviors": ["fly"]
}

dog = {
    "name": 'dog', 
    "sound": "woof",
    "behaviors": ["walk"]
}

cat = {
    "name": 'cat', 
    "sound": "meow",
    "behaviors": ["fly"]
}

Cat = Animal(cat)
Dog = Animal(dog)

Cat.make_sound()
Dog.make_sound()
Cat.fly()
Cat.walk()
Dog.fly()
Dog.walk()