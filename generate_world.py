import math
import random
import datetime
import pygame
from models import Organism, all_sprites, species, pre_defined_organisms as Organisms
from constants import WINDOW_WIDTH, WINDOW_HEIGHT
from globals import APP
from utils import generate_exponential_numbers

def generate_world():
    print("Initializing the simulation world.")
    # Add tracking data to the simulation
    APP['sim_start_time'] = datetime.datetime.now()
    APP['sim_start_time_ticks'] = pygame.time.get_ticks()
    for specie in species:
        APP['tracker'][f"total_{specie}"] = 0
    APP['tracker']['total_species'] = len(Organisms)
    APP['tracker']['total_organisms'] = 0
    APP['tracker']['total_current_energy'] = 0.0
    APP['tracker']['average_current_energy'] = 0.0
    
    # Add initial organisms
    BASE_PLANKTON = 1000
    
    # Add plankton with exponential distribution
    # With most plankton at the top of the screen
    # and least plankton at the bottom of the screen
    y_values = generate_exponential_numbers(0, WINDOW_HEIGHT, BASE_PLANKTON)
    for i in range(BASE_PLANKTON):
        y = random.choice(y_values)
        del y_values[0]
        #random_position = [random.randrange(WINDOW_WIDTH - Plankton["max_width"]), random.randrange(WINDOW_WIDTH - Plankton["max_height"])]
        plankton = Organism(Organisms["Plankton"], [random.randrange(WINDOW_WIDTH - Organisms["Plankton"]["max_width"]), y])
        #plankton = Organism(Plankton, random_position)
        all_sprites.add(plankton)
        
    # Add small fishes
    for i in range(math.ceil(BASE_PLANKTON * 0.06)):
        random_position = [random.randrange(WINDOW_WIDTH - Organisms["Small_fish"]["max_width"]), random.randrange(WINDOW_HEIGHT - Organisms["Small_fish"]["max_height"])]
        small_fish = Organism(Organisms["Small_fish"], random_position)
        all_sprites.add(small_fish)

    # Add big fishes
    for i in range(math.ceil(BASE_PLANKTON * 0.004)):
        random_position = [random.randrange(WINDOW_WIDTH - Organisms["Big_fish"]["max_width"]), random.randrange(WINDOW_HEIGHT - Organisms["Big_fish"]["max_height"])]
        big_fish = Organism(Organisms["Big_fish"], random_position)
        all_sprites.add(big_fish)

    # Add sharks
    for i in range(math.ceil(BASE_PLANKTON * 0.001)):
        random_position = [random.randrange(WINDOW_WIDTH - Organisms["Shark"]["max_width"]), random.randrange(WINDOW_HEIGHT - Organisms["Shark"]["max_height"])]
        shark = Organism(Organisms["Shark"], random_position)
        all_sprites.add(shark)

    # Add orcas
    for i in range(math.ceil(BASE_PLANKTON * 0.0004)):
        random_position = [random.randrange(WINDOW_WIDTH - Organisms["Orca"]["max_width"]), random.randrange(WINDOW_HEIGHT - Organisms["Orca"]["max_height"])]
        orca = Organism(Organisms["Orca"], random_position)
        all_sprites.add(orca)

    # Add whales
    for i in range(math.ceil(BASE_PLANKTON * 0.0020)):
        random_position = [random.randrange(WINDOW_WIDTH - Organisms["Whale"]["max_width"]), random.randrange(WINDOW_HEIGHT - Organisms["Whale"]["max_height"])]
        whale = Organism(Organisms["Whale"], random_position)
        all_sprites.add(whale)
    
    print("-------")
    print("Initializiation of sim world done.")
    
    
