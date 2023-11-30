import sys
import os
import json
import random
import pygame
import numpy as np
#from pygame.math import Vector2
from constants import COLORS, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from models import all_sprites, species
from generate_world import generate_world
from utils import update_tracker, log_tracker
from globals import APP

rng = np.random.default_rng() # Numpy's random number generator
pygame.init()
# Create the program window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyLife2")

# Clear the screen
def clear_screen():
    window.fill(COLORS["BLACK"])
    #window.fill((4, 217, 255)) # Light blue
    #window.fill((255, 255, 255)) # White
    
def purge(purge_percentage):    
    # Check which species is the most abundant
    organism_counts = {}
    for specie in species:
        organism_counts[specie] = (APP['tracker'][f"total_{specie}"]) #specie.replace("total_", "")
    most_abudant_species = max(organism_counts, key=organism_counts.get)
    print(f"Most abundant species: {most_abudant_species} count: {organism_counts[most_abudant_species]}")
    all_organisms_most_abundant = []
    for organism in all_sprites:
        if organism.params["species"] == most_abudant_species:
            all_organisms_most_abundant.append(organism)
    # Randomize the organisms to be deleted
    rng.shuffle(all_organisms_most_abundant)
    # only get a percentage of organisms
    organisms_to_delete = all_organisms_most_abundant[:int(len(all_organisms_most_abundant) * purge_percentage)]
    for organism in organisms_to_delete:
        organism.destroy()
        #organism_to_delete.kill()
    print(f"Done deleting {len(organisms_to_delete)} ({purge_percentage*100}%) of {most_abudant_species}")
    

def handle_input(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
        
    if event.type == pygame.KEYDOWN:
        # Check if the key pressed is 'L'
        if event.key == pygame.K_l:
            # Force logging
            update_tracker(species, all_sprites)
            log_tracker()
            print("Force logged to file.")

# Update simulation logic here
def update():
    all_sprites.update()

# Draw simulation elements here
def draw():
    all_sprites.draw(window)

# Main program loop
def main_loop():
    clock = pygame.time.Clock()
    # To which point the framerate need to drop below, to purge organisms from the game for maintaining smooth framerate
    framerate_purge_level = 5.0
    # Purge every x seconds if framerate below framerate_purge_level
    purge_rate = 3500 
    # How much percentage of the species to purge
    purge_percentage = 0.7
    # time_since_last_purge is used to track when was the last time the framerate droppepd below
    # the frame_rate purge level.
    time_since_last_purge = pygame.time.get_ticks()
    time_since_last_logging = pygame.time.get_ticks()
    while True:
        for event in pygame.event.get():
            handle_input(event)

        update()
        clear_screen()
        draw()

        # Update the display
        pygame.display.update()
        clock.tick(FPS)
        # Get the current frame rate
        current_fps = clock.get_fps()
        # If the frame rate drops we will kill off purge_percentage of the most abudant species
        if current_fps < framerate_purge_level and pygame.time.get_ticks() - time_since_last_purge > purge_rate:
            print(f"Framerate dropping... Current FPS: {current_fps} ... Deleting {purge_percentage*100}% of the most abundant species.")
            purge(purge_percentage)
            time_since_last_purge = pygame.time.get_ticks()
                
        # Log statistics to file every 5 seconds
        elapsed_time = pygame.time.get_ticks() - time_since_last_logging
        if elapsed_time >= APP['logging_rate']:
            update_tracker(species, all_sprites)
            log_tracker()
            # Update the time_since_last_logging for the next interval
            time_since_last_logging = pygame.time.get_ticks()

# Start the simulation
def start():
    print("Welcome to PyLife2")
    print("Press 'L' to force write statistics data to the log")
    print("Loaded species: ", species)
    generate_world()
    update_tracker(species, all_sprites)
    # Create the tracker file
    tracker_file_dict = {
        "track_data": []
    }
    file_timestamp = APP['sim_start_time'].strftime("%H:%M:%S %d-%m-%y")
    folder = "sim_statistics_data"
    filename = f'sim_tracker_{file_timestamp}.json'
    path = path = os.path.join(folder, filename)
    APP["current_log_file"] = path
    with open(path, 'w') as file:
        json.dump(tracker_file_dict, file, indent=4)
    # Write the initial tracking data
    log_tracker()
    # Start the main loop
    main_loop()

if __name__ == "__main__":
    start()
