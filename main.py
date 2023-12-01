import sys
import os
import json
import random
import pygame
import numpy as np
#from pygame.math import Vector2
from constants import COLORS, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from models import all_sprites, species
from environment import Environment
from generate_world import generate_world
from utils import update_tracker, log_tracker
from globals import APP

rng = np.random.default_rng() # Numpy's random number generator
pygame.init()
environment = Environment() # Create the simulation's environment
# Create the program window and screen(drawing surface)
APP['screen'] = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyLife2")

# Clear the screen
def clear_screen():
    # Night time
    if environment.current_environment['is_night'] == True:
        APP['screen'].fill(COLORS["BLACK"])
    # Day time
    else:
        APP['screen'].fill((4, 217, 255)) # Light blue
    #screen.fill((255, 255, 255)) # White
    
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
        # 'S' key press.
        if event.key == pygame.K_s:
            if APP['selected_organism']:
                APP['selected_organism'].save_AI_params()
        # 'I' key press
        if event.key == pygame.K_i:
            if APP['selected_organism']:
                APP['selected_organism'].save_info()
    # Left mouse button clicked
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        # Select an organism  
        mouse_pos = pygame.mouse.get_pos()
        clicked_organisms = [organism for organism in all_sprites if organism.rect.collidepoint(mouse_pos)]
        if clicked_organisms and len(clicked_organisms) > 0:
            APP['selected_organism'] = clicked_organisms[0] 
            print(f"Selected organism: {APP['selected_organism'].params['species']} - {APP['selected_organism'].instance_id}")
            print("Press 'S' to save the AI trained model of this organism.")
            print("Press 'I' to save the organism's state and 'dna'.")
        

# Update simulation logic here
def update(now):
    all_sprites.update() # Update the organisms / sprites
    environment.update(now) # Update the environment

# Draw simulation elements here and rendering logic
def draw():
    all_sprites.draw(APP['screen'])
    # Update the display
    pygame.display.update()

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
        now = pygame.time.get_ticks()

        update(now)
        clear_screen()
        draw()
        
        # Get the current frame rate
        current_fps = clock.get_fps()
        # If the frame rate drops we will kill off purge_percentage of the most abudant species
        if current_fps < framerate_purge_level and pygame.time.get_ticks() - time_since_last_purge > purge_rate:
            print(f"Framerate dropping... Current FPS: {current_fps} ... Deleting {purge_percentage*100}% of the most abundant species.")
            purge(purge_percentage)
            time_since_last_purge = pygame.time.get_ticks()
                
        # Log statistics to file every APP['logging_rate'] seconds
        elapsed_time = now - time_since_last_logging
        if elapsed_time >= APP['logging_rate']:
            update_tracker(species, all_sprites)
            log_tracker()
            # Update the time_since_last_logging for the next interval
            time_since_last_logging = pygame.time.get_ticks()
            
        clock.tick(FPS)

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
    filename = filename.replace(":", "-") # We need to remove ':' characters from the time to make Windows happy.
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
