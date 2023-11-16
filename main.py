import sys
import json
import time
import pygame
#from pygame.math import Vector2
from constants import COLORS, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from models import all_sprites, species
from generate_world import generate_world
from utils import update_tracker, log_tracker
from globals import APP

pygame.init()

# Create the program window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyLife2")

# Clear the screen
def clear_screen():
    window.fill(COLORS["BLACK"])
    #window.fill((4, 217, 255)) # Light blue
    #window.fill((255, 255, 255)) # White

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

# Update simulation logic here
def update():
    all_sprites.update()

# Draw simulation elements here
def draw():
    all_sprites.draw(window)

# Main program loop
def main_loop():
    clock = pygame.time.Clock()
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
    generate_world()
    update_tracker(species, all_sprites)
    # Create the tracker file
    tracker_file_dict = {
        "track_data": []
    }
    file_timestamp = APP['sim_start_time'].strftime("%H:%M:%S %d-%m-%y")
    folder = "sim_statistics_data"
    filename = f'sim_tracker_{file_timestamp}.json'
    path = f'{folder}/{filename}'
    APP["current_log_file"] = path
    with open(path, 'w') as file:
        json.dump(tracker_file_dict, file, indent=4)
    # Write the initial tracking data
    log_tracker()
    # Start the main loop
    main_loop()

start()
