import sys
import json
import pygame
#from pygame.math import Vector2
from constants import COLORS, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from models import all_sprites, species
from generate_world import generate_world
from utils import update_tracker, log_tracker
from globals import APP

pygame.init()
clock = pygame.time.Clock()

# Create the program window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyLife2")

# Clear the screen
def clear_screen():
    window.fill(COLORS["BLACK"])
    #window.fill((4, 217, 255)) # Light blue
    #window.fill((255, 255, 255)) # White

# Update simulation logic here
def update():
    all_sprites.update()

# Draw simulation elements here
def draw():
    all_sprites.draw(window)

# Main program loop
def main_loop():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        update()
        clear_screen()
        draw()

        # Update the display
        pygame.display.update()
        clock.tick(FPS)

# Start the simulation
def start():
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