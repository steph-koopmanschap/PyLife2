import pygame
#from pygame.math import Vector2
import sys
from constants import BLACK, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from models import all_sprites
from generate_world import generate_world

pygame.init()
clock = pygame.time.Clock()

# Create the game window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PyLife2")

# Clear the screen
def clear_screen():
    window.fill(BLACK)
    #window.fill((4, 217, 255)) # Light blue
    #window.fill((255, 255, 255))

# Update game logic here
def update():
    all_sprites.update()

# Draw game elements here
def draw():
    all_sprites.draw(window)

# Game loop
def game_loop():
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

def start():
    generate_world()
    game_loop()

start()