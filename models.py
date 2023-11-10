import math
import random
import numpy as np
import uuid
import pygame
from constants import *
from utils import calc_new_paramater, calc_distance

# Sprite groups
all_sprites = pygame.sprite.Group()

class Organism(pygame.sprite.Sprite):
    def __init__(self, params, position):
        # Instance specific
        self.instance_id = str(uuid.uuid4())
        self.params = params
        self.current_params = {
            "width": calc_new_paramater(params["min_width"], params["max_width"]),
            "height": calc_new_paramater(params["min_height"], params["max_height"]),
            "current_vision_range": calc_new_paramater(params["min_vision_range"], params["max_vision_range"]),
            "current_speed": 0,
            "current_energy": params["max_energy"] * 0.5,
            "current_energy_loss_rate": calc_new_paramater(params["min_energy_loss_rate"], params["max_energy_loss_rate"]),
            "current_reproduction_rate": calc_new_paramater(params["min_reproduction_rate"], params["max_reproduction_rate"]),
            "lifespan": calc_new_paramater(params["min_lifespan"], params["max_lifespan"])
        }
        # Logic specific
        self.predator_detected_time = 0 # The time at which the predator was detected
        self.creation_time = pygame.time.get_ticks()
        self.last_energy_update_time = pygame.time.get_ticks()
        self.last_reproduction_time = pygame.time.get_ticks()
        # PyGame Specific
        super().__init__()
        self.image = pygame.Surface((self.current_params["width"], self.current_params["height"]))
        self.image.fill(self.params["color"])
        self.rect = self.image.get_rect()
        self.rect.x = position[0] #random.randrange(WINDOW_WIDTH - width)
        self.rect.y = position[1] #random.randrange(WINDOW_HEIGHT - height)
        # OUTDATED ?
        self.move_direction = random.uniform(0, 2 * math.pi)
        self.move_timer = pygame.time.get_ticks() + random.randint(1000, 3000)
    
    # Update itself
    def update(self):
        self.screen_wrap()
        # Reduce the food levels of the organism from hunger
        self.hunger()
        # Check if organism should die
        self.die()
        
    # Decide and choose on a behavior or action to perform based on its current state.
    def decide(self):
        # Possible behaviors:
            # Find and chase a prey
            # Move away from a predator if its close
            # Move in a random direction until a prey or predator is found
            # Reproduce if possible, but there is a penalty of 50% loss of energy upon reproduction
            # Idle (Do not move to conserve energy, because moving decreases the current energy the organism has)
        
        # The organism has 2 goals:
            # Remain alive for as long as possible
                # (current energy is slowly depleted unless the organism eats food)
                # There is a constant threat of being eaten by predators
            # Reproduce as much as possible.
        pass
        
    def screen_wrap(self):
        # Only screen wrap on x-axis
        if self.rect.left > WINDOW_WIDTH:
            self.rect.right = 0
        elif self.rect.right < 0:
            self.rect.left = WINDOW_WIDTH
        # TODO: Remove y-axis screen wrapping
        if self.rect.top > WINDOW_HEIGHT:
            self.rect.bottom = 0
        elif self.rect.bottom < 0:
            self.rect.top = WINDOW_HEIGHT
            
    # Move in a given direction
    # Direction is in an angle from 0 to 2 * math.pi
    def move(self, direction_angle):
        dx = math.cos(direction_angle) * self.current_params["current_speed"]
        dy = math.sin(direction_angle) * self.current_params["current_speed"]
        self.rect.x += dx
        self.rect.y += dy
            
    # Checks if the target is within the vision range
    def is_within_vision(self, target_sprite) -> bool:
        dx = target_sprite.rect.centerx - self.rect.centerx
        dy = target_sprite.rect.centery - self.rect.centery
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance <= self.current_params["current_vision_range"]
    
    # Returns a list of nearby organisms
    def find_nearby_organisms(self) -> list:
        nearby_organisms = [
            organism
            for organism in all_sprites
            if self.is_within_vision(organism)
        ]
        return nearby_organisms
    
    # returns the organism closest to the reference organism
    # also returns the distances of all the compared organisms
    def get_closest_organism(self, organisms: list) -> dict:
        reference_point = [self.rect.centerx, self.rect.centery]
        distances = []
        for organism in organisms:
            point = [organism.rect.centerx, organism.rect.centery]
            # Calculate distances
            distances.append(calc_distance(reference_point, point))
        # Find the index of the point with the shortest distance
        index_of_shortest_distance = np.argmin(distances)
        return {
            "closest_organism": organisms[index_of_shortest_distance],
            "distances": list(zip(organisms, distances))
        }
        
    def is_prey_or_predator(self, organism) -> str:
        if organism.params["name"] in self.params["prey"]:
            return "prey"
        elif organism.params["name"] in self.params["predator"]:
            return "predator"
        
    def eat_prey(self, prey):
        if self.rect.colliderect(prey.rect):
            self.current_params["current_energy"] += prey.current_params["current_energy"] * 0.5 
            prey.kill()
    
    def reproduce(self):
        current_time = pygame.time.get_ticks()
        if self.current_params["current_energy"] >= self.params["max_energy"] and current_time - self.last_reproduction_time > self.current_params["reproduction_rate"]:        
            self.current_params["current_energy"] *= 0.5 # Reproduction penalty reduces energy of current organism by half
            offset = 50 # Position offset of where the new organism should be.
            new_position = [self.rect.centerx + random.randint(-(offset + self.current_params["width"]), offset + self.current_params["width"]),
                            self.rect.centery + random.randint(-(offset + self.current_params["height"]), offset + self.current_params["height"])]
            new_organism = Organism(self.params, new_position) 
            all_sprites.add(new_organism)
            self.last_reproduction_time = current_time
            
    # Reduce the energy level of the organism from hunger
    def hunger(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_energy_update_time > self.current_params["current_energy_loss_rate"]:
            # Energy loss is also based the current moving speed of the organism
            self.current_params["current_energy"] -= 1 + (self.current_params["current_speed"] * 0.8)
            self.last_energy_update_time = current_time

    # When and what happens on death
    def die(self):
        # Death happens if energy level < 0 (starvation) or the organism reached its lifespan
        if self.current_params["current_energy"] <= 0 or pygame.time.get_ticks() - self.creation_time > self.current_params["lifespan"]:
            self.kill()

# Pre-defined organisms

# Just used as a reference
template = {
        "name": "organism",
        "min_width": 1.0,
        "max_width": 2.0,
        "min_height": 1.0,
        "max_height": 2.0,
        "color": (255, 255, 255),
        "min_speed": 0.1,
        "max_speed": 0.2,
        "min_vision_range": 1.0,
        "max_vision_range": 2.0,
        "prey ": [],
        "predators": [],
        "max_energy": 1.0,
        "min_energy_loss_rate": 1.0,
        "max_energy_loss_rate": 2.0,
        "min_reproduction_rate": 10000.0,
        "max_reproduction_rate": 20000.0,
        "min_lifespan": 10000.0,
        "max_lifespan": 20000.0,
}

Plankton = {
    "name": "plankton",
    "min_width": 3.0,
    "max_width": 13.0,
    "min_height": 3.0,
    "max_height": 13.0,
    "color": GREEN,
    "min_speed": 0.1,
    "max_speed": 0.2,
    "min_vision_range": 0.0,
    "max_vision_range": 0.0,
    "prey ": [],
    "predators": [],
    "max_energy": 1.0,
    "min_energy_loss_rate": 0.0,
    "max_energy_loss_rate": 0.0,
    "min_reproduction_rate": 10000.0,
    "max_reproduction_rate": 20000.0,
    "min_lifespan": 21000.0,
    "max_lifespan": 25000.0,
}

Small_fish = {
    "name": "small_fish",
    "min_width": 10.0,
    "max_width": 20.0,
    "min_height": 5.0,
    "max_height": 10.0,
    "color": BLUE,
    "min_speed": 1.5,
    "max_speed": 2.0,
    "min_vision_range": 20.0,
    "max_vision_range": 40.0,
    "prey ": ["plankton"],
    "predators": [],
    "max_energy": 10.0,
    "min_energy_loss_rate": 2000.0,
    "max_energy_loss_rate": 3000.0,
    "min_reproduction_rate": 30000.0,
    "max_reproduction_rate": 60000.0,
    "min_lifespan": 60000.0,
    "max_lifespan": 120000.0,
}

Big_fish = {
    "name": "big_fish",
    "min_width": 15.0,
    "max_width": 10.0,
    "min_height": 30.0,
    "max_height": 20.0,
    "color": RED,
    "min_speed": 1.5,
    "max_speed": 3.0,
    "min_vision_range": 20.0,
    "max_vision_range": 45.0,
    "prey ": ["small_fish"],
    "predators": [],
    "max_energy": 20.0,
    "min_energy_loss_rate": 2000.0,
    "max_energy_loss_rate": 3000.0,
    "min_reproduction_rate": 45000.0,
    "max_reproduction_rate": 90000.0,
    "min_lifespan": 90000.0,
    "max_lifespan": 180000.0,
}

Big_fish = {
    "name": "big_fish",
    "min_width": 10.0,
    "max_width": 15.0,
    "min_height": 15.0,
    "max_height": 20.0,
    "color": RED,
    "min_speed": 1.5,
    "max_speed": 3.0,
    "min_vision_range": 20.0,
    "max_vision_range": 45.0,
    "prey ": ["small_fish"],
    "predators": [],
    "max_energy": 20.0,
    "min_energy_loss_rate": 2000.0,
    "max_energy_loss_rate": 3000.0,
    "min_reproduction_rate": 45000.0,
    "max_reproduction_rate": 90000.0,
    "min_lifespan": 90000.0,
    "max_lifespan": 180000.0,
}

Shark = {
    "name": "shark",
    "min_width": 20.0,
    "max_width": 50.0,
    "min_height": 30.0,
    "max_height": 35.0,
    "color": (230, 230, 230),
    "min_speed": 1.5,
    "max_speed": 4.0,
    "min_vision_range": 40.0,
    "max_vision_range": 50.0,
    "prey ": ["small_fish, big_fish"],
    "predators": [],
    "max_energy": 25.0,
    "min_energy_loss_rate": 3000.0,
    "max_energy_loss_rate": 3500.0,
    "min_reproduction_rate": 180000.0,
    "max_reproduction_rate": 240000.0,
    "min_lifespan": 300000.0,
    "max_lifespan": 480000.0,
}

Orca = {
    "name": "orca",
    "min_width": 20.0,
    "max_width": 50.0,
    "min_height": 30.0,
    "max_height": 35.0,
    "color": (230, 230, 230),
    "min_speed": 1.5,
    "max_speed": 4.0,
    "min_vision_range": 40.0,
    "max_vision_range": 50.0,
    "prey ": ["shark"],
    "predators": [],
    "max_energy": 25.0,
    "min_energy_loss_rate": 3000.0,
    "max_energy_loss_rate": 3500.0,
    "min_reproduction_rate": 180000.0,
    "max_reproduction_rate": 240000.0,
    "min_lifespan": 300000.0,
    "max_lifespan": 480000.0,
}

Whale = {
    "name": "whale",
    "min_width": 75.0,
    "max_width": 150.0,
    "min_height": 30.0,
    "max_height": 40.0,
    "color": MEDIUM_BLUE,
    "min_speed": 1.0,
    "max_speed": 2.0,
    "min_vision_range": 40.0,
    "max_vision_range": 50.0,
    "prey ": ["plankton"],
    "predators": [],
    "max_energy": 40.0,
    "min_energy_loss_rate": 3000.0,
    "max_energy_loss_rate": 3500.0,
    "min_reproduction_rate": 240000.0,
    "max_reproduction_rate": 300000.0,
    "min_lifespan": 480000.0,
    "max_lifespan": 600000.0,
}
