import math
import random
import uuid
import pygame
from constants import WINDOW_WIDTH, WINDOW_HEIGHT
from utils import calc_new_paramater, calc_distance

# Sprite groups
all_sprites = pygame.sprite.Group()

class Organism(pygame.sprite.Sprite):
    def __init__(self, params, position):
        # Instance specific
        self.instance_id = str(uuid.uuid4())
        # Initial parameters of the organism are like the "genetic material" of this organism
        # Upon reproduction the organism passes these parameters to the next organism
        self.params = params
        # Current state of the organism
        # These are internal variables that can potentially change over time.
        # Note: We probably can only use floats and integers here
        self.current_state = {
            "width": calc_new_paramater(params["min_width"], params["max_width"]),
            "height": calc_new_paramater(params["min_height"], params["max_height"]),
            "current_vision_range": calc_new_paramater(params["min_vision_range"], params["max_vision_range"]),
            "current_speed": 0.0,
            "current_energy": params["max_energy"] * 0.5,
            "current_energy_loss_rate": calc_new_paramater(params["min_energy_loss_rate"], params["max_energy_loss_rate"]),
            "current_reproduction_rate": calc_new_paramater(params["min_reproduction_rate"], params["max_reproduction_rate"]),
            "lifespan": calc_new_paramater(params["min_lifespan"], params["max_lifespan"]),
            "predator_detected_time": 0.0, # The time at which the predator was detected
            "creation_time": pygame.time.get_ticks(),
            "last_energy_update_time": pygame.time.get_ticks(),
            "last_reproduction_time": pygame.time.get_ticks(),
            "closest_predator_distance": -1.0, 
            "closest_prey_distance": -1.0,
            "closest_same_species_distance": -1.0
        }
        # References to the class instances of the closest organisms
        self.closest_predator = None
        self.closest_prey = None
        self.closest_same_species = None
        self.reproduction_pentalty = 0.5
        # PyGame Specific
        super().__init__()
        self.image = pygame.Surface((self.current_state["width"], self.current_state["height"]))
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
        # See around itself to detect other organisms
        self.vision()
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
            
    # The organism "sees" around itself to detect other organisms
    def vision(self):
        nearby_organisms = self.find_nearby_organisms()
        organisms_distances = self.get_organisms_distances(nearby_organisms)
        # Extract the closest predator, prey, and same species data
        closest_predators = next(([org, dist] for org, dist in organisms_distances if org.params["species"] in self.params["predators"]), None)
        closest_preys = next(([org, dist] for org, dist in organisms_distances if org.params["species"] in self.params["prey"]), None)
        closest_same_speciess = next(([org, dist] for org, dist in organisms_distances if org.params["species"] == self.params["species"]), None)
        # Update state
        self.current_state["closest_predator_distance"] = closest_predators[1] if closest_predators else -1.0
        self.current_state["closest_prey_distance"] = closest_preys[1] if closest_preys else -1.0
        self.current_state["closest_same_species_distance"] = closest_same_speciess[1] if closest_same_speciess else -1.0
        self.closest_predator = closest_predators[0] if closest_predators else None
        self.closest_prey = closest_preys[0] if closest_preys else None
        self.closest_same_species = closest_same_speciess[0] if closest_same_speciess else None
    
    # This is work in progress function not yet implemented
    # This function allows the organism to communicate with organisms of the same species
    def communicate():
        pass
            
    # Move in a given direction
    # Direction is in an angle from 0 to 2 * math.pi
    def move(self, direction_angle):
        dx = math.cos(direction_angle) * self.current_state["current_speed"]
        dy = math.sin(direction_angle) * self.current_state["current_speed"]
        self.rect.x += dx
        self.rect.y += dy
            
    # Checks if the target is within the vision range
    def is_within_vision(self, target_sprite) -> bool:
        dx = target_sprite.rect.centerx - self.rect.centerx
        dy = target_sprite.rect.centery - self.rect.centery
        # distance = (dx ** 2 + dy ** 2) ** 0.5 # OLD Code
        # We calculate the squared distance and the compare to vision range squared
        # So we don't have to do a square root to get the actual distance
        # This increase performance because square roots are difficult to calculate
        squared_distance = dx ** 2 + dy ** 2
        vision_range_squared = self.current_state["current_vision_range"] ** 2
        #return distance <= self.current_state["current_vision_range"] # OLD code
        return squared_distance <= vision_range_squared
    
    # Returns a list of nearby organisms
    def find_nearby_organisms(self) -> list:
        nearby_organisms = [
            organism
            for organism in all_sprites
            if self.is_within_vision(organism)
        ]
        return nearby_organisms
    
    # Returns a list of tuples where
    # sorted_distances[0][0] is the closest organism and sorted_distances[0][1] is the distance to the closest organism
    # Sorted from low to high
    def get_organisms_distances(self, organisms: list) -> list:
        reference_point = [self.rect.centerx, self.rect.centery]
        distances = []
        for organism in organisms:
            point = [organism.rect.centerx, organism.rect.centery]
            # Calculate distances
            distances.append(calc_distance(reference_point, point))
        # Sort the list of tuples based on the second element (distance)
        # So that the closest organism is distances[0][0]
        zipped_distances = list(zip(organisms, distances))
        sorted_distances = sorted(zipped_distances, key=lambda x: x[1])
        return sorted_distances
        
    def is_prey_or_predator(self, organism) -> str:
        if organism.params["species"] in self.params["prey"]:
            return "prey"
        elif organism.params["species"] in self.params["predator"]:
            return "predator"
        
    def eat_prey(self, prey):
        if self.rect.colliderect(prey.rect):
            self.current_state["current_energy"] += prey.current_state["current_energy"] * 0.5 
            prey.kill()
    
    def reproduce(self):
        current_time = pygame.time.get_ticks()
        if self.current_state["current_energy"] >= self.params["max_energy"] and current_time - self.last_reproduction_time > self.current_state["reproduction_rate"]:        
            # Reproduction penalty reduces energy of current organism by a certein percentage
            self.current_state["current_energy"] *= self.reproduction_pentalty
            offset = 50 # Position offset of where the new organism should be.
            new_position = [self.rect.centerx + random.randint(-(offset + self.current_state["width"]), offset + self.current_state["width"]),
                            self.rect.centery + random.randint(-(offset + self.current_state["height"]), offset + self.current_state["height"])]
            new_organism = Organism(self.params, new_position) 
            all_sprites.add(new_organism)
            self.last_reproduction_time = current_time
            
    # Reduce the energy level of the organism from hunger
    def hunger(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.current_state["last_energy_update_time"] > self.current_state["current_energy_loss_rate"]:
            # Energy loss is also based the current moving speed of the organism
            self.current_state["current_energy"] -= 1 + (self.current_state["current_speed"] * 0.8)
            self.current_state["last_energy_update_time"] = current_time

    # When and what happens on death
    def die(self):
        # Death happens if energy level < 0 (starvation) or the organism reached its lifespan
        if self.current_state["current_energy"] <= 0 or pygame.time.get_ticks() - self.current_state["creation_time"] > self.current_state["lifespan"]:
            self.kill()
