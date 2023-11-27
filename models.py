import math
import random
import uuid
import pygame
from globals import APP
from constants import WINDOW_WIDTH, WINDOW_HEIGHT
from utils import load_organism_definitions, random_with_bias, calc_distance

# Sprite groups
all_sprites = pygame.sprite.Group()

pre_defined_organisms = load_organism_definitions()
species = [organism['species'] for organism_definition, organism in pre_defined_organisms.items()]

class Organism(pygame.sprite.Sprite):
    # The position is the initial position where the organism is "created" or "born"
    def __init__(self, params, position):
        # instance_id for keeping track of this instance
        self.instance_id = str(uuid.uuid4())
        # Initial parameters of the organism are like the "genetic material" of this organism
        # Upon reproduction the organism passes these parameters to the next organism
        self.params = params
        # Current state of the organism
        # These are internal variables that can potentially change over time.
        # Note: We probably can only use floats and integers here, because it will be used in the Neural Network
        self.current_state = {
            "width": int(random_with_bias(params["min_width"], params["max_width"])),
            "height": int(random_with_bias(params["min_height"], params["max_height"])),
            # How far the organism can see
            "current_vision_range": random_with_bias(params["min_vision_range"], params["max_vision_range"]),
            # How fast the organism is moving. The current_energy of the organism goes down faster the faster it moves.
            "current_speed": 0.0,
            # The current direction the organism is facing
            "current_direction": 0.0,
            # How much energy reserves the organism has.
            # When current energy =< 0 the organism dies. When current_energy => max_energy the organism can reproduce
            "current_energy": params["max_energy"] * 0.5,
            # After how many time steps the organism loses energy from metabolism. 
            "current_energy_loss_rate": random_with_bias(params["min_energy_loss_rate"], params["max_energy_loss_rate"]),
            # After how many time steps the organism can reproduce
            "current_reproduction_rate": random_with_bias(params["min_reproduction_rate"], params["max_reproduction_rate"]),
            # After how many time steps the organism dies from old age.
            "lifespan": random_with_bias(params["min_lifespan"], params["max_lifespan"]),
            # For how many time steps the organism has been alive
            "current_age": 0.0,
            # At which timestep the last predator was detected
            "predator_detected_time": 0.0,
            # At which time step the organism was born.
            "creation_time": pygame.time.get_ticks(),
            # At which time step the energy was last updated (This is related to current_energy_loss_rate)
            "last_energy_update_time": pygame.time.get_ticks(),
            # At which time step the organism last reproduced. (This is related to current_reproduction_rate)
            "last_reproduction_time": pygame.time.get_ticks(),
            # The distance to the closest predator within current_vision_range
            "closest_predator_distance": -1.0, 
            # The distance to the closest prey within current_vision_range
            "closest_prey_distance": -1.0,
            # The distance to the closest same_species within current vision range
            "closest_same_species_distance": -1.0,
            # How many offspring are produced at the time of next reproduction
            "current_offspring_produced": random_with_bias(params["min_offspring_produced"], params["max_offspring_produced"])
        }    
        # References to the class instances of the closest organisms
        self.closest_predator = None
        self.closest_prey = None
        self.closest_same_species = None
        # reproduction_pentalty is the percentage of how much energy the organism loses upon reproduction
        self.reproduction_pentalty = 0.5
        # The available methods are all the methods of this class.
        self.available_methods = [method for method in dir(self) if "__" not in method]
        # PyGame Specific
        super().__init__()
        self.image = pygame.Surface((self.current_state["width"], self.current_state["height"]))
        self.image.fill(self.params["color"])
        self.rect = self.image.get_rect()
        # self.rect.x and self.rect.y is the actual current position of the organism
        self.rect.x = position[0]
        self.rect.y = position[1]
        # Tracking for statistics
        APP['tracker'][f"total_{self.params['species']}"] += 1
        # OUTDATED ?
        self.move_direction = random.uniform(0, 2 * math.pi)
        self.move_timer = pygame.time.get_ticks() + random.randint(1000, 3000)
        # AI using PPO algorithm
        if self.params["has_brain"]:
            # The possible actions/behaviors the organism can take    
            self.action_space = [
                "change_speed",
                "change_direction",
                "rest",
                "reproduce"
                "flee_from_predator",
                "chase_prey",
            ]
            self.time_since_last_action = pygame.time.get_ticks()
        
    def is_behavior_allowed(self, behavior): 
        return behavior in self.params['behaviors'] # self.available_methods[self.available_methods.index(behavior)]
            #return True
            #return False
        
    # Update itself
    def update(self):
        self.screen_wrap()
        # Reduce the food levels of the organism from hunger
        self.hunger()
        # See around itself to detect other organisms
        self.vision()
        # Check if colliding with other organisms
        self.check_collision()
        # Decide on an action to do
        self.decide()
        # Move the organism
        self.move(radians=False)
        # Check if organism should reproduce
        self.reproduce()
        # Update age
        self.current_state["current_age"] = pygame.time.get_ticks() - self.current_state["creation_time"]
        # Check if organism should die from hunger or old age
        self.die()
        
    # Decide and choose on a behavior or action to perform based on its current state.
    def decide(self):
        # The organism has 2 goals:
        # Remain alive for as long as possible
            # (current energy is slowly depleted unless the organism eats food)
            # There is a constant threat of being eaten by predators
        # Reproduce as much as possible.
        if self.params["has_brain"]:
            #pass
            # Choose an action from the action space
            new_action = random.choice(self.action_space)
            if pygame.time.get_ticks() - self.time_since_last_action > random.randint(200, 5000):
                if new_action == "change_speed":
                    self.change_speed(round(random.uniform(self.params['min_speed'], self.params['max_speed']), 2))
                elif new_action == "change_direction":
                    self.change_direction(round(random.uniform(0.0, 360.0), 2)) 
                elif new_action == "rest":
                    self.rest()
                elif new_action == "flee_from_predator":
                    self.flee_from_predator()
                elif new_action == "chase_prey":
                    self.chase_prey()
                elif new_action == "reproduce":
                    self.reproduce()
                #elif new_action == "random_movement":
            
                self.time_since_last_action = pygame.time.get_ticks()
    
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
        if self.is_behavior_allowed("vision"):
            nearby_organisms = self.find_nearby_organisms()
            organisms_distances = self.get_organisms_distances(nearby_organisms)
            # Extract the closest predator, prey, and same species data
            closest_predators = next(([org, dist] for org, dist in organisms_distances if org.params["species"] in self.params["predators"]), None)
            closest_preys = next(([org, dist] for org, dist in organisms_distances if org.params["species"] in self.params["prey"]), None)
            closest_same_speciess = next(([org, dist] for org, dist in organisms_distances if org.params["species"] == self.params["species"]), None)
            # Update state
            # -1.0 is used as "None"
            self.current_state["closest_predator_distance"] = closest_predators[1] if closest_predators else -1.0
            self.current_state["closest_prey_distance"] = closest_preys[1] if closest_preys else -1.0
            self.current_state["closest_same_species_distance"] = closest_same_speciess[1] if closest_same_speciess else -1.0
            self.closest_predator = closest_predators[0] if closest_predators else None
            self.closest_prey = closest_preys[0] if closest_preys else None
            self.closest_same_species = closest_same_speciess[0] if closest_same_speciess else None
    
    # This is work in progress function not yet implemented
    # This function allows the organism to communicate with organisms of the same species
    def communicate(self):
        pass
    
    def change_direction(self, new_direction: float):
        self.current_state["current_direction"] = new_direction
    
    def change_speed(self, new_speed: float):
        if new_speed > self.params['max_speed']:
            self.current_state["current_speed"] = self.params['max_speed']
        else:
            self.current_state["current_speed"] = new_speed
    
    # time_to_rest is not used (yet)
    def rest(self, time_to_rest=0.0):
        self.change_speed(0.0)
        
    # Move in a given direction
    # Direction is in an angle. if radians=True then direction_angle is given in radians
    # if radians=False then direction_angle is given in degrees from 0 to 360
    def move(self, radians=True):
        if self.is_behavior_allowed("move"):
            # We only need to 'move' if speed != 0
            if self.current_state["current_speed"] != 0.0:
                direction_angle = self.current_state["current_direction"]
                if radians == False:
                    direction_angle = math.radians(direction_angle)
                if self.is_behavior_allowed("move"):
                    dx = math.cos(direction_angle) * self.current_state["current_speed"]
                    dy = math.sin(direction_angle) * self.current_state["current_speed"]
                    self.rect.x += dx
                    self.rect.y += dy
                
    def flee_from_predator(self):
        self.change_speed(self.params['max_speed'])
        
    def chase_prey(self):
        self.change_speed(self.params['max_speed'])
            
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
        elif organism.params["species"] in self.params["predators"]:
            return "predator"
        else:
            return ""
        
    def eat_prey(self, prey):
        #if self.rect.colliderect(prey.rect):
        # When eating a prey. The organism receives consumption_efficiency % of the energy of the prey
        consumption_efficiency = 0.75
        self.current_state["current_energy"] += prey.current_state["current_energy"] * consumption_efficiency 
        prey.destroy()
    
    # Check for collisions with other organisms
    def check_collision(self):
        if self.is_behavior_allowed("check_collision"):
            # collisions is a list of sprites that collide with the current organism
            collisions = pygame.sprite.spritecollide(self, all_sprites, False)
            for collided_organism in collisions:
                if self.is_prey_or_predator(collided_organism) == "prey":
                    self.eat_prey(collided_organism)
    
    # Specific reproduction for plants        
    def reproduce_plant(self):
        pass
    
    # Specific reproduction for non-plants
    def reproduce(self):
        # If offspring produced is 0 the organism is infertile
        if self.current_state['current_offspring_produced'] != 0.0:
            # Reproduction happens if there is enough energy and if enough time has passed
            current_time = pygame.time.get_ticks()
            energy_condition = self.current_state["current_energy"] >= self.params["max_energy"]
            time_condition = current_time - self.current_state["last_reproduction_time"] > self.current_state["current_reproduction_rate"]
            reproduce = False
            if "reproduce_plant" in self.params["behaviors"]:
                if time_condition:
                    reproduce = True        
            else:
                if energy_condition and time_condition:        
                    reproduce = True
            if reproduce == True:
                # Reproduction penalty reduces energy of current organism by a certein percentage
                self.current_state["current_energy"] *= self.reproduction_pentalty
                offset = 50 # Position offset of where the new organism should be.
                # The offspring_produced decides how many offspring are created based on a normal distribution
                for i in range(int(self.current_state['current_offspring_produced'])):
                    new_position = [self.rect.centerx + random.randint(-(offset + self.current_state["width"]), offset + self.current_state["width"]),
                                    self.rect.centery + random.randint(-(offset + self.current_state["height"]), offset + self.current_state["height"])]
                    new_organism = Organism(self.params, new_position)  #copy.deepcopy.self.params Should we deep copy the params?
                    all_sprites.add(new_organism)
                # Recalculate the offspring produced for next time
                self.current_state["current_offspring_produced"]: random_with_bias(self.params["min_offspring_produced"], self.params["max_offspring_produced"])
                self.current_state["last_reproduction_time"] = current_time
            
    # Reduce the energy level of the organism from hunger
    def hunger(self):
        if self.is_behavior_allowed("hunger"):
            current_time = pygame.time.get_ticks()
            if current_time - self.current_state["last_energy_update_time"] > self.current_state["current_energy_loss_rate"]:
                # Energy loss is also based the current moving speed of the organism
                self.current_state["current_energy"] -= 1 + (self.current_state["current_speed"] * 0.8)
                self.current_state["last_energy_update_time"] = current_time

    # When and what happens on death
    def die(self):
        # Death happens if energy level < 0 (starvation) or the organism reached its lifespan
        lifespan_condition = self.current_state["current_age"] >= self.current_state["lifespan"]
        if "die_plant" in self.params["behaviors"]:
            if lifespan_condition:
                self.destroy()
        else:
            energy_condition = self.current_state["current_energy"] <= 0
            if energy_condition or lifespan_condition:
                self.destroy()
    
    # Delete/kill/destroy the instance of the organism from the simulation        
    def destroy(self):
        APP['tracker'][f"total_{self.params['species']}"] -= 1
        self.kill()
        
