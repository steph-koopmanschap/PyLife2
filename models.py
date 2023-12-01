import os
import json
import math
import random
import uuid
import numpy as np
import torch as T
import pygame
from globals import APP
from constants import WINDOW_WIDTH, WINDOW_HEIGHT
from utils import load_organism_definitions, random_with_bias, calc_distance, opposite_angle
from AI import AgentMemory, ActorNeuralNetwork, CriticNeuralNetwork, squeeze_vars, state_to_tensor

# Sprite groups
all_sprites = pygame.sprite.Group()

pre_defined_organisms = load_organism_definitions()
species = [organism['species'] for organism_definition, organism in pre_defined_organisms.items()]

class Organism(pygame.sprite.Sprite):
    # The position is the initial position where the organism is "created" or "born"
    # The actor_params, and critic_params are the trained neural network parameters
    def __init__(self, params, position, actor_params=None, critic_params=None):
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
            "current_speed": round(random.uniform(self.params['min_speed'], self.params['max_speed']), 2),
            # The current direction the organism is facing
            "current_direction": round(random.uniform(0.0, 360.0), 2),
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
            #self.move_direction = random.uniform(0, 2 * math.pi)
            #self.move_timer = pygame.time.get_ticks() + random.randint(1000, 3000)
            # The available methods are all the methods of this class.
            #self.available_methods = [method for method in dir(self) if "__" not in method]
        # AI using PPO algorithm
        if self.params["has_brain"]:
            # The possible actions/behaviors the organism can take    
            self.action_space = [
                #"change_speed",
                #"change_direction",
                #"move_random",
                "flee_from_predator",
                "chase_prey",
                #"rest",
                #"reproduce",
            ]
            state_to_parse = self.prepare_state_for_AI()
            self.time_since_last_action = pygame.time.get_ticks()
            # Checkpoint files are used for saving and loading a trained organism
            self.checkpoint_file_actor = f"nnActor_{self.params['species']}_{self.instance_id}_{APP['simulation_id']}_{APP['sim_start_time_ticks']}"
            self.checkpoint_file_critic = f"nnCritic_{self.params['species']}_{self.instance_id}_{APP['simulation_id']}_{APP['sim_start_time_ticks']}"
            # HYPER PARAMATERS of the AI
            learning_rate = 0.0003
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.epsilon = 0.2
            self.policy_clip_range = [1 - self.epsilon, 1 + self.epsilon]
            neural_network_deepness = 256
            batch_size = 5
            self.n_epochs = 4 # How many epochs/updates to perform in a learning cycle
            # How many actions have to be performed for the AI to learn
            self.n_actions_trigger_learning = 20
            # How many actions have been performed. (resets every learning cycle)
            self.action_counter = 0
            self.current_action = ""
            self.memory = AgentMemory(batch_size)
            self.actorNN = ActorNeuralNetwork(len(self.action_space), len(state_to_parse), learning_rate, neural_network_deepness, neural_network_deepness)
            self.criticNN = CriticNeuralNetwork(len(state_to_parse), learning_rate, neural_network_deepness, neural_network_deepness)
            # Load trained model parameters from a previous neural network into the current neural network.
            if actor_params and critic_params:
                self.actorNN.load_params_mem(actor_params)
                self.criticNN.load_params_mem(critic_params)
                
    
    # Save info of this organism to a json file
    def save_info(self):
            folder = "organism_info"
            file_name = f"{self.params['species']}_{self.instance_id}_{APP['simulation_id']}_{APP['simulation_id']}.json"
            path = os.path.join(folder, file_name)
            info = {
                "instance_id": self.instance_id,
                "simulation_id": APP['simulation_id'],
                "state": self.current_state,
                "params": self.params
            }
            with open(path, 'w') as json_file:
                json.dump(info, json_file, indent=4)
            print(f"Saved info of {self.params['species']}.")
    
    # Save the learned parameters of neural network of this organism
    def save_AI_params(self):
        self.actorNN.save_checkpoint(self.checkpoint_file_actor)
        self.criticNN.save_checkpoint(self.checkpoint_file_critic)
        print(f"Saved AI learning progress of {self.params['species']}.")
    # Transforms the state of the organism into a format the AI can understand
    def prepare_state_for_AI(self):
        return [
            self.current_state['current_vision_range'],
            self.current_state['current_speed'],
            self.current_state['current_direction'],
            self.current_state['current_energy'],
            self.current_state['current_energy_loss_rate'],
            self.current_state['current_reproduction_rate'],
            self.current_state['lifespan'],
            self.current_state['current_age'],
            self.current_state['predator_detected_time'],
            self.current_state['creation_time'],
            self.current_state['last_energy_update_time'],
            self.current_state['last_reproduction_time'],
            self.current_state['closest_predator_distance'],
            self.current_state['closest_prey_distance'],
            self.current_state['closest_same_species_distance'],
            self.current_state['current_offspring_produced']
        ]
    
    def is_behavior_allowed(self, behavior): 
        return behavior in self.params['behaviors'] # self.available_methods[self.available_methods.index(behavior)]
            #return True
            #return False
        
    # Update itself
    def update(self):
        #pygame.draw.circle(APP['screen'], self.params['color'], (self.rect.x, self.rect.y), int(self.current_state['current_vision_range']), 2)
        #APP['screen'].blit(self.image, self.rect)
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
        # Reproduce as much as possible. Reproduction can happen if there is enough energy reserves
        if self.params["has_brain"]:
            #pass
            if pygame.time.get_ticks() - self.time_since_last_action > random.randint(200, 5000):
                state_to_parse = self.prepare_state_for_AI()
                # Decide on which action to take from the action_space based on current_state
                # Get the distribution from the actor (predict) based on current state
                probability_distribution = self.actorNN(state_to_tensor(state_to_parse))
                # Get the value of the current state
                value = self.criticNN(state_to_tensor(state_to_parse))
                action = probability_distribution.sample()
                squeezed = squeeze_vars(probability_distribution, action, value)
                probability = squeezed[0]
                action = squeezed[1]
                critic_value = squeezed[2]
                # Choose an action from the action space
                chosen_action = self.action_space[action]
                #print(f"Chosen action: {chosen_action}, from {self.params['species']} ID: {self.instance_id}")
                #chosen_action = random.choice(self.action_space)
            #if pygame.time.get_ticks() - self.time_since_last_action > random.randint(200, 5000):
                if chosen_action == "change_speed":
                    reward = self.change_speed(round(random.uniform(self.params['min_speed'], self.params['max_speed']), 2))
                elif chosen_action == "change_direction":
                    reward = self.change_direction(round(random.uniform(0.0, 360.0), 2)) 
                elif chosen_action == "move_random":
                    reward = self.move_random()
                elif chosen_action == "rest":
                    reward = self.rest()
                elif chosen_action == "flee_from_predator":
                    reward = self.flee_from_predator()
                elif chosen_action == "chase_prey":
                    reward = self.chase_prey()
                
                # print("chosen action:", chosen_action)
                # print("reward:", reward)
                #elif chosen_action == "random_movement":
                # Remember the state and outcome of the action just performed.
                self.memory.store_memory(state_to_parse, action, critic_value, probability, reward, 0)
                self.action_counter += 1
                # Start learning every self.n_actions_trigger_learning
                if (self.action_counter >= self.n_actions_trigger_learning):
                    self.learn()
                    self.action_counter = 0 # Reset action acounter

                self.time_since_last_action = pygame.time.get_ticks()
    
    # Make the AI learn (train the model)
    def learn(self):
        if self.params["has_brain"]:
            #print("Learning...")
            # First put the models into training mode
            self.actorNN.neuralNetwork.train()
            self.criticNN.neuralNetwork.train()
            for _ in range(self.n_epochs):
                # Retrieve our momery which is the data used for training
                mem = self.memory.generate_batches()
                batches = mem[1]
                states = np.array(mem[0]['states'])
                actions = np.array(mem[0]['actions'])
                probabilities = np.array(mem[0]['probabilities'])
                critic_outputs = np.array(mem[0]['critic_outputs'])
                rewards = np.array(mem[0]['rewards'])
                dones = np.array(mem[0]['dones'])
                # Advantage is the 'goodness'/benefit of the state compared to the previous state
                # The advantage needs to be calculated at each action
                advantage = np.zeros(len(rewards), dtype=np.float32)
                # For each action
                for t in range(len(rewards - 1)):
                    discount = 1 # Discount factor
                    a_t = 0 # Advantage at each action
                    for k in range(t, len(rewards)-1):
                        a_t += discount * (rewards[k] + self.gamma * critic_outputs[k+1] * (1 - int(dones[k])) - critic_outputs[k])  
                        discount *= self.gamma * self.gae_lambda
                    advantage[t] = a_t
                        
                advantage = T.tensor(advantage).to(self.actorNN.device)
                critic_outputs = T.tensor(critic_outputs).to(self.actorNN.device)
                # Learning with mini batches
                for batch in batches:
                    batch_states = T.tensor(states[batch], dtype=T.float).to(self.actorNN.device)
                    old_probabilities = T.tensor(probabilities[batch]).to(self.actorNN.device)
                    batch_actions = T.tensor(actions[batch]).to(self.actorNN.device)
                
                    prob_distribution = self.actorNN(batch_states) # Make prediction
                    critic_value = self.criticNN(batch_states) # Make prediction
                    critic_value = T.squeeze(critic_value)
                        
                    new_probabilities = prob_distribution.log_prob(batch_actions)
                    prob_ratio = new_probabilities.exp() / old_probabilities.exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = T.clamp(prob_ratio, self.policy_clip_range[0], self.policy_clip_range[1]) * advantage[batch]
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                    # Return = advantage + critic_output_memory
                    returns = advantage[batch] + critic_outputs[batch]
                    # critic_loss = MSE(return - critic_output_network)
                    critic_loss = (returns - critic_value)**2
                    critic_loss = critic_loss.mean()
                    total_loss = actor_loss + 0.5 * critic_loss
                    self.actorNN.optimizer.zero_grad()
                    self.criticNN.optimizer.zero_grad()
                    total_loss.backward()
                    self.actorNN.optimizer.step()
                    self.criticNN.optimizer.step()
            # Clear memory at end of learning.
            # After this the memory is build up again from the next actions
            self.memory.clear_memory()
                
    def screen_wrap(self):
        # Only screen wrap on x-axis
        if self.rect.left > WINDOW_WIDTH:
            self.rect.right = 0
        elif self.rect.right < 0:
            self.rect.left = WINDOW_WIDTH
        # TODO: Remove y-axis screen wrapping
        if self.rect.top > WINDOW_HEIGHT:
            self.rect.bottom = 0 # y-axis wrapping
            #self.change_direction(45.0) # Make the organism turn around
        elif self.rect.bottom < 0:
            self.rect.top = WINDOW_HEIGHT # y-axis wrapping
            #self.change_direction(315.0) # Make the organism turn around
            
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
        reward = 1.0
        return reward
    
    def change_speed(self, new_speed: float):
        if new_speed > self.params['max_speed']:
            self.current_state["current_speed"] = self.params['max_speed']
        else:
            self.current_state["current_speed"] = new_speed
        reward = 1.0
        return reward
    
    # Move in a random direction at a random speed
    def move_random(self):
        reward = 2.0
        self.change_direction(round(random.uniform(0.0, 360.0), 2))
        self.change_speed(round(random.uniform(self.params['min_speed'], self.params['max_speed']), 2))
        return reward
    
    # time_to_rest is not used (yet)
    def rest(self, time_to_rest=0.0):
        self.change_speed(0.0)
        # Set reward for the AI
        reward = 0
        # Resting when low on energy is good
        if self.current_state["current_energy"] < self.current_state["current_energy"] * 0.1:
            reward += 2.0
        # Resting if a prey/food is nearby is bad
        if self.current_state["closest_prey_distance"] != -1.0:
            reward -= 10.0
        # Resting if a predator is nearby is bad
        if self.current_state["closest_predator_distance"] != -1.0:
            reward -= 10.0 
        return reward
    
    def flee_from_predator(self):
        reward = 0.0
        # If there is a predator we need to move in the opposite direction as the predator at max speed
        if self.closest_predator != None:
            self.change_direction(opposite_angle(self.closest_predator.current_state['current_direction']))
            self.change_speed(self.params['max_speed'])
            reward = 16.0
        # If no predator move in random direction
        else:
            reward = self.move_random()
        return reward
    
    def chase_prey(self):
        reward = 0.0
        # If there is a prey we need to move in the same direction as the prey at max speed
        if self.closest_prey != None:
            self.change_direction(self.closest_prey.current_state['current_direction'])
            self.change_speed(self.params['max_speed'])
            reward = 15.0
        # If no prey move in random direction
        else:
            reward = self.move_random()
        return reward
        
    # Move in a given direction at a certein speed
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
                # And reduces speed to 0.0
                self.current_state["current_energy"] *= self.reproduction_pentalty
                self.change_speed(0.0)
                offset = 50 # Position offset of where the new organism should be.
                # The offspring_produced decides how many offspring are created based on a normal distribution
                for i in range(int(self.current_state['current_offspring_produced'])):
                    new_position = [self.rect.centerx + random.randint(-(offset + self.current_state["width"]), offset + self.current_state["width"]),
                                    self.rect.centery + random.randint(-(offset + self.current_state["height"]), offset + self.current_state["height"])]
                    # We give the train neural network parameters to the offspring
                    actor_params = None
                    critic_params = None
                    if self.params["has_brain"]:
                        actor_params = self.actorNN.save_params_mem()
                        critic_params = self.criticNN.save_params_mem()
                    new_organism = Organism(self.params, new_position, actor_params, critic_params)  #copy.deepcopy.self.params Should we deep copy the params?
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
        
