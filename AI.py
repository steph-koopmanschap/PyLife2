import os
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def set_random_seed(seed = 73):
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def state_to_tensor(state):
    state = np.array(state, dtype=np.float64)
    state = T.tensor(state, dtype=T.float)
    return state

def squeeze_vars(probability_distribution, action, critic_value):
    probabilities = T.squeeze(probability_distribution.log_prob(action)).item()
    action = T.squeeze(action).item()
    critic_value = T.squeeze(critic_value).item()
    return [probabilities, action, critic_value]

class AgentMemory():
    def __init__(self, batch_size):
        # Each item in the lists is one piece of memory
        self.memory = {
            "states": [], # States
            "probabilities": [], # log probabilities
            "critic_outputs": [], # values the critic calculates
            "actions": [],
            "rewards": [], # Rewards received
            "dones": [] # Terminal flags?
        }
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = self.get_memory_size() # Current memory size
        batch_start = np.arange(0, n_states, self.batch_size)
        # Each index is 1 memory
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # Create the start and end indices for each batch
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return [self.memory, batches]
    
    def store_memory(self, state, action, critic_value, probs, reward, done):
        self.memory["states"].append(state)
        self.memory["actions"].append(action)
        self.memory["probabilities"].append(probs)
        self.memory["critic_outputs"].append(critic_value)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)
        
    # Clear memory at the end of a trajectory
    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []
            
    def get_memory_size(self) -> int:
        return len(self.memory["states"])

class ActorNeuralNetwork(nn.Module):
    def __init__(self, n_actions, state_dims, lr, fc1_dims=256, fc2_dims=256):
        super(ActorNeuralNetwork, self).__init__()
        self.checkpoint_folder = "ai_checkpoints"
        self.neuralNetwork = nn.Sequential(
            nn.Linear(state_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        # Set model to evaluation mode to make predictions
        self.neuralNetwork.eval()
        state = state.to(self.device) # Send state to the GPU
        dist = self.neuralNetwork(state)
        dist = Categorical(dist) # Define categorical distribution
        return dist
    
    def save_checkpoint(self, checkpoint_file):
        path = os.path.join(self.checkpoint_folder, checkpoint_file)
        T.save(self.state_dict(), path)
        
    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.checkpoint_folder, checkpoint_file)
        self.load_state_dict(T.load(path))

class CriticNeuralNetwork(nn.Module):
    def __init__(self, state_dims, lr, fc1_dims=256, fc2_dims=256):
        super(CriticNeuralNetwork, self).__init__()
        self.checkpoint_folder = "ai_checkpoints"
        self.neuralNetwork = nn.Sequential(
            nn.Linear(state_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        self.neuralNetwork.eval()
        state = state.to(self.device) # Send state to the GPU
        value = self.neuralNetwork(state)
        return value  
    
    def save_checkpoint(self, checkpoint_file):
        path = os.path.join(self.checkpoint_folder, checkpoint_file)
        T.save(self.state_dict(), path)
        
    def load_checkpoint(self, checkpoint_file):
        path = os.path.join(self.checkpoint_folder, checkpoint_file)
        self.load_state_dict(T.load(path))

