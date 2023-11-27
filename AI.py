import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


epsilon = 0.2
epislon_range = [1 - epsilon, 1 + epsilon]
batch_size = 5
lr = 0.002

# Random Seed
seed = 73
#torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#env.seed(seed)