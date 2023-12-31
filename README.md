# PyLife2
The Improved version of PyLife (now with AI)

PyLife is an Ocean Food Web Simulation.
The simulation uses PyGame for rendering, Numpy for calculations and PyTorch for neural networks.

The simulation consists of sea life animals where each animal is powered by its own AI.
The AI consists of Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm implemented with PyTorch.
Trough this the animal learns to change its behavior over time as the simulation runs.


The animal has 2 objectives:
1. To survive for as long as possible.
2. To reproduce as much as possible.

## Rules of the simulation

### The rules of the animals are as follows:

1. The animal has an energy reserve.
2. If the energy reserve is depleted (<0) then the animal dies.
3. The energy reserve of animal goes down over time.
4. The energy reserves goes down faster, the faster the animal moves.
5. If an animal eats another animal, it gains a percentage of the energy reserves of the other animal.
6. If an animal is eaten, it dies.
7. An animal can reproduce if its energy reserve reaches a certein upper limit.
8. If an animal reproduces its energy reserve is depleted by some %
9. Every animal has a lifespan.
10. If the animal reaches its lifespan it dies.
11. Each animal has prey that it can eat. 
12. Visa versa, each animal has predators that it can be eaten by.

## Usage instructions

### Install

Make sure to install the requirements in `requirements.txt`. <br/>
You can use the command `pip install -r requirements.txt` for this.

### Startup

To run the simulation use the command `python main.py`.  <br/>

### Interactions

Use the left mouse button to select an animal. <br/>
Once an animal has been selected: <br/>
* Press 'S' to save the training process of the animal's neural network to a file. <br/>
* Press 'I' to save the information of that animal to a file. <br/>

Press 'L' to force log current statistics to the statistics file (the normal rate is to log every 2.5 seconds)

## The food web and animals

You can see the current animals used in the simulation in the `organisms.json` file. <br/>
By using and modifying this file you can easily create, add, and change animals in simulation yourself.

The animals in the simulation I have currently added are:
```
Plankton:
    "prey": [],
    "predators": []
Small_fish:
        "prey": [
            "plankton"
        ],
        "predators": [
            "big_fish", "shark"
        ]
Big_fish:
        "prey": [
            "small_fish"
        ],
        "predators": [
            "shark"
        ]
Shark:
        "prey": [
            "small_fish", "big_fish"
        ],
        "predators": [
            "orca"
        ]
Orca:
        "prey": [
            "shark"
        ],
        "predators": []
Whale:
        "prey": [
            "plankton"
        ],
        "predators": []
```

Using the prey and predators list you can construct your food web structures and food chain hierarchies. <br/>
The other paramaters of the animals such reproduction rate, movement speed, and how many of each species are created at the start of the simulation
will decide the outcome and stability of the ecosystem. In this way the ecosystem simulation follows a chaotic pattern. <br/>
Especially the plankton species which is the primary producer of the ecosystem, largely influences the outcomes of the simulation.

## How the AI works

Each animal consists of 2 Neural Networks. An actor neural network and a critic neural network. <br/>
The actor takes in the current state of the animal and then outputs a probability distribution of actions the animal can perform. <br/>
At the same time the critic also takes in the current state of the animal and outputs an evaluation of how "good" the current state of the animal is. <br/>
An action is then performed. The action returns a reward or punishment. <br/>
The action that was performed, the state that was accociated with that action, and the reward of the action are stored in the memory of the animal. <br/>
When the animal has collected a certein number of memories. <br/>
The actor and critic neural networks are then trained on the memories of the animal. <br/>
When the neural networks are trained the memory is deleted and a new cycle of actions, memory, and learning starts again. <br/>
After the neural networks are trained, the animal will change its behavior (the actions it choses based on its state) accordingly.

## Statistics

With every simulation a new file is created in the `sim_statistics_data` called `sim_tracker_{datetime}`. <br/>
This file records data over time of the simulation such as counts per species, total animals, and energy reserves. <br/>
If you run this command `python stats_plotter.py` then graphs/charts of the simulation's statistics will be plotted in `.png` images. <br/>
In the `plot_results` folder. It will automatically use the data of the latest run simulation. <br/>
Re-running the `stats_plotter.py` will override the existing images.

## Performance

For the best performance It is best to run this on a computer with CUDA(12.1) capable Nvidia GPU. <br/>
However the simulation has been tested to run on a Intel Core i5 CPU where it runs relativaly smoothly. <br/>
The biggest performance hog is the amount of plankton in the simulation. <br/>
While the plankton does not have any AI, because it is a plant without behhavior, as a primary producer the amount of plankton needs to be at least 1000 or more for the ecosystem to be stable.

## Contributing to PyLife

Because this is a free open souce project anyone is allowed to contribute to PyLife. <br/>
To contribute you may contact me (the project creator) on Discord with username: `greylien`
