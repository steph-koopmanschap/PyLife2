from constants import COLORS

# Pre-defined organisms

# Just used as a reference
template = {
        "species": "organism", # String
        "min_width": 1.0, # In pixels
        "max_width": 2.0, # In pixels
        "min_height": 1.0, # In pixels
        "max_height": 2.0, # In pixels
        "color": (255, 255, 255), # In RGB
        "min_speed": 0.1,
        "max_speed": 0.2,
        "min_vision_range": 1.0, # In pixels
        "max_vision_range": 2.0, # In pixels
        "prey ": [], # String array
        "predators": [], # String array
        "max_energy": 1.0,
        "min_energy_loss_rate": 1.0, # In miliseconds
        "max_energy_loss_rate": 2.0, # In miliseconds
        "min_reproduction_rate": 10000.0, # In miliseconds
        "max_reproduction_rate": 20000.0, # In miliseconds
        "min_lifespan": 10000.0, # In miliseconds
        "max_lifespan": 20000.0, # In miliseconds
}

Organisms = {
    "Plankton": {
        "species": "plankton",
        "min_width": 2.0,
        "max_width": 5.0,
        "min_height": 3.0,
        "max_height": 13.0,
        "color": COLORS["GREEN"],
        "min_speed": 0.1,
        "max_speed": 0.2,
        "min_vision_range": 0.0,
        "max_vision_range": 0.0,
        "prey": [],
        "predators": [],
        "max_energy": 10.0,
        "min_energy_loss_rate": 9999999.0,
        "max_energy_loss_rate": 9999999.0,
        "min_reproduction_rate": 10000.0,
        "max_reproduction_rate": 20000.0,
        "min_lifespan": 21000.0,
        "max_lifespan": 25000.0,
    },
    "Small_fish": {
        "species": "small_fish",
        "min_width": 10.0,
        "max_width": 20.0,
        "min_height": 5.0,
        "max_height": 10.0,
        "color": COLORS["BLUE"],
        "min_speed": 1.5,
        "max_speed": 2.0,
        "min_vision_range": 20.0,
        "max_vision_range": 40.0,
        "prey": ["plankton"],
        "predators": [],
        "max_energy": 10.0,
        "min_energy_loss_rate": 2000.0,
        "max_energy_loss_rate": 3000.0,
        "min_reproduction_rate": 30000.0,
        "max_reproduction_rate": 60000.0,
        "min_lifespan": 60000.0,
        "max_lifespan": 120000.0,
    },
    "Big_fish": {
        "species": "big_fish",
        "min_width": 15.0,
        "max_width": 10.0,
        "min_height": 30.0,
        "max_height": 20.0,
        "color": COLORS["RED"],
        "min_speed": 1.5,
        "max_speed": 3.0,
        "min_vision_range": 20.0,
        "max_vision_range": 45.0,
        "prey": ["small_fish"],
        "predators": [],
        "max_energy": 20.0,
        "min_energy_loss_rate": 2000.0,
        "max_energy_loss_rate": 3000.0,
        "min_reproduction_rate": 45000.0,
        "max_reproduction_rate": 90000.0,
        "min_lifespan": 90000.0,
        "max_lifespan": 180000.0,
    },
    "Shark": {
        "species": "shark",
        "min_width": 20.0,
        "max_width": 50.0,
        "min_height": 30.0,
        "max_height": 35.0,
        "color": (230, 230, 230),
        "min_speed": 1.5,
        "max_speed": 4.0,
        "min_vision_range": 40.0,
        "max_vision_range": 50.0,
        "prey": ["small_fish, big_fish"],
        "predators": [],
        "max_energy": 25.0,
        "min_energy_loss_rate": 3000.0,
        "max_energy_loss_rate": 3500.0,
        "min_reproduction_rate": 180000.0,
        "max_reproduction_rate": 240000.0,
        "min_lifespan": 300000.0,
        "max_lifespan": 480000.0,
    },
    "Orca": {
        "species": "orca",
        "min_width": 20.0,
        "max_width": 50.0,
        "min_height": 30.0,
        "max_height": 35.0,
        "color": (230, 230, 230),
        "min_speed": 1.5,
        "max_speed": 4.0,
        "min_vision_range": 40.0,
        "max_vision_range": 50.0,
        "prey": ["shark"],
        "predators": [],
        "max_energy": 25.0,
        "min_energy_loss_rate": 3000.0,
        "max_energy_loss_rate": 3500.0,
        "min_reproduction_rate": 180000.0,
        "max_reproduction_rate": 240000.0,
        "min_lifespan": 300000.0,
        "max_lifespan": 480000.0,
    },
    "Whale": {
        "species": "whale",
        "min_width": 75.0,
        "max_width": 150.0,
        "min_height": 30.0,
        "max_height": 40.0,
        "color": COLORS["MEDIUM_BLUE"],
        "min_speed": 1.0,
        "max_speed": 2.0,
        "min_vision_range": 40.0,
        "max_vision_range": 50.0,
        "prey": ["plankton"],
        "predators": [],
        "max_energy": 40.0,
        "min_energy_loss_rate": 3000.0,
        "max_energy_loss_rate": 3500.0,
        "min_reproduction_rate": 240000.0,
        "max_reproduction_rate": 300000.0,
        "min_lifespan": 480000.0,
        "max_lifespan": 600000.0,
    }
}
