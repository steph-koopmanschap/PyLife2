import uuid


# Globals are changeable variables that can be accesed troughout the program
APP = {
    "simulation_id": str(uuid.uuid4()), # A unique id for this run of the simulation
    "sim_start_time": 0.0,
    'sim_start_time_ticks': 0,
    "current_log_file": "",
    # How many seconds the statistics of the simulation needs to be logged
    "logging_rate": 2500,
    "tracker": {}
}

# "logging_rate: 5000"
