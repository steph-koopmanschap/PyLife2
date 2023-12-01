import pygame
from globals import APP

# The environment class regulates environmental conditions of the simulation
class Environment:
    def __init__(self):
        self.current_environment = {
            # How many in game ticks 24 hours is
            "day_cycle_time_ticks": 60000,
            # The current time of day
            "time_of_day": 0,
            "is_night": False,
        }
        self.time_since_last_day_reset = pygame.time.get_ticks()
        APP['days_passed'] = 0
        
    def update(self, now):
        self.current_environment['time_of_day'] = now - self.time_since_last_day_reset
        if self.current_environment['time_of_day'] < self.current_environment['day_cycle_time_ticks'] * 0.5:
            self.current_environment['is_night'] = False
        else:
            self.current_environment['is_night'] = True
            
        # Reset the day
        if  self.current_environment['time_of_day'] > self.current_environment['day_cycle_time_ticks']:
            self.time_since_last_day_reset = now
            self.current_environment['time_of_day'] = 0
            APP['days_passed'] += 1
            print("Days passed: ", APP['days_passed'])
        