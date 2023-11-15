# This is a 'seperate' program that can plot the statistical data generated by a PyLife simulation
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from utils import load_log_tracker 

def load_data(filename = None):
    folder_path = "sim_statistics_data"
    # If no filename is given then get the latest file
    if not filename:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)
        del files[files.index("placeholder")]
        # Extract timestamps from filenames [:-5] removes '.json' from filename
        timestamps = [file[:-5].split("_")[2] for file in files]
        # Convert timestamps to datetime objects
        date_objects = [datetime.strptime(timestamp, "%H:%M:%S %y-%m-%d") for timestamp in timestamps]
        # Find the index of the file with the latest timestamp
        latest_index = date_objects.index(max(date_objects))
        # Get the filename of the latest file
        filename = files[latest_index]
    
    return load_log_tracker(filename)

# Options: "real" or "ticks"
def plot(time_format = "real"):
    print("Plotting...")
    data = load_data()
    time_data = []
    total_organisms = []
    total_current_energy = []
    average_current_energy = []
    species_labels = []
    species_counts_dict = {}
    species_counts_list = []
    for key in data[0]:
        if "total" in key and key != "total_organisms" and key != "total_current_energy" and key != "total_species":
            species_labels.append(key)
    for entry in data:   
        if time_format == "real":
            time_data.append(entry["time_stamp"])
        elif time_format == "ticks":
            time_data.append(entry["pygame_tick"])
        total_organisms.append(entry["total_organisms"])
        total_current_energy.append(entry["total_current_energy"]) 
        average_current_energy.append(entry["average_current_energy"]) 
        for label, value in entry.items():
            if label in species_labels: # and label != "total_plankton"
                if label not in species_counts_dict:
                    species_counts_dict[label] = []
                species_counts_dict[label].append(value)
    for label, values in species_counts_dict.items():
        species_counts_list.append(values)

    # print("time_data ", time_data)
    # print("species", species_labels)
    # print("species_counts_list", species_counts_list)
    # print("total_organisms ", total_organisms)
    # print("total_current_energy ", total_current_energy)
    # print("average_current_energy ", average_current_energy)

    
    xlabel = 'Time stamp' if time_data == "real" else 'Sim ticks'
    # Plotting total_organisms vs time_stamp
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, total_organisms, marker='o')
    plt.title('Total Organisms over Time')
    plt.xlabel(xlabel)
    plt.ylabel('Total Organisms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plot_results/total_organisms_{time_format}.png')
    #plt.show()

    # Plotting total_current_energy vs time_stamp
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, total_current_energy, marker='o', color='orange')
    plt.title('Total Current Energy over Time')
    plt.xlabel(xlabel)
    plt.ylabel('Total Current Energy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plot_results/total_current_energy_{time_format}.png')
    #plt.show()

    # Plotting average_current_energy vs time_stamp
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, average_current_energy, marker='o', color='green')
    plt.title('Average Current Energy over Time')
    plt.xlabel(xlabel)
    plt.ylabel('Average Current Energy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plot_results/average_current_energy_{time_format}.png')
    #plt.show()
    
    # Plotting organisms counts vs time_stamp
    plt.figure(figsize=(10, 5))
    for i in range(len(species_counts_list)):
        plt.plot(time_data, species_counts_list[i], marker='o')
    plt.legend(species_labels) #species_labels[1:]
    plt.title('Species counts over Time')
    plt.xlabel(xlabel)
    plt.ylabel('Species counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plot_results/species_counts_{time_format}.png')
    
    print("Plotting done.")
    
plot('real')
plot('ticks')