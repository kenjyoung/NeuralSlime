import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import argparse
import jax
import jax.numpy as jnp

#Cell types:
wall_index = 0
photosynth_index = 1
absorber_index = 2
enzyme_index = 3
mover_index = 4
channel_index = 5

wall_color = [131 / 255, 136 / 255, 145 / 255]  # grey
photosynth_color = [0, 1, 0]  # green
absorber_color = [217 / 255, 0, 255 / 255]  # purple
enzyme_color = [1, 0, 0]  # red
mover_color = [0, 0, 1]  # blue
channel_color = [1, 1, 0]  # yellow
organic_matter_color = [0, 0, 0]  # black
nutrient_color = [1, 165 / 255, 0]  # orange

parser = argparse.ArgumentParser()
parser.add_argument("--loadfile", help="specify a file to load", default=None)
parser.add_argument("--frequency", help="how often to plot a point", type=int, default=1)
args = parser.parse_args()

with open(args.loadfile, "rb") as f:
    record = pkl.load(f)

#go through record and log certain statistics over time
#things to log:
#   - amount of organic matter
#   - amount of nutrients
#   - amount of each cell type (photosynth, absorber, wall, enzyme, channel, mover)
def process_record(record):
    states = record["states"]
    C = record["config"]
    organic_matter = []
    nutrients = []
    enzyme = []
    photosynth = []
    absorber = []
    wall = []
    channel = []
    mover = []
    
    for S in states[::args.frequency]:
        organic_matter.append(np.sum(S["organic_matter"]))
        nutrients.append(np.sum(S["nutrients"]))
        cell_type_weights = S["organic_matter"] * jax.nn.softmax(S["cell_type_logits"], axis=0)**C["type_sharpness"]
        enzyme.append(np.sum(cell_type_weights[enzyme_index]))
        photosynth.append(np.sum(cell_type_weights[photosynth_index]))
        absorber.append(np.sum(cell_type_weights[absorber_index]))
        wall.append(np.sum(cell_type_weights[wall_index]))
        channel.append(np.sum(cell_type_weights[channel_index]))
        mover.append(np.sum(cell_type_weights[mover_index]))
    return_dict = {"organic_matter": organic_matter,
                   "nutrients": nutrients,
                   "enzyme": enzyme,
                   "photosynth": photosynth,
                   "absorber": absorber,
                   "wall": wall,
                   "channel": channel,
                   "mover": mover}
    colors = {"organic_matter": organic_matter_color,
              "nutrients": nutrient_color,
              "enzyme": enzyme_color,
              "photosynth": photosynth_color,
              "absorber": absorber_color,
              "wall": wall_color,
              "channel": channel_color,
              "mover": mover_color}
    return return_dict, colors

to_plot = [
    "organic_matter",
    "nutrients",
    "enzyme", 
    "photosynth", 
    "absorber", 
    "wall", 
    "channel", 
    "mover"
]
def plot_dict(dict, colors=None):
    for key in to_plot:
        plt.plot(dict[key], label=key, color=None if colors is None else colors[key])
    plt.legend()

plot_dict(*process_record(record))
plt.show()

        

