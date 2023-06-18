import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import jit, vmap
# from config import config as C
from time import time
from functools import partial
import pickle as pkl
import argparse
# from importlib import import_module
import json
import os

try:
    import pygame as pg
except:
    print("pygame not installed, visualization not available")

from visualize import Visualizer

try:
    import cv2
except:
    print("opencv not installed, video recording not available")

parser = argparse.ArgumentParser()
parser.add_argument("--loadfile", help="specify a file to load", default=None)
parser.add_argument("--config", help="specify a config file to load", default="config.json")
parser.add_argument("--seed", help="specify a seed", default=None)
args = parser.parse_args()

with open(args.config, "r") as f:
    C = json.load(f)

if args.seed is not None:
    key = random.PRNGKey(int(args.seed))
else:
    print("warning: no seed specified, using time as seed")
    key = random.PRNGKey(int(time()*1000))

hor_sobel_array = jnp.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])/8
vert_sobel_array = jnp.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])/8
laplace_array = jnp.asarray([[0.25,0.5,0.25],[0.5,-3,0.5],[0.25,0.5,0.25]])
smooth_array = jnp.asarray([[0,1,0],[1,1,1],[0,1,0]])/5
neighbor_array = jnp.asarray([[0,1,0],[1,0,1],[0,1,0]])

#Cell types:
if("enabled_types" not in C):
    C["enabled_types"] = ["wall", "photosynth", "absorber", "enzyme", "mover", "channel"]
enabled_types = C["enabled_types"]

num_cell_types = len(enabled_types)

#ELU
activation = jax.nn.elu

visualize = C["visualize"]

enable_video = C["bookkeeping_config"]["enable_video"]
video_frame_frequency = C["bookkeeping_config"]["video_frame_frequency"]
video_file_name = C["bookkeeping_config"]["video_file_name"]

enable_checkpoint = C["bookkeeping_config"]["enable_checkpoint"]
checkpoint_frequency = C["bookkeeping_config"]["checkpoint_frequency"]
checkpoint_file_prefix = C["bookkeeping_config"]["checkpoint_file_prefix"]

enable_save = C["bookkeeping_config"]["enable_save"]
save_frequency = C["bookkeeping_config"]["save_frequency"]
save_file_name = C["bookkeeping_config"]["save_file_name"]

enable_record = C["bookkeeping_config"]["enable_record"]
record_frequency = C["bookkeeping_config"]["record_frequency"]
record_file_prefix = C["bookkeeping_config"]["record_file_prefix"]
dump_record_frequency = C["bookkeeping_config"]["dump_record_frequency"]

def create_directories_for_prefix(prefix):
    directory = os.path.dirname(prefix)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

try:
    if(enable_checkpoint):
        create_directories_for_prefix(checkpoint_file_prefix)
    if(enable_record):
        create_directories_for_prefix(record_file_prefix)
except:
    print("Failed to create directories for checkpoint and/or record!")
    exit()

#TODO: may be a better way to express this
cell_type_names = ["wall", "photosynth", "absorber", "enzyme", "mover", "channel"]
i = 0
photosynth = absorber = enzyme = mover = channel = -1
for name in cell_type_names:
    if name in enabled_types:
        globals()[name] = i
        i+=1
        

#this is is simple way to ensure all index partition entries are the same length which makes it convenient to store as a jnp array
C["world_size"] = 6*(C["world_size"]//6)

# Compute a set of masks for which movement can be safely processed in parallel without causing conflicts
movement_partition = []
#tile layout:
#1,2,3,4,5,6#
#5,6,1,2,3,4#
#3,4,5,6,1,2#
for m in range(6):
    mask = jnp.zeros((C["world_size"], C["world_size"]))
    for i in range(C["world_size"]):
        mask = mask.at[i, jnp.arange((2*i+m)%6, C["world_size"], 6)%C["world_size"]].set(1)
    movement_partition.append(mask)

movement_partition = jnp.asarray(movement_partition)

nutrients = jnp.zeros((C["world_size"], C["world_size"]))
energy = jnp.zeros((C["world_size"], C["world_size"]))
waste = jnp.zeros((C["world_size"], C["world_size"]))
organic_matter = jnp.zeros((C["world_size"], C["world_size"]))
information = jnp.zeros((C["num_info_channels"], C["world_size"], C["world_size"]))
cell_type_logits = jnp.zeros((num_cell_types,C["world_size"], C["world_size"]))

#input consists of all information, nutrients, and cell type weights from the 5 cells nearest the agent
input_size = 5*(C["num_info_channels"]+num_cell_types+1)
#action consists of target cell type, target cell (out of 4 cardinal directions+no target), cling and spread
action_size = num_cell_types+5+2
if("mover" in enabled_types):
    action_size+=1
W_f = random.normal(key, (C["num_info_channels"], input_size, C["world_size"], C["world_size"]))
#U_f is not nessesary because h is included in x
b_f = random.normal(key, (C["num_info_channels"], C["world_size"], C["world_size"]))
W_h = random.normal(key, (C["num_info_channels"], input_size, C["world_size"], C["world_size"]))
U_h = random.normal(key, (C["num_info_channels"], C["num_info_channels"], C["world_size"], C["world_size"]))
b_h = random.normal(key, (C["num_info_channels"], C["world_size"], C["world_size"]))
W_o = random.normal(key, (action_size, C["num_info_channels"], C["world_size"], C["world_size"]))
b_o = random.normal(key, (action_size, C["world_size"], C["world_size"]))
weights = (W_f, b_f, W_h, U_h, b_h, W_o, b_o)

#adjust all relevant configs by time multiplier
#this just provides a convenient way to scale all time dependent quantities at once
C["nutrient_diffusion_rate"] *= C["time_multiplier"]
C["organic_matter_seed_rate"] *= C["time_multiplier"]
C["type_change_rate"] *= C["time_multiplier"]
C["channel_velocity_scale"] *= C["time_multiplier"]
C["energy_diffusion_rate"] *= C["time_multiplier"]

C["maximum_spread_rate"] *= np.sqrt(C["time_multiplier"])
C["default_spread_rate"] *= np.sqrt(C["time_multiplier"])

# Multiplicative quantities handled differently
C["photosynthesis_rate"] = (1+C["photosynthesis_rate"])**C["time_multiplier"]-1
C["absorber_rate"] = 1-(1-C["absorber_rate"])**C["time_multiplier"]
C["mutation_rate"] = 1-(1-C["mutation_rate"])**C["time_multiplier"]
C["movement_rate"] = 1-(1-C["movement_rate"])**C["time_multiplier"]
C["enzyme_damage_rate"] = 1-(1-C["enzyme_damage_rate"])**C["time_multiplier"]
C["organic_matter_decay_rate"] = 1-(1-C["organic_matter_decay_rate"])**C["time_multiplier"]
C["energy_generation_rate"] = (1+C["energy_generation_rate"])**C["time_multiplier"]-1

if(C["enable_waste"]):
    C["waste_diffusion_rate"] *= C["time_multiplier"]
    C["waste_decay_rate"] = 1-(1-C["waste_decay_rate"])**C["time_multiplier"]
    C["waste_damage_rate"] = 1-(1-C["waste_damage_rate"])**C["time_multiplier"]
    C["waste_generation_rate"] = (1+C["waste_generation_rate"])**C["time_multiplier"]-1
if(C["enable_metaevolution"]):
    C["metaevolution_noise_scale"] *= C["time_multiplier"]

if(C["enable_metaevolution"]):
    mutation_rate_logits = jnp.ones((C["world_size"], C["world_size"]))*jnp.log(C["mutation_rate"])

key, subkey = random.split(key)
assert(not (C["enable_waste"] and C["enable_energy"]))
S = {"nutrients": nutrients, "organic_matter": organic_matter, "cell_type_logits": cell_type_logits, "information": information, "weights": weights, "key": subkey}
if(C["enable_waste"]):
    S["waste"] = waste
elif(C["enable_energy"]):
    S["energy"] = energy
if(C["enable_metaevolution"]):
    S["mutation_rate_logits"] = mutation_rate_logits
    
def agent(weights, x, h):
    #minimally gated GRU, note x includes h
    W_f, b_f, W_h, U_h, b_h, W_o, b_o = weights
    f = jax.nn.sigmoid(jnp.dot(W_f, x)+b_f)
    h_hat = jax.nn.tanh(jnp.dot(W_h, x)+jnp.dot(U_h, f*h)+b_h)
    h = (1-f)*h+f*h_hat
    y = jnp.dot(W_o, h)+b_o
    return y, h

def agents(S):
    #TODO: double check shapes and axis alignment
    weights = S["weights"]
    information = S["information"] 
    nutrients = S["nutrients"]/C["nutrient_cap"]
    cell_type_weights = S["organic_matter"]/C["organic_matter_hardcap"] * jax.nn.softmax(S["cell_type_logits"], axis=0)
    def get_input():
        info = information
        nut = nutrients
        cell_type_weight = cell_type_weights
        local_input = jnp.concatenate([info, jnp.expand_dims(nut,axis=0), cell_type_weight], axis=0)
        left_input = jnp.roll(local_input, 1, axis=1)
        right_input = jnp.roll(local_input, -1, axis=1)
        up_input = jnp.roll(local_input, 1, axis=2)
        down_input = jnp.roll(local_input, -1, axis=2)
        return jnp.concatenate([local_input, left_input, right_input, up_input, down_input], axis=0)
    x = get_input()
    y, h = vmap(vmap(agent,in_axes=(-1,-1,-1)),in_axes=(-1,-1,-1))(weights, x, information)
    #above operation ends up transposing horizontal and vertical axes, so transpose back and also move channel axis to the beginning
    y = y.transpose((2,1,0))
    h = h.transpose((2,1,0))
    return y, h

#TODO: thoroughly check directions
def step_sim(S, move=False, steps=1):
    def step_loop(S,_):
        y, h = agents(S)
        metrics = {}

        S["information"] = h
    
        # split y into target cell type, target cell, and cling
        target_cell_type_logits = y[:num_cell_types]
        target_cell_logits = y[num_cell_types:num_cell_types+5]
        target_cell_type = jax.nn.softmax(target_cell_type_logits, axis=0)
        #Only need cling if mover is enabled
        if("mover" in enabled_types):
            cling_logit = y[-2]
        spread_logit = y[-1]

        S["cell_type_logits"] = S["cell_type_logits"] + C["type_change_rate"] * (target_cell_type-1/num_cell_types)
        
        target_cell_type_entropy = -jnp.sum(jax.nn.log_softmax(target_cell_type_logits, axis=0) * jax.nn.softmax(target_cell_type_logits, axis=0), axis=0)
        metrics["target_cell_type_entropy"] = jnp.mean(target_cell_type_entropy)

        #clip cell type logits to prevent overflow
        S["cell_type_logits"] = jnp.clip(S["cell_type_logits"], -C["max_cell_type_logit"], C["max_cell_type_logit"])

        cell_type_weights = jnp.clip(S["organic_matter"]/C["organic_matter_softcap"],0,1) * jax.nn.softmax(S["cell_type_logits"], axis=0)**C["type_sharpness"]

        target_cell_weights = jax.nn.softmax(target_cell_logits, axis=0)
        
        #######
        #ENZYME
        #######
        #Compute enzyme damage for each cell
        if("enzyme" in enabled_types):
            enzyme_weight = cell_type_weights[enzyme]
            dmg_from_up = jnp.roll(target_cell_weights[0]*cell_type_weights[enzyme], 1, axis=0)
            dmg_from_down = jnp.roll(target_cell_weights[1]*cell_type_weights[enzyme], -1, axis=0)
            dmg_from_left = jnp.roll(target_cell_weights[2]*cell_type_weights[enzyme], 1, axis=1)
            dmg_from_right = jnp.roll(target_cell_weights[3]*cell_type_weights[enzyme], -1, axis=1)

            #WALL
            if("wall" in enabled_types):
                wall_defense = (1-cell_type_weights[wall])+C["wall_defense_factor"]*cell_type_weights[wall]
            else:
                wall_defense = 1
            total_dmg = (dmg_from_up + dmg_from_down + dmg_from_left + dmg_from_right)*C["enzyme_damage_rate"]*wall_defense*S["organic_matter"]
            S["organic_matter"] = S["organic_matter"] - total_dmg
            S["nutrients"] = S["nutrients"] + total_dmg

        ###########
        #PHOTOSYNTH
        ###########
        #produce organic matter via photosynthesis and decay
        if("photosynth" in enabled_types):
            photosynth_weight = cell_type_weights[photosynth]
        else:
            photosynth_weight = 0
        #currently don't support energy and waste at the same time
        if(C["enable_waste"]):
            waste_damage_rate = jnp.clip(S["waste"]*C["waste_damage_rate"], 0, 1.0)
            S["organic_matter"] = S["organic_matter"] + (photosynth_weight * C["photosynthesis_rate"]- waste_damage_rate * S["organic_matter"])
        elif(C["enable_energy"]):
            absorbed_energy = photosynth_weight * C["photosynthesis_rate"]*S["energy"]
            S["energy"] = S["energy"] - absorbed_energy
            S["organic_matter"] = S["organic_matter"] + absorbed_energy
            #decay organic matter
            S["organic_matter"] = S["organic_matter"] - C["organic_matter_decay_rate"] * S["organic_matter"]

            #diffuse energy
            padded_energy = jnp.pad(S["energy"], 1, mode="wrap")
            energy_laplacian = jax.scipy.signal.convolve2d(padded_energy, laplace_array, mode="valid")

            energy_flux = C["energy_diffusion_rate"] * energy_laplacian
            
            total_energy = jnp.sum(S["energy"])
            S["energy"] = S["energy"] + energy_flux
            #Clip to zero
            S["energy"] = jnp.clip(S["energy"], 0, None)
            #Normalize
            S["energy"] = S["energy"] * jnp.where(jnp.sum(S["energy"]) > 0, (total_energy / jnp.sum(S["energy"])), 1)

            #generate energy
            S["energy"] = S["energy"] + C["energy_generation_rate"]

            #Clip to cap
            S["energy"] = jnp.clip(S["energy"], None, C["energy_cap"])

        else:
            S["organic_matter"] = S["organic_matter"] + (photosynth_weight * C["photosynthesis_rate"]- C["organic_matter_decay_rate"] * S["organic_matter"])

        #########
        #ABSORBER
        #########
        if("absorber" in enabled_types):
            absorber_weight = cell_type_weights[absorber]
        else:
            absorber_weight = 0

        #absorb nutrients
        absorbed_nutrients = absorber_weight * C["absorber_rate"] * S["nutrients"]
        S["nutrients"] = S["nutrients"] - absorbed_nutrients

        S["organic_matter"] = S["organic_matter"] + absorbed_nutrients

        #generate waste
        if(C["enable_waste"]):
            S["waste"] = S["waste"] + S["organic_matter"] * C["waste_generation_rate"]

        ########
        #CHANNEL
        ########
        #Compute velocity field due to channel cells
        #target_cell layout: [up, down, left, right, no target]
        if("channel" in enabled_types):
            channel_weight = cell_type_weights[channel]
        else:
            channel_weight = 0
        vel_x = (target_cell_weights[3]-target_cell_weights[2])*channel_weight*C["channel_velocity_scale"]
        padded_vel_x = jnp.pad(vel_x, 1, mode="wrap")
        vel_x_grad = jax.scipy.signal.convolve2d(padded_vel_x, hor_sobel_array, mode="valid")
        vel_y = (target_cell_weights[1]-target_cell_weights[0])*channel_weight*C["channel_velocity_scale"]
        padded_vel_y = jnp.pad(vel_y, 1, mode="wrap")
        vel_y_grad = jax.scipy.signal.convolve2d(padded_vel_y, vert_sobel_array, mode="valid")
        
        #diffuse nutrients
        padded_nutrients = jnp.pad(S["nutrients"], 1, mode="wrap")
        nutrient_laplacian = jax.scipy.signal.convolve2d(padded_nutrients, laplace_array, mode="valid")
        hor_nutrient_grad = jax.scipy.signal.convolve2d(padded_nutrients, hor_sobel_array, mode="valid")
        vert_nutrient_grad = jax.scipy.signal.convolve2d(padded_nutrients, vert_sobel_array, mode="valid")
        nutrient_flux = C["nutrient_diffusion_rate"] * nutrient_laplacian+\
                        (-S["nutrients"]*(vel_x_grad+vel_y_grad))+\
                        (-(vel_x*hor_nutrient_grad+vel_y*vert_nutrient_grad))

        total_nutrients = jnp.sum(S["nutrients"])
        S["nutrients"] = S["nutrients"] + nutrient_flux
        #Clip to zero
        S["nutrients"] = jnp.clip(S["nutrients"], 0, None)
        #Normalize
        S["nutrients"] = S["nutrients"] * jnp.where(jnp.sum(S["nutrients"]) > 0, (total_nutrients / jnp.sum(S["nutrients"])), 1)
        #Clip to cap
        S["nutrients"] = jnp.clip(S["nutrients"], None, C["nutrient_cap"])

        if(C["enable_waste"]):
            #diffuse waste
            padded_waste = jnp.pad(S["waste"], 1, mode="wrap")
            waste_laplacian = jax.scipy.signal.convolve2d(padded_waste, laplace_array, mode="valid")
            hor_waste_grad = jax.scipy.signal.convolve2d(padded_waste, hor_sobel_array, mode="valid")
            vert_waste_grad = jax.scipy.signal.convolve2d(padded_waste, vert_sobel_array, mode="valid")
            waste_flux = C["waste_diffusion_rate"] * waste_laplacian+\
                            (-S["waste"]*(vel_x_grad+vel_y_grad))+\
                            (-(vel_x*hor_waste_grad+vel_y*vert_waste_grad))

            total_waste = jnp.sum(S["waste"])
            S["waste"] = S["waste"] + waste_flux
            #Clip to zero
            S["waste"] = jnp.clip(S["waste"], 0, None)
            #Normalize
            S["waste"] = S["waste"] * jnp.where(jnp.sum(S["waste"]) > 0, (total_waste / jnp.sum(S["waste"])), 1)
            #Clip to cap
            S["waste"] = jnp.clip(S["waste"], None, C["waste_cap"])

            #decay waste
            S["waste"] = S["waste"] * (1-C["waste_decay_rate"])

        #diffuse organic matter
        matter_saturation = jnp.clip(S["organic_matter"]/C["organic_matter_softcap"], 0, 1)
        #increased matter saturation leads to increased control over spread, low saturation regesses to default spread rate
        spread = jax.nn.sigmoid(spread_logit)*matter_saturation*C["maximum_spread_rate"]+(1-matter_saturation)*C["default_spread_rate"]
        metrics["spread"] = jax.nn.sigmoid(spread_logit)
        padded_spread = jnp.pad(spread, 1, mode="wrap")

        padded_organic_matter = jnp.pad(S["organic_matter"], 1, mode="wrap")
        neighbor_spread = jax.scipy.signal.convolve2d(padded_spread, neighbor_array, mode="valid")
        neighbor_spread_weighted_organic_matter = jax.scipy.signal.convolve2d(padded_spread*padded_organic_matter, neighbor_array, mode="valid")

        organic_matter_flux = spread*(neighbor_spread_weighted_organic_matter-neighbor_spread*S["organic_matter"])

        #####################Diffuse weights according to all organic matter
        #diffuse weights according to organic matter
        diffusing_matter = spread*S["organic_matter"]
        diffusing_matter_weighted_weights = jax.tree_util.tree_map(lambda x: x*diffusing_matter, S["weights"])
        padded_matter_weighted_weights = jax.tree_util.tree_map(lambda x: jnp.pad(x, ((0,0),)*(len(x.shape)-2)+((1,1),)*2, mode="wrap"), diffusing_matter_weighted_weights)
        leading_shapes = jax.tree_util.tree_map(lambda x: x.shape[:-2], padded_matter_weighted_weights)

        #flatten nonspatial dimensions to make it easier to convolve
        padded_matter_weighted_weights = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+x.shape[-2:]), padded_matter_weighted_weights)
        neighbor_weighted_weights = jax.tree_util.tree_map(lambda x: vmap(partial(jax.scipy.signal.convolve2d, mode="valid"), in_axes=(0,None))(x, neighbor_array), padded_matter_weighted_weights)

        #reshape back to original shape
        neighbor_weighted_weights = jax.tree_util.tree_map(lambda x,y: x.reshape(y+(C["world_size"],C["world_size"])), neighbor_weighted_weights, leading_shapes)

        #normalize weights by total neighbor diffusing matter
        padded_diffusing_matter = jnp.pad(diffusing_matter, 1, mode="wrap")
        neighbor_diffusing_matter = jax.scipy.signal.convolve2d(padded_diffusing_matter, neighbor_array, mode="valid")
        neighbor_weighted_weights = jax.tree_util.tree_map(lambda x: jnp.where(neighbor_diffusing_matter>0,x/neighbor_diffusing_matter,0), neighbor_weighted_weights)

        # flattened_weights = jax.tree_util.tree_map(lambda x: x.reshape((-1,)+x.shape[-2:]), S["weights"])

        diffusing_matter_frac = jnp.where(neighbor_diffusing_matter>0, spread*neighbor_diffusing_matter/(S["organic_matter"]+spread*neighbor_diffusing_matter), 0)
        
        S["weights"] = jax.tree_util.tree_map(lambda x,y: diffusing_matter_frac*x+(1-diffusing_matter_frac)*y, neighbor_weighted_weights, S["weights"])

        if(C["enable_metaevolution"]):
            #diffuse metaevolution logits according to organic matter
            diffusing_matter_weighted_mutation_rate_logits = diffusing_matter*S["mutation_rate_logits"]
            padded_matter_weighted_mutation_rate_logits = jnp.pad(diffusing_matter_weighted_mutation_rate_logits, 1, mode="wrap")

            neighbor_matter_weighted_mutation_rate_logits = jax.scipy.signal.convolve2d(padded_matter_weighted_mutation_rate_logits, neighbor_array, mode="valid")

            neighbor_matter_weighted_mutation_rate_logits = jnp.where(neighbor_diffusing_matter>0, neighbor_matter_weighted_mutation_rate_logits/neighbor_diffusing_matter, 0)

            S["mutation_rate_logits"] = diffusing_matter_frac*neighbor_matter_weighted_mutation_rate_logits+(1-diffusing_matter_frac)*S["mutation_rate_logits"]
        ##############################################################

        # Update total organic matter
        total_organic_matter = jnp.sum(S["organic_matter"])
        S["organic_matter"] = S["organic_matter"] + organic_matter_flux
        #Clip above zero
        S["organic_matter"] = jnp.clip(S["organic_matter"], 0, None)
        #Normalize
        S["organic_matter"] = S["organic_matter"] * jnp.where(jnp.sum(S["organic_matter"]) > 0, (total_organic_matter / jnp.sum(S["organic_matter"])), 1)
        #Clip below hardcap
        S["organic_matter"] = jnp.clip(S["organic_matter"], None, C["organic_matter_hardcap"])

        #randomly seed new organic matter
        S["key"], subkey = random.split(S["key"])
        S["organic_matter"] = S["organic_matter"] + random.uniform(subkey, S["organic_matter"].shape) * C["organic_matter_seed_rate"]

        #mutate weights
        S["key"], subkey = random.split(S["key"])
        if(C["enable_metaevolution"]):
            mutation_rate = jnp.exp(S["mutation_rate_logits"])
            noise_scaling = jnp.sqrt(1-(1-mutation_rate)**2)*C["weight_scale"]
            S["weights"] = jax.tree_util.tree_map(lambda x: x*(1-mutation_rate) + random.normal(subkey, x.shape) * noise_scaling, S["weights"])
            metrics["avg_mutation_rate"] = jnp.mean(mutation_rate)
            S["key"], subkey = random.split(S["key"])
            S["mutation_rate_logits"] = S["mutation_rate_logits"] + random.normal(subkey, S["mutation_rate_logits"].shape) * C["metaevolution_noise_scale"]
        else:
            noise_scaling = jnp.sqrt(1-(1-C["mutation_rate"])**2)*C["weight_scale"]
            S["weights"] = jax.tree_util.tree_map(lambda x: x*(1-C["mutation_rate"]) + random.normal(subkey, x.shape) * noise_scaling, S["weights"])

        return S, metrics
    S, metrics = jax.lax.scan(step_loop, S, None, length=steps)
    #just return the metrics fromt he final step in case we are doing multiple steps
    metrics = {k: v[-1] for k,v in metrics.items()}

    #Note: movement occurs after main loop
    ######
    #MOVER
    ######
    if("mover" in enabled_types):
        y, h = agents(S)
        # metrics = {}

        # S["information"] = h
    
        # split y into target cell type, target cell, and cling
        target_cell_logits = y[num_cell_types:num_cell_types+5]

        cell_type_weights = jnp.clip(S["organic_matter"]/C["organic_matter_softcap"],0,1) * jax.nn.softmax(S["cell_type_logits"], axis=0)**C["type_sharpness"]
        #Only need cling if mover is enabled
        if("mover" in enabled_types):
            cling_logit = y[-2]

        # probability to cancel movement proportional to cling of target cell
        S["key"], subkey = random.split(S["key"])
        cling_weight = jax.nn.sigmoid(cling_logit)*jnp.clip(S["organic_matter"]/C["organic_matter_softcap"], 0, 1)
        metrics["cling"] = jax.nn.sigmoid(cling_logit)
    if("mover" in enabled_types and move):
        S["key"], subkey = random.split(S["key"])
        permuted_movement_indices = jax.random.permutation(subkey, jnp.arange(6))

        direction_mask = jnp.expand_dims(jnp.array([1,1,1,1,0]), axis=(1,2))
        movement_scale = cell_type_weights[mover]*C["movement_rate"]
        metrics["max_movement_scale"] = jnp.max(movement_scale)

        S["cell_type_logits"] = jnp.clip(S["cell_type_logits"], -C["max_cell_type_logit"], C["max_cell_type_logit"])

        cell_type_weights = jnp.clip(S["organic_matter"]/C["organic_matter_softcap"],0,1) * jax.nn.softmax(S["cell_type_logits"], axis=0)**C["type_sharpness"]

        target_cell_weights = jax.nn.softmax(target_cell_logits, axis=0)

        total_movement_weight = jnp.sum(target_cell_weights*direction_mask, axis=0)
        target_movement_weights = jnp.clip(target_cell_weights*movement_scale*direction_mask+(1-direction_mask)*(target_cell_weights+(1-movement_scale)*total_movement_weight), 0.0001, None)
        S["key"], subkey = random.split(S["key"])
        target_movement_cell = jax.random.categorical(subkey, jnp.log(target_movement_weights), axis=0)

        cling = random.uniform(subkey, cling_logit.shape) < cling_weight
        cling_up = (target_movement_cell==0)*jnp.roll(cling, -1, axis=-2)
        cling_down = (target_movement_cell==1)*jnp.roll(cling, 1, axis=-2)
        cling_left = (target_movement_cell==2)*jnp.roll(cling, 1, axis=-1)
        cling_right = (target_movement_cell==3)*jnp.roll(cling, -1, axis=-1)
        cling_mask = cling_up+cling_down+cling_left+cling_right
        target_movement_cell = target_movement_cell*(1-cling_mask)+4*cling_mask

        #update state based on movement computed at masked locations
        #TODO: check this function more thoroughly
        def movement_loop_function(carry,mask_index):
            S,target_cells,masks = carry
            mask = masks[mask_index]
            #Swap all of these state elements
            swap_keys = ["organic_matter", "nutrients", "weights", "information", "cell_type_logits", "information", "mutation_rate_logits"]
            swap_keys = [k for k in swap_keys if k in S.keys()]
            target_up = (target_cells==0)*mask
            target_down = (target_cells==1)*mask
            target_left = (target_cells==2)*mask
            target_right = (target_cells==3)*mask

            update_from_up = jnp.roll(target_up, 1, axis=-2)+target_up
            update_from_down = jnp.roll(target_down, -1, axis=-2)+target_down
            update_from_left = jnp.roll(target_left, -1, axis=-1)+target_left
            update_from_right = jnp.roll(target_right, 1, axis=-1)+target_right
            update_mask = (update_from_up+update_from_down+update_from_left+update_from_right)

            for key in swap_keys:
                from_up = jax.tree_util.tree_map(lambda x: jnp.roll(target_up*x, 1, axis=-2)+target_up*jnp.roll(x, -1, axis=-2), S[key])
                from_down = jax.tree_util.tree_map(lambda x: jnp.roll(target_down*x, -1, axis=-2)+target_down*jnp.roll(x, 1, axis=-2), S[key])
                from_left = jax.tree_util.tree_map(lambda x: jnp.roll(target_left*x, -1, axis=-1)+target_left*jnp.roll(x, 1, axis=-1), S[key])
                from_right = jax.tree_util.tree_map(lambda x: jnp.roll(target_right*x, 1, axis=-1)+target_right*jnp.roll(x, -1, axis=-1), S[key])
                update = jax.tree_util.tree_map(lambda x,y,z,w: x+y+z+w, from_up, from_down, from_left, from_right)
                S[key] = jax.tree_util.tree_map(lambda x,y: x*(1-update_mask)+y, S[key], update)
            
            carry = (S, target_cells, masks)
            return carry, None
        carry, _ = jax.lax.scan(movement_loop_function, (S, target_movement_cell, movement_partition), permuted_movement_indices)
        S, _, _ = carry

    return S, metrics
step_sim = jit(step_sim, static_argnums=(1,2))

if(visualize):
    visualizer = Visualizer(C)

if(enable_video):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_file_name, fourcc, 30, (C["screen_size"], C["screen_size"]))

step = 0
print_freq = 10
done = False
if(args.loadfile):
    #check if loadfile exists
    if(not os.path.isfile(args.loadfile)):
        print("Warning: loadfile does not exist, starting from scratch")
    else:
        with open(args.loadfile, "rb") as f:
            S, step = pkl.load(f)

if(enable_record):
    #record does not store weights
    record = {"config":C,"states":[], "metrics":[], "start_step":step}
    recorded_state_keys = ["organic_matter", "nutrients", "cell_type_logits"]
    if(C["enable_waste"]):
        recorded_state_keys.append("waste")
    if(C["enable_energy"]):
        recorded_state_keys.append("energy")
    recorded_metric_keys = ["spread"]
    if("mover" in enabled_types):
        recorded_metric_keys.append("cling")
    if(C["enable_metaevolution"]):
        recorded_metric_keys.append("avg_mutation_rate")
start_time = time()
avg_fps = 0.0
metrics = None
normalize = True
while not done:
    if(visualize and step%C["display_frequency"] == 0):
       done = visualizer.update(S, step, metrics)

    if("mover" in enabled_types):
        S, metrics = step_sim(S, step%C["movement_frequency"]==0, C["updates_per_step"])
    else:
        S, metrics = step_sim(S, False, C["updates_per_step"])
    step+=1
    
    if(step%10 == 0):
        jax.block_until_ready(S)
        avg_fps = 0.9*avg_fps + 0.1*(10/(time()-start_time))
        # fps = 10/(time()-start_time)
        print("fps: "+str(avg_fps)[:7]+" | step: "+str(step), end="\r")
        start_time = time()

    if(enable_video):
        if step % video_frame_frequency == 0:
            #save video frame
            img = visualizer.cell_type_img(S)
            surf = pg.surfarray.make_surface(np.asarray(img))
            surf = pg.transform.scale(surf, (C["screen_size"], C["screen_size"]))
            surface_array = pg.surfarray.array3d(surf)
            frame = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

    if(enable_checkpoint):
        if step % checkpoint_frequency == 0:
            #save checkpoint
            with open(checkpoint_file_prefix+str(step)+".pkl", "wb") as f:
                pkl.dump((S,step), f)

    if(enable_save):
        if step % save_frequency == 0:
            #save state
            with open(save_file_name, "wb") as f:
                pkl.dump((S,step), f)
    
    if(enable_record):
        if step % record_frequency == 0:
            #record state
            recorded_state = {k: S[k] for k in recorded_state_keys}
            recorded_metric = {k: metrics[k] for k in recorded_metric_keys}
            record["states"].append(recorded_state)
            record["metrics"].append(recorded_metric)
        if step%dump_record_frequency == 0:
            #dump record
            with open(record_file_prefix+str(step)+".pkl", "wb") as f:
                pkl.dump(record, f)
                record = {"config":C, "states":[], "metrics":[], "start_step":step}

if(enable_record):
    #dump record
    with open(record_file_prefix+str(step)+".pkl", "wb") as f:
        pkl.dump(record, f)

if(enable_save):
    with open(save_file_name, "wb") as f:
        pkl.dump((S,step), f)

if(enable_video):
    video_writer.release()