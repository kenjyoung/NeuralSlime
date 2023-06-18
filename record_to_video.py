import argparse
import pygame as pg
import numpy as np
import jax
import jax.numpy as jnp
import pickle as pkl
from math import ceil
import cv2
from visualize import Visualizer
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--loadfile", help="specify a file to load", default=None)
parser.add_argument("--savefile", help="specify a file to save", default=None)
parser.add_argument("--screen_size", help="specify a screen size", default=256)
parser.add_argument("--frame_skip", help="specify a frame skip", default=1, type=int)
args = parser.parse_args()

def to_numpy(x):
    return jax.tree_map(lambda y: np.array(y), x)

if os.path.isdir(args.loadfile):
    files = os.listdir(args.loadfile)
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
else:
    files = [args.loadfile]

record = pkl.load(open(os.path.join(args.loadfile, files[0]), "rb"))
C = record["config"]

visualizer = Visualizer(C, visualize=False)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
screen_size = args.screen_size
video_writer = cv2.VideoWriter(args.savefile, fourcc, 30, (int(C["screen_size"]), int(C["screen_size"])))

i=0
for f in tqdm(files):
    record = to_numpy(pkl.load(open(os.path.join(args.loadfile, f), "rb")))
    num_steps = len(record["states"])
    for S in tqdm(record["states"]):
        if i % args.frame_skip == 0:
            img = visualizer.cell_type_img(S)
            surf = pg.surfarray.make_surface(np.asarray(img))
            surf = pg.transform.scale(surf, (C["screen_size"], C["screen_size"]))
            surface_array = pg.surfarray.array3d(surf)
            frame = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        i+=1

video_writer.release()