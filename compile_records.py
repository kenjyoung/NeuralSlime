import argparse
import pygame as pg
import numpy as np
import jax
import jax.numpy as jnp
import pickle as pkl

parser = argparse.ArgumentParser()
#provide a list of all record files to load
parser.add_argument("--loadfiles", help="specify a file to load", nargs="+", default=None)
parser.add_argument("--savefile", help="specify a file to save", default=None)
args = parser.parse_args()

def to_numpy(x):
    return jax.tree_map(lambda y: np.array(y), x)

records = []
for loadfile in args.loadfiles:
    with open(loadfile, "rb") as f:
        record = pkl.load(f)
        record["states"] = to_numpy(record["states"])
        record["metrics"] = to_numpy(record["metrics"])
        # records.append(to_numpy(pkl.load(f)))
        records.append(record)

start_steps = [record["start_step"] for record in records]

print("start steps:")
for start_step in start_steps:
    print(start_step)

records = [x for _,x in sorted(zip(start_steps,records))]

#print all start steps
print("start steps:")
for record in records:
    print(record["start_step"])

record = {"config":records[0]["config"],"states":[], "metrics":[], "start_step":records[0]["start_step"]}
for i in range(len(records)):
    record["states"].extend(records[i]["states"])
    record["metrics"].extend(records[i]["metrics"])

with open(args.savefile, "wb") as f:
    pkl.dump(record, f)