import argparse
import pygame as pg
import numpy as np
import jax
import jax.numpy as jnp
import pickle as pkl
from math import ceil
import os
import threading
from queue import Queue

from visualize import Visualizer

from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--loadfile", help="specify a file to load", default=None)
args = parser.parse_args()

def to_numpy(x):
    return jax.tree_map(lambda y: np.array(y), x)

class RecordStreamer:
    def __init__(self, record_directory, window_size=5):
        self.record_directory = record_directory
        self.window_size = window_size
        files = os.listdir(record_directory)
        self.files = sorted(files, key=lambda x: int(x.split(".")[0]))
        self.file_index = 0
        self.num_files = len(self.files)
        self.done = False
        self.current_record = self.load_record(self.files[self.file_index])
        self.loaded_records = [self.current_record]
        self.loaded_indices = [self.file_index]
        self.config = self.current_record["config"]
        self.record_frequency = self.config["bookkeeping_config"]["record_frequency"]
        start_step = self.current_record["start_step"]
        self.curr_file_range = (start_step//self.record_frequency,
                                start_step//self.record_frequency+len(self.current_record["states"]))
        self.lower_bound = start_step//self.record_frequency
        last_record = self.load_record(self.files[-1])
        self.upper_bound = last_record["start_step"]//self.record_frequency + len(last_record["states"])
        
        self.lock = threading.Lock()
        self.load_records_in_background()
    
    def load_record(self, file):
        with open(os.path.join(self.record_directory, file), "rb") as f:
            record = pkl.load(f)
            record["states"] = to_numpy(record["states"])
            record["metrics"] = to_numpy(record["metrics"])
        return record
    
    def load_records_in_background(self):
        # preload the next and previous records in the background
        def load_records():
            while not self.done:
                index = self.file_index
                if index not in self.loaded_indices:
                    record = self.load_record(self.files[index])
                    with self.lock:
                        self.loaded_records.append(record)
                        self.loaded_indices.append(index)
                # Only load previous or next record if current was loaded
                else:
                    for i in range(1,self.window_size):
                        index = self.file_index-i
                        if(index>=0 and (index not in self.loaded_indices)):
                            record = self.load_record(self.files[index])
                            with self.lock:
                                self.loaded_records.append(record)
                                self.loaded_indices.append(index)
                            #break after loading one previous record
                            break
                        index = self.file_index+i
                        if(index<self.num_files and (index not in self.loaded_indices)):
                            record = self.load_record(self.files[index])
                            with self.lock:
                                self.loaded_records.append(record)
                                self.loaded_indices.append(index)
                            #break after loading one next record
                            break
                sleep(0.01)
                while(len(self.loaded_indices)>self.window_size*2+1):
                    with self.lock:
                        self.loaded_indices.pop(0)
                        self.loaded_records.pop(0)
        self.thread = threading.Thread(target=load_records, daemon=True).start()

    def get_record(self, key, i=None):
        if(i is None):
            return self.current_record[key]
        else:
            if(not (self.curr_file_range[0]<=i<self.curr_file_range[1])):
                while(not (self.curr_file_range[0]<=i<self.curr_file_range[1])):
                    if(i>=self.curr_file_range[1]):
                        self.file_index+=1
                    elif(i<self.curr_file_range[0]):
                        self.file_index-=1
                    while(self.file_index not in self.loaded_indices):
                        sleep(0.01)
                    assert(self.file_index in self.loaded_indices)
                    self.current_record = self.loaded_records[self.loaded_indices.index(self.file_index)]
                    start_step = self.current_record["start_step"]
                    self.curr_file_range = (start_step//self.record_frequency,
                                            start_step//self.record_frequency+len(self.current_record["states"]))
            return self.current_record[key][i-self.curr_file_range[0]]
        
    def finish(self):
        self.done = True

#check if file is a directory
if os.path.isdir(args.loadfile):
    streamer = RecordStreamer(args.loadfile)
    streaming = True
    C = streamer.get_record("config")
else:
    with open(args.loadfile, "rb") as f:
        record = pkl.load(f)
    streaming = False
    C = record["config"]
    
visualizer = Visualizer(C, control_frame_time=True)

pg.init()
display = pg.display.set_mode((C["screen_size"], C["screen_size"]))
font = pg.font.Font('freesansbold.ttf', 24)

# num_steps = len(record["states"])
# start_step = record["start_step"]
min_wait = 0.05
done = False
i = 0
while not done:
    if(streaming):
        S = streamer.get_record("states", i)
        lower_bound = streamer.lower_bound
        upper_bound = streamer.upper_bound
        metrics = streamer.get_record("metrics", i)
    else:
        S = record["states"][i]
        metrics = record["metrics"][i]
    done, frame_time = visualizer.update(S, i, metrics)
    # frame_time = frame_times[frame_time_index]
    # show_frametime(frame_time)
    pg.display.update()
    wait_time = max(min_wait, abs(frame_time))
    pg.time.wait(int(wait_time*1000))
    if(frame_time>0):
        frame_skip = ceil(min_wait/abs(frame_time))
        i+=frame_skip
        # i = min(i, num_steps-1)
        i = min(i, upper_bound-1)
    elif(frame_time<0):
        frame_skip = ceil(min_wait/abs(frame_time))
        i-=frame_skip
        # i = max(i, 0)
        i = max(i, lower_bound)
    if(done):
        streamer.finish()
        break
