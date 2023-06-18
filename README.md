# Neural Slime
Neural Slime is an experimental artificial life simulator written in JAX which takes place on a 2D grid. Each grid cell contains a small recurrent neural network which controls the cell's behavior. Network weights are mutated randomly over time and diffuse to neighboring cells along with organic matter such that those cells which produce the most organic matter will be most successful in spreading their weights. The operation of each cell is parallelized using JAX and best run on a GPU. This project was partially inspired by the work of [Mordvintsev et al.](https://distill.pub/2020/selforg/) [1] and [Gregor et al.](https://arxiv.org/abs/2101.07627) [2].

<p align="center">
  <a href="https://www.youtube.com/watch?v=CAl4BnHLzHk">
    <img src="img/example.gif" alt="Alt Text" width="400">
  </a>
</p>

## Usage
To run the simulator you will need JAX, pygame, and opencv. Pygame is not needed if visualization is disabled and opencv is not needed if video is disabled in config.json). 

To run do:
```
python3 sim.py
```

By defualt this will launch a visualizer of the simulation as it runs. Three command line arguments are supported:
- **--config** - Path to config file (default: config.json)
- **--seed** - Random seed (default: initialized from time)
- **--loadfile** - Path to load simulator from (default: None)

The simulator will save its state to a file save.pkl every 1000 steps when using the default config. It will also save a checkpoint to checkpoint/n.pkl where n is the number of steps which is identical to the save file but unlike save it not overwritten. The simulator can be loaded from a save file using the --loadfile argument. By default, the simulator will also write a video to output.avi every 1000 steps. Pressing Q at any time will quit the simulator and save the current state and record.

Finally, the simulator will dump a record file to record/n.pkl which contains a partial simulator state on every 10th step. These files can be viewed by calling:
```
python3 replay_record.py --loadfile record
```
This will launch an interface similar to the realtime visualizer but which can also be sped up, slowed down or reversed with the a and d keys.

There are a large number of configuration options that can be tweaked in config.json (or another config file passed as an argument). Some of the configuration options are disabled entirely by default such as everything relating to "waste" and may or may not work as intended.

## Cell types
Each cell has a type which is a weighted mixture of 6 primitive types:
- **photosynth** - Photosynthetic cells which generates organic matter form energy over time. Green in visualizer.
- **enzyme** - Which converts organic matter into nutrients which diffuse over time. Red in visualizer.
- **absorber** - Which absorbs nutrients from the environment and converts them into organic matter. Purple in visualizer.
- **channel** - Which pushes nutrients in chosen directions. Yellow in visualizer.
- **wall** - Which resists damage from enzymes. Grey in visualizer.
- **mover** - Which can swap places with adjacent cells entirely to facilitate movement. Blue in visualizer.

## Networks and Control
Each cell contains a tiny GRU network with its own weights. The network input consists of cell type, organic matter, nutrients, and the GRU hidden state in the cell and each of its cardinal neighbors. This setup allows cells to transmit information to their neighbors over time via the GRU state. The network output consists of an updated GRU state, a targeted cell type, a targeted direction, and a "cling" and "spread" value.

The output target cell type will slowly change the cells type overtime, the output target direction will have different effects for different types of cells. For the enzyme it determines how damage is distributed to neighboring cells, for the channel it determines which direction nutrients are pushed, for the mover it determines which direction the cell will move. There is also a "no-op" direction which will avoid applying the cells effect to any neighboring direction. For photosynth, wall and absorber, the target direction has no effect. The cling value resists movement in order to prevent cells from being pushed around by movers if they don't want to be. The spread value determines the rate at which a cells organic matter will move into its neighbors as well as the rate at which its neighbors matter will move into it.

## Visualizer
The visualizer is a simple pygame application which displays various views of the grid and allows the user to click on individual cells to see a summary of their status. The visualizer can be disabled by setting "visualize" to false in config.json.

Available view modes can be cycled through using the left and right arrow keys, as well as accessed by pressing the number keys 1-9. The available view modes are:
- **cell type** - Displays the type of each cell as a color photosynth=green, enzyme=red, absorber=purple, channel=yellow, wall=grey, mover=blue. There is some ambiguity as the cells have a weighted mixture of types displayed as a weighted mixture of colors, for example a cell which is 50% photosynth and 50% absorber will appear as a mix of green and purple, which ends up as a shade of grey very similar to a pure wall cell.
- **organic matter** - Displays the amount of organic matter in each cell as a shade of red.
- **nutrients** - Displays the amount of nutrients in each cell as a shade of green.
- **organic matter and nutrients and energy** - Displays the amount of organic matter in each cell as a shade of red, the amount of nutrients in each cell as a shade of green and the amount of energy as a shade of blue.
- **cling** - Displays the amount of cling in each cell as a shade of green.
- **spread** - Displays the amount of cling in each cell as a shade of green.
- **individual cell type** - Displays the amount of organic matter in each cell as a shade of red and the amount of a specific type of cell as a shade of green. The specific type of cell displayed can be changed by pressing the up and down arrow keys.
- **stats** - Displays some numerical summary statistics regarding the simulator state including the total organic matter, total nutrients, and fraction of each cell type.
- **no view** - Displays nothing. Included because it slightly speeds up the simulator when no view is being displayed.

At any time the user can also click on a cell which will print a brief summary of its status to the terminal.

Pressing "n" will toggle normalization which determines whether certain views will be scaled proportional to the amount of organic matter in each cell.

Pressuing space bar will jump to no view.

The visualizer can be disabled entirely in config.json by setting "visualize" to false.

## References
[1] Mordvintsev, A., Randazzo, E., Niklasson, E., Levin, M., & Greydanus, S. (2020). Thread: Differentiable Self-organizing Systems. Distill. doi:10.23915/distill.00027

[2] Gregor, K., & Besse, F. (2021). Self-Organizing Intelligent Matter: A blueprint for an AI generating algorithm. arXiv e-prints, arXiv-2101.07627.
