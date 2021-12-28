# Planning Heros

Planning and Decision Making for Autonomous Robots Project, Fall 2021 @ETHZ

## Installation

Required python version:
| | |
| ----------------------- |----------------------|
| Required Python Version | `3.9` |
| Deadline | 31.12.2021, 23:59:59 |

Running from the console, assuming you run this from the project root:

```bash
# install requirements
$ pip install -r requirements.txt
# export PYTHONPATH to run from console
$ export PYTHONPATH="$PWD/src"
# run app_main.py
$ python src/pdm4ar/app_main.py
```

Running from the provided Docker container (Requires a recent docker version to be installed).

```bash
$ make run-final21
```

## Solution approach

- RRT\* - run for each timestep:
  - Create Motion Primitives
  - KD-tree to compute distances
  - Caution, use box-distance measure
  - Collision checking via safety certificates
  - For state x choose best motion primitive to get to new state y
  - Optimal path in RRT\* graph
  - Cost function
- Time required for each motion primitive


# Note

The given setup has been evaluated on two additional, custom dgscenaries. In order to enable them, set the "custom_cases" flag in /src/pdma4ar/exercises_def/final21/ex.py:get_final21" to True

# Progress Bar

```
8==========================================D
0---------------------------------------------1
```
