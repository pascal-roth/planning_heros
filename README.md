# Planning Heros

Planning and Decision Making for Autonomous Robots Project, Fall 2021 @ETHZ

## Installation

Required python version:
| Item | Value |
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

## Generated Solutions

Below you can find some visualizations of the scenarios we used to evaluate our approach.

### Static Scenario:

![static scenario](gifs/PDM4AR-final21-staticenvironment0-PDM4AR-EpisodeVisualisation-figure1-Animation.gif)

```
EpisodeOutcome(
	goal_reached        =	    1,
	has_collided        =	    0,
	distance_travelled  =	133.38,
	episode_duration    =	10.68,
	actuation_effort    =	11.03,
	max_acc_lat         =	 9.63,
	max_acc_long        =	23.56,
	avg_computation_time=	 0.10,

)
```

### Dynamic Scenario:

![static scenario](gifs/PDM4AR-final21-dynamicenvironment1-PDM4AR-EpisodeVisualisation-figure1-Animation.gif)

```
EpisodeOutcome(
	goal_reached        =	    1,
	has_collided        =	    0,
	distance_travelled  =	136.09,
	episode_duration    =	12.30,
	actuation_effort    =	10.77,
	max_acc_lat         =	 9.63,
	max_acc_long        =	20.21,
	avg_computation_time=	 0.36,

)

```

### Custom Dynamic Scenario 1:

![static scenario](gifs/PDM4AR-final21-dynamicenvironmentcustom12-PDM4AR-EpisodeVisualisation-figure1-Animation.gif)

```
EpisodeOutcome(
	goal_reached        =	    1,
	has_collided        =	    0,
	distance_travelled  =	134.12,
	episode_duration    =	 9.57,
	actuation_effort    =	11.25,
	max_acc_lat         =	11.65,
	max_acc_long        =	22.88,
	avg_computation_time=	 0.55,

)
```

### Custom Dynamic Scenario 2:

![static scenario](gifs/PDM4AR-final21-dynamicenvironmentcustom23-PDM4AR-EpisodeVisualisation-figure1-Animation.gif)

```
EpisodeOutcome(
	goal_reached        =	    1,
	has_collided        =	    0,
	distance_travelled  =	145.94,
	episode_duration    =	16.29,
	actuation_effort    =	 8.31,
	max_acc_lat         =	42.71,
	max_acc_long        =	42.71,
	avg_computation_time=	 1.18,

)
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

## Note on running additional scenarios

The given setup has been evaluated on two additional, custom dgscenaries. In order to enable them, set the "custom_cases" flag in `/src/pdma4ar/exercises_def/final21/ex.py:get_final21"` to `True`.

## Progress Bar

```
==============================================>
0---------------------------------------------1
```
