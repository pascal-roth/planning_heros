planning_heros
==============
Planning and Decision Making for Autonomuous Robots Project


|                         |                      |
| ----------------------- |----------------------|
| Required Python Version | `3.9`                |
| Deadline                | 31.12.2021, 23:59:59 |



Solution approach
-----------------
- RRT* - run for each timestep:
  - Create Motion Primitives 
  - KD-tree to compute distances
  - Caution, use box-distance measure
  - Collision checking via safety certificates
  - For state x choose best motion primitive to get to new state y
  - Optimal path in RRT* graph
  - Cost function
- Time required for each motion primitive




Progress Bar
=========

```
8==D
   0---------------------------------------------1
``` 
