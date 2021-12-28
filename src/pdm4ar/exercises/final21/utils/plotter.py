import matplotlib.pyplot as plt
import matplotlib
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders


def plot_scenario(dg_scenario, goal) -> None:
    matplotlib.use('TkAgg')
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_facecolor('k')
    ax.set_aspect("equal")
    plt.show()
