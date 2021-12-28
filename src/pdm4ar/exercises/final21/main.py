from pdm4ar.exercises_def.final21.scenario import get_dgscenario

from utils.plotter import plot_scenario


def main():
    # get scenario
    dg_scenario, goal, _ = get_dgscenario()

    # plot scenario
    plot_scenario(dg_scenario, goal)


if __name__ == '__main__':
    main()
