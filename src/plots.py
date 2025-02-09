import matplotlib.pyplot as plt


def plot_fitness_history(stats) -> None:

    gens = [stat[0] for stat in stats]
    mins_fit = [stat[1]["min"] for stat in stats]
    avgs_fit = [stat[1]["avg"] for stat in stats]

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(gens, mins_fit)
    axs[1].plot(gens, avgs_fit)
    plt.show()


def plot_func(func, x, y) -> None:

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="3d"))
    ax.plot_surface(x[0, :], x[1, :], func(*x))
    ax.plot_surface(x[0, :], x[1, :], y)
    plt.show()
