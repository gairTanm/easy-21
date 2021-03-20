"""
Plotting code taken from https://github.com/analog-rl/
"""
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_value_function(agent, title="Value Function", generate_gif=False, train_steps=None):
    fig = plt.figure(title, figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    V = agent.get_value_function()
    if generate_gif:
        print('gif will be saved as %s' % title)

    def plot_frame(ax):
        min_x = 1
        max_x = V.shape[0]
        min_y = 1
        max_y = V.shape[1]

        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)

        X, Y = np.meshgrid(x_range, y_range)

        def get_stat_val(x, y):
            return V[x, y]

        Z = get_stat_val(X, Y)

        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.get_cmap("coolwarm"), linewidth=0,
                               antialiased=False)

    def animate(frame):
        ax.clear()
        surf = plot_frame(ax)
        if generate_gif:
            i = agent.iterations
            if train_steps is None:
                step_size = int(min(max(1, agent.iterations), 2 ** 16))
            else:
                step_size = train_steps

            agent.train(step_size)
            plt.title('%s MC score: %s frame: %s' % (title, float(agent.wins) / agent.iterations * 100, frame))
        else:
            plt.title(title)

        fig.canvas.draw()
        return surf

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
    if generate_gif:
        ani.save(title + '.gif', writer='imagemagick', fps=3)
    else:
        plt.show()


def plot_error_vs_episode(sqrt_error, lambdas, train_steps=1000000, eval_steps=1000,
                          title='SQRT error VS episode number', save_as_file=False):
    assert eval_steps != 0
    x_range = np.arange(0, train_steps, eval_steps)
    assert len(sqrt_error) == len(lambdas)
    for e in sqrt_error:
        assert len(list(x_range)) == len(e)
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)

    for i in range(len(sqrt_error) - 1, -1, -1):
        ax.plot(x_range, sqrt_error[i], label='lambda %.2f' % lambdas[i])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_as_file:
        plt.savefig(title)
    plt.show()


def plot_error_vs_lambda(sqrt_error, lambdas, title='SQRT error vs lambda', save_as_file=False):
    assert len(sqrt_error) == len(lambdas)
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)
    y = [s[-1] for s in sqrt_error]
    ax.plot(lambdas, y)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_as_file:
        plt.savefig(title)
    plt.show()
