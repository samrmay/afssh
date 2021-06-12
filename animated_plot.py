import numpy as np
import models as models
import stopping_functions as fcns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_1d(fssh, fig, ax, stopping_function=fcns.f(-6, 6), min_x=-10, max_x=10, interval=50, num_points=400):
    x_linspace = np.linspace(min_x, max_x, num_points)
    model = fssh.model
    models.plot_1d(ax, model, x_linspace)

    point, = ax.plot(fssh.r, model.get_adiabatic_energy(
        fssh.r)[fssh.e_state], 'ro')
    v_text = plt.text(-5, 0, fssh.v)

    def animate(_):
        if not fssh.step(stopping_function):
            return None
        point.set_xdata(fssh.r)
        point.set_ydata(model.get_adiabatic_energy(fssh.r)[fssh.e_state])
        v_text.set_text(fssh.v)
        return point, v_text,

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)
    return ani


def animate_2d(fssh, fig, ax, x_linspace, y_linspace, stopping_function=fcns.f(-6, 6), interval=50, num_points=400, colors=["black", "red"]):
    model = fssh.model
    models.plot_2d(ax, model, x_linspace, y_linspace, colors)

    point, = ax.plot(fssh.r[0], fssh.r[1], model.get_adiabatic_energy(
        fssh.r)[fssh.e_state], marker='o')

    def animate(_):
        if not fssh.step(stopping_function):
            return None
        point.set_data(fssh.r)
        point.set_3d_properties(
            model.get_adiabatic_energy(fssh.r)[fssh.e_state])
        return point,

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)

    return ani
