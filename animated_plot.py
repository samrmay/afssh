import numpy as np
import models as models
import stopping_functions as fcns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

f = fcns.basic_1d(-6, 6)


def animate_1d(fssh, fig, ax, stopping_function=f, min_x=-10, max_x=10, interval=50, num_points=400):
    x_linspace = np.linspace(min_x, max_x, num_points)
    model = fssh.model
    models.plot_1d(ax, model, x_linspace)

    point, = ax.plot(fssh.r[0], model.get_adiabatic_energy(
        fssh.r[0])[fssh.lam], 'ro')
    v_text = plt.text(-5, 0, fssh.v[0])

    def animate(_):
        if not fssh.step(fssh.dt_c):
            return None
        point.set_xdata(fssh.r[0])
        point.set_ydata(model.get_adiabatic_energy(fssh.r[0])[fssh.lam])
        v_text.set_text(fssh.v[0])
        return point, v_text,

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)
    return ani


def animate_2d(fssh, fig, ax, x_linspace, y_linspace, stopping_function=f, interval=50, num_points=400, colors=["black", "red"]):
    model = fssh.model
    models.plot_2d(ax, model, x_linspace, y_linspace, colors)

    point, = ax.plot(fssh.r[0], fssh.r[1], model.get_adiabatic_energy(
        fssh.r)[fssh.lam], marker='o')

    def animate(_):
        if not fssh.step(fssh.dt_c):
            return None
        point.set_data(fssh.r)
        point.set_3d_properties(
            model.get_adiabatic_energy(fssh.r)[fssh.lam])
        return point,

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)

    return ani
