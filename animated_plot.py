import numpy as np
import models as models
import stopping_functions as fcns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

f = fcns.basic_1d(-8, 8)


def animate_1d(fssh, fig, ax, stopping_function=f, min_x=-10, max_x=10, interval=.5, num_points=400):
    x_linspace = np.linspace(min_x, max_x, num_points)
    model = fssh.model
    models.plot_1d(ax, model, x_linspace)

    U = model.get_adiabatic_energy(fssh.r[0])[fssh.lam]
    v = fssh.v[0]
    point, = ax.plot(fssh.r[0], U, 'ro')
    v_text = plt.text(-5, 0.012, f"v: f{v}")
    e_text = plt.text(-5, .01, f"E: f{U + fssh.calc_KE(v)}")

    def animate(_):
        if not fssh.step(fssh.dt_c) or f(fssh):
            return None
        U = model.get_adiabatic_energy(fssh.r[0])[fssh.lam]
        v = fssh.v[0]
        point.set_xdata(fssh.r[0])
        point.set_ydata(U)
        v_text.set_text(v)
        e_text.set_text(f"E: {U + fssh.calc_KE(v)}")
        return point, v_text, e_text

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)
    return ani


def animate_2d(fssh, fig, ax, x_linspace, y_linspace, stopping_function=f, interval=50, num_points=400, colors=["black", "red"]):
    model = fssh.model
    models.plot_2d(ax, model, x_linspace, y_linspace, colors)

    point, = ax.plot(fssh.r[0], fssh.r[1], model.get_adiabatic_energy(
        fssh.r)[fssh.lam], marker='o')

    def animate(_):
        if not fssh.step(fssh.dt_c) or f(fssh):
            return None
        point.set_data(fssh.r)
        point.set_3d_properties(
            model.get_adiabatic_energy(fssh.r)[fssh.lam])
        return point,

    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=0)

    return ani
