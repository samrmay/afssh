from matplotlib.pyplot import xcorr
import numpy as np
import math as math


class Diabatic_Model:
    def __init__(self, num_states, dim=1):
        self.num_states = num_states
        self.dim = dim

    def get_adiabatic(self, x, correction=0):
        v, ev = np.linalg.eigh(self.V(x) + correction)
        d = {}
        for i in range(len(v)):
            d[v[i]] = ev[:, i]

        v_sorted = np.sort(v)
        ev_sorted = np.zeros((self.num_states, self.num_states))
        for i in range(len(v_sorted)):
            ev_sorted[:, i] = d[v[i]]

        return v_sorted, ev_sorted

    def get_adiabatic_energy(self, x, correction=0):
        return self.get_adiabatic(x, correction)[0]

    def get_wave_function(self, x, correction=0):
        return self.get_adiabatic(x, correction)[1]

    def get_d_adiabatic_energy(self, x, step=0.00001):
        grad_v = np.zeros((self.num_states, self.dim))
        for i in range(self.dim):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] -= step
            x2[i] += step

            V1 = self.get_adiabatic_energy(x1)
            V2 = self.get_adiabatic_energy(x2)
            grad_v[:, i] = (V2 - V1)/(2*step)

        return grad_v

    def get_d_wave_functions(self, x, step=0.00001):
        if not hasattr(x, "__len__"):
            x = np.asarray([x])

        grad_phi = np.zeros((self.num_states, self.num_states, self.dim))
        for i in range(self.dim):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] -= step
            x2[i] += step

            phi1 = self.get_wave_function(x1)
            phi2 = self.get_wave_function(x2)
            grad_phi[:, :, i] = (phi2 - phi1)/(2*step)

        return grad_phi


class Simple_Avoided_Crossing(Diabatic_Model):
    def __init__(self, a=0.01, b=1.6, c=0.005, d=1.0, discont=0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.discont = discont
        self.num_states = 2

        super().__init__(self.num_states)

    def V(self, x):
        if x > self.discont:
            V11 = self.A*(1-(math.exp(-self.B*x)))
        else:
            V11 = -self.A*(1-(math.exp(self.B*x)))

        V22 = -V11
        V12 = V21 = self.C*math.exp(-self.D*(x**2))

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        if x > self.discont:
            dV11 = self.A*self.B*math.exp(-self.B*x)
        else:
            dV11 = -self.A*self.B*math.exp(self.B*x)

        dV22 = -dV11
        dV12 = dV21 = -2*self.C*self.D*x*math.exp(-self.D*(x**2))

        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Double_Avoided_Crossing(Diabatic_Model):
    def __init__(self, a=.1, b=.28, e0=.05, c=.015, d=.06):
        self.A = a
        self.B = b
        self.E0 = e0
        self.C = c
        self.D = d
        self.num_states = 2

        super().__init__(self.num_states)

    def V(self, x):
        V11 = 0.0
        V22 = (-self.A*math.exp(-self.B*(x**2))) + self.E0
        V12 = V21 = self.C*math.exp(-self.D*(x**2))

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        dV11 = 0
        dV22 = 2*self.A*self.B*x*math.exp(-self.B*(x**2))
        dV12 = dV21 = -2*self.C*self.D*x*math.exp(-self.D*(x**2))

        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Extended_Coupling_With_Reflection(Diabatic_Model):
    def __init__(self, a=6e-4, b=.1, c=.9, discont=0):
        self.A = a
        self.B = b
        self.C = c
        self.discont = discont
        self.num_states = 2

        super().__init__(self.num_states)

    def V(self, x):
        V11 = self.A
        V22 = -self.A
        if x > self.discont:
            V12 = V21 = self.B*(2-math.exp(-self.C*x))
        else:
            V12 = V21 = self.B*math.exp(self.C*x)

        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        dV11 = 0
        dV22 = 0
        if x > self.discont:
            dV12 = dV21 = self.B*self.C*math.exp(-self.C*x)
        else:
            dV12 = dV21 = self.C*self.B*math.exp(self.C*x)

        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Coupled_Osc1d(Diabatic_Model):
    """
    From Landry and Subotnik 2011 https://doi.org/10.1063/1.3663870
    """

    def __init__(self, omega=3.5e-4, er=2.39e-2, e0=1.5e-2, coup=1.49e-5, mass=2000):
        self.omega = omega
        self.Er = er
        self.E0 = e0
        self.coup = coup
        self.mass = mass
        self.M = np.sqrt(self.Er*self.mass*(self.omega**2)/2)
        self.num_states = 2

        super().__init__(self.num_states)

    def V(self, x):
        if hasattr(x, "__len__"):
            x = x[0]

        V11 = (.5*self.mass*(self.omega**2)*(x**2)) + self.M*x
        V22 = (.5*self.mass*(self.omega**2)*(x**2)) - self.M*x - self.E0
        V12 = V21 = self.coup
        return np.asarray([[V11, V12], [V21, V22]])

    def dV(self, x):
        if hasattr(x, "__len__"):
            x = x[0]

        dV11 = self.mass*(self.omega**2)*x + self.M
        dV22 = self.mass*(self.omega**2)*x - self.M
        dV12 = dV21 = 0
        return np.asarray([[dV11, dV12], [dV21, dV22]])


class Coupled_Osc2d(Diabatic_Model):
    def __init__(self, omega=3.5e-4, er=2.39e-2, e0=1.5e-2, coup=1.49e-5, mass=2000):
        self.omega = omega
        self.Er = er
        self.E0 = e0
        self.coup = coup
        self.mass = mass
        self.M = np.sqrt(self.Er*self.mass*(self.omega**2)/2)
        self.num_states = 2

        super().__init__(self.num_states, 2)

    def V(self, x):
        V11 = (.5*self.mass*(self.omega**2)*(sum(x**2))) + self.M*(sum(x))
        V22 = (.5*self.mass*(self.omega**2)*(sum(x**2))) - \
            self.M*(sum(x)) - self.E0
        V12 = V21 = self.coup
        return np.asarray([[V11, V12], [V21, V22]])


class NState_Spin_Boson(Diabatic_Model):
    def __init__(self, omega=3.5e-4, er=2.39e-2, e0=1.5e-2, coup=1.49e-5, mass=2000, l_states=1, r_states=1):
        self.omega = omega
        self.Er = er
        self.E0 = e0
        self.coup = coup
        self.mass = mass
        self.M = np.sqrt(self.Er*self.mass*(self.omega**2)/2)

        self.l_states = l_states
        self.r_states = r_states
        self.num_states = l_states + r_states
        self.d = e0

        super().__init__(self.num_states)

    def V(self, x):
        result = np.zeros((self.num_states, self.num_states))
        result += self.coup

        osc = .5*self.mass*(self.omega**2)*(x**2)
        for i in range(self.l_states):
            result[i, i] = osc + (self.M*x) - self.E0 + (i*self.d)
        for j in range(self.r_states):
            i = self.l_states + j
            result[i, i] = osc - (self.M*x) + (j*self.d)

        return result


def plot_1d(ax, model, x_linspace):
    potentials = np.zeros((len(x_linspace), model.num_states))
    for i in range(len(x_linspace)):
        x = x_linspace[i]
        potentials[i] = model.get_adiabatic_energy(x)

    for i in range(model.num_states):
        ax.plot(x_linspace, potentials[:, i])


def plot_1d_coupling(ax, model, x_linspace, coupling_scaling_factor):
    l = len(x_linspace)
    coupling = np.zeros((model.num_states, model.num_states, l))
    for i in range(l):
        x = x_linspace[i]
        phi = model.get_wave_function(x)
        grad_phi = model.get_d_wave_functions(x)
        for j in range(model.num_states):
            for k in range(model.num_states):
                coupling[j, k, i] = phi[:, j]@grad_phi[:, k]

    for i in range(model.num_states):
        for j in range(model.num_states):
            if i != j:
                ax.plot(x_linspace, coupling[i, j, :]*coupling_scaling_factor)


def plot_diabats_1d(ax, model, x_linspace):
    diabats = np.zeros((len(x_linspace), model.num_states))
    for i in range(len(x_linspace)):
        V = model.V(x_linspace[i])
        for j in range(model.num_states):
            diabats[i, j] = V[j, j]

    for i in range(model.num_states):
        ax.plot(x_linspace, diabats[:, i])


def plot_2d(ax, model, x_linspace, y_linspace, colors):
    l = len(x_linspace)
    if l != len(y_linspace):
        raise ValueError("Size of axes must be equal")

    if len(colors) != model.num_states:
        raise ValueError("Number of colors must equal number of states")

    xx, yy = np.meshgrid(x_linspace, y_linspace)
    V = np.zeros((l, l, model.num_states))
    for i in range(l):
        for j in range(l):
            V[j, i] = model.get_adiabatic_energy(
                np.asarray([xx[j, i], yy[j, i]]))

    for i in range(model.num_states):
        ax.plot_wireframe(xx, yy, V[:, :, i], color=colors[i])
