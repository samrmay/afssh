import numpy as np
import math as math


class Diabatic_Model:
    def __init__(self, num_states, dim=1):
        self.num_states = num_states
        self.dim = dim

    def get_adiabatic(self, x, correction=0):
        v, ev = np.linalg.eig(self.V(x) + correction)
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
    def __init__(self, A=0.01, B=1.6, C=0.005, D=1.0, discont=0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
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
    def __init__(self, A=.1, B=.28, E0=.05, C=.015, D=.06):
        self.A = A
        self.B = B
        self.E0 = E0
        self.C = C
        self.D = D
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
    def __init__(self, A=6e-4, B=.1, C=.9, discont=0):
        self.A = A
        self.B = B
        self.C = C
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
