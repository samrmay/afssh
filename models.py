import numpy as np


class Diabatic_Model:
    def __init__(self, num_states, dim=1):
        self.num_states = num_states
        self.dim = dim

    def get_adiabatic(self, x):
        v, ev = np.linalg.eig(self.V(x))
        d = {}
        for i in range(len(v)):
            d[v[i]] = ev[:, i]

        v_sorted = np.sort(v)
        ev_sorted = np.zeros((self.num_states, self.num_states))
        for i in range(len(v_sorted)):
            ev_sorted[:, i] = d[v[i]]

        return v_sorted, ev_sorted

    def get_adiabatic_energy(self, x):
        return self.get_adiabatic(x)[0]

    def get_wave_function(self, x):
        return self.get_adiabatic(x)[1]

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
