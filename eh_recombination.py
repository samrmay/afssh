import numpy as np
import scipy.linalg as linalg
import random as random


def constant_gamma(v):
    return lambda _: v


class Eh_Recombiner():
    def __init__(self, fssh, gamma, rho_eq, err_check=1e-3):
        self.fssh = fssh
        self.gamma = gamma
        self.rho_eq = rho_eq
        self.err_check = err_check

    def to_diab(self, R, coeff):
        U = self.fssh.model.get_wave_function(R)
        return np.linalg.inv(U)@coeff

    def build_density(self, coeff):
        return np.outer(coeff, coeff.conjugate())

    def adj_d_mtx(self, d_mtx, dt_q, R):
        d_d_mtx = -self.gamma(R)*(d_mtx - self.rho_eq)*dt_q
        return d_mtx + d_d_mtx

    def should_relax(self, R):
        rn = random.random()
        if rn < self.gamma(R):
            return True

        return False

    def collapse_density(self):
        return self.rho_eq

    def get_diab_state_vector(self, d_mtx):
        v, ev = np.linalg.eigh(d_mtx)
        max_v = np.max(v)
        max_i = np.argmax(v)

        if abs(1 - max_v) > self.err_check:
            raise ValueError("Density matrix inpure (should never happen)")

        return ev[max_i]
