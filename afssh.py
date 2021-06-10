import numpy as np
import random as rand


class AFSSH():
    def __init__(self, model, r0, v0, dt_c, dt_q, coeff=None, mass=2000, t0=0, state0=0, seed=None):
        self.model = model
        self.m = mass
        self.lam = state0
        self.dt_c = dt_c
        self.dt_q = dt_q

        # Bookkeeping variables
        self.debug = False
        self.i = 0
        self.t = t0
        # Track switches with dict {old_state, new_state, position, delta_v, nacv, success}
        self.switches = []

        # Initialize electronic state
        self.num_states = model.num_states
        if state0 >= self.num_states:
            raise ValueError(
                "ground state must be less than total number of states")
        if coeff == None:
            self.coeff = np.zeros(self.num_states, dtype=complex)
            self.coeff[state0] = 1
        else:
            self.coeff = coeff

        # Initialize position and ensure matches model
        self.dim = len(r0) if hasattr(r0, "__len__") else 1
        if self.dim != model.dim:
            raise ValueError(
                "position vector dimension must match model dimension")

        if self.dim == 1:
            self.r = np.array([r0])
            self.v = np.array([v0])
        else:
            self.r = r0
            self.v = v0

        # Track acceleration to save time when calculating trajectory
        self.a = None

        # Initialize seed if given
        if seed != None:
            rand.seed(seed)

        # Initialize constants
        self.HBAR = 1

    def calc_traj(self, r0, v0, del_t, m):
        """
        Propagates position, velocity using velocity Verlet algorithm (with half step velocity).
        Uses self.a as the inital a0. If None, calculate at t0.
        Uses self.lam for current PES

        Returns:
            r: new position at time t0 + del_t
            v: new velocity at time t0 + del_t
            a: new acceleration at time t0 + del_t
        """
        if self.a == None:
            a0 = self.model.get_d_adiabatic_energy(r0)[self.lam]/m
        else:
            a0 = self.a

        half_v = v0 + .5*a0*del_t
        r = r0 + (half_v)*del_t
        a = self.model.get_d_adiabatic_energy(r)[self.lam]/m
        v = half_v + .5*a*del_t

        return r, v, a
