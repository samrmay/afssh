import numpy as np
import random as rand


class AFSSH():
    def __init__(self, model, r0, v0, dt_c, dt_q, coeff=None, mass=2000, t0=0, state0=0, seed=None):
        self.model = model
        self.m = mass
        self.e_state = state0
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
        # Initialize seed if given
        if seed != None:
            rand.seed(seed)

        # Initialize constants
        self.HBAR = 1
