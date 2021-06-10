import numpy as np
import random as rand
import scipy.linalg as linalg
import scipy.integrate as integrate


class AFSSH():
    """
    Class to carry out Augmented Fewest Switches Surface Hopping algorithm. 
    Closely follows http://dx.doi.org/10.1021/acs.jctc.6b00673 and equation references are from this paper
    """

    def __init__(self, model, r0, v0, dt_c, e_tol=1e-6, coeff=None, mass=2000, t0=0, state0=0, seed=None):
        self.model = model
        self.m = mass
        self.lam = state0
        self.dt_c = dt_c
        self.e_tol = e_tol
        self.t = t0

        # Bookkeeping variables
        self.debug = False
        self.i = 0
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

    def calc_traj(self, r0, v0, del_t, m, a0=None):
        """
        Propagates position, velocity using velocity Verlet algorithm (with half step velocity).
        If a == None, calculate a0 from model
        Uses self.lam for current PES

        Returns:
            r: new position at time t0 + del_t
            v: new velocity at time t0 + del_t
            a: new acceleration at time t0 + del_t
        """
        if a0 == None:
            a0 = self.model.get_d_adiabatic_energy(r0)[self.lam]/m

        half_v = v0 + .5*a0*del_t
        r = r0 + (half_v)*del_t
        a = self.model.get_d_adiabatic_energy(r)[self.lam]/m
        v = half_v + .5*a*del_t

        return r, v, a

    def calc_overlap_mtx(self, r0, r1, correction=1e-6):
        # Calculate phi(t0) and phi(t0 + tau)
        ev0 = self.model.get_wave_functions(r0)
        ev1 = self.model.get_wave_functions(r1)

        # Calculate u_mtx, ensuring positive diagonal (eq. 5)
        def calc(ev0, ev1):
            u_mtx = np.zeros((self.num_states, self.num_states))
            for i in range(self.num_states):
                for j in range(self.num_states):
                    u_mtx[i, j] = ev0[:, i]@ev1[:, j]
            return u_mtx
        u_mtx = calc(ev0, ev1)

        if np.any(np.less(np.diag(u_mtx), np.zeros(self.num_states))):
            u_mtx = calc(-1*ev0, ev1)

        # Check for trivial crossing edge case and correct if necessary (eq. 30-33)
        if np.any(np.equal(u_mtx, np.zeros(self.num_states))):
            adj_ev0 = self.model.get_wave_functions(r0, correction=correction)
            adj_ev1 = self.model.get_wave_functions(r1, correction=correction)
            print("0 IN U_MTX. UNHANDLED CASE")
            # ADD CORRECTION HERE

        # Orthogonalize U (eq. 34) (Check if exponent is element wise)
        u_mtx = u_mtx@((np.transpose(u_mtx)@u_mtx)**(-1/2))

        return u_mtx

    def calc_t_mtx(self, u_mtx, d_tc):
        """
        Calculate time density mtx. at d_tc/2 using eq. 29 and a proper U mtx.
        """
        return (1/d_tc)*linalg.logm(u_mtx)

    def calc_coeff(self, t0, t1, V, T, coeff0):
        """
        Calculate quantum coefficients (eq. 11)

        Parameters:
            t0 (int): initial time
            t1 (int): end time
            V (func): function that takes in single argument t and returns potential matrix at time t. 
            Paper suggests linear interpolation between potential at times t0 and t1
            T (iterable): time derivative matrix
            coeff0: intial coefficients at time t0
        """
        def f(t, c):
            energy = V(t)
            summation = np.sum(T*np.tile(c, (self.num_states, 1)), axis=1)
            return (energy*c - summation)/(1j*self.HBAR)

        result = integrate.solve_ivp(f, (t0, t1), self.coeff)
        return result.y[:, -1]

    def step(self):
        dt_c = self.dt_c
        t0 = self.t

        # Nuclear classical evolution using classical time step dt_c
        r0 = self.r
        v0 = self.v
        a0 = self.a
        r, v, a = self.calc_traj(r0, v0, dt_c, self.m, a0=a0)

        # Calculate overlap matrix and time density mtx
        u_mtx = self.calc_overlap_mtx(dt_c)
        t_mid = self.calc_t_mtx(u_mtx, dt_c)

        # Determine dt_q (eq. 20, 21)
        u = self.model.get_adiabatic_energy((r + r0)/2)
        dt_q_prime = min(dt_c, .02/np.max(np.absolute(t_mid)))
        # Check that V here is correct (using V at x0)
        dt_q_prime = min(dt_q_prime, .02*self.HBAR /
                         np.max(np.absolute(u - np.average(u))))

        dt_q = dt_c/int(round(dt_c/dt_q_prime))

        # Carry out quantum time steps from t0 -> t0 + dt_c by dt_q
        # T is constant, V is varied linearly from u0, u1
        n_q = int(dt_c/dt_q)
        u0 = self.model.get_adiabatic_energy(r0)
        u1 = self.model.get_adiabatic_energy(r)
        def u_interp(t): return u0 + ((t - t0)/(dt_c))*(u1-u0)
        c0 = self.coeff

        for k in range(1, n_q + 1):
            t1 = t0 + (k-1)*dt_q
            t2 = t0 + k*dt_q
            c = self.calc_coeff(t1, t2, u_interp, t_mid, c0)
