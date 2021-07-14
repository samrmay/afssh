import numpy as np
import random as rand
import scipy.linalg as linalg
import scipy.integrate as integrate
import math as math


def mag(v, axis=0):
    return np.sqrt(np.sum(np.square(v), axis=axis))


def angle(v1, v2):
    cos_theta = np.dot(v1, v2)/mag(v1)/mag(v2)
    cos_theta = max(min(cos_theta, 1), 0)
    return math.acos(cos_theta)


def quadratic(a, b, c):
    try:
        p2 = math.sqrt((b**2) - 4*a*c)
    except:
        p2 = 0
    return (-b + p2)/2/a, (-b - p2)/2/a


class AFSSH():
    """
    Class to carry out Augmented Fewest Switches Surface Hopping algorithm.
    Closely follows http://dx.doi.org/10.1021/acs.jctc.6b00673 and equation references are from this paper
    """

    def __init__(self, model, r0, v0, dt_c, e_tol=1e-3, coeff=None, mass=2000, t0=0, state0=0, deco=None, langevin=None, seed=None):
        """
        Instantiates AFSSH class

        Parameters
            model (model): Diabatic potential model on which trajectory will travel
            r0 (ndarray): vector describing start position of particle
            v0 (ndarray): vector describing start velocity of particle
            dt_c (int): default classical time step for algorithm
            e_tol (float): energy tolerance when trying to conserve energy. Will not attempt to conserve energy if None
            coeff (ndarray): start coefficients of particle. If None, instantiate fully in ground state
            state0 (int): start PES for particle. Should match coeff
            deco (dict): dict with keys delta_R, delta_P. If None, dont run with decoherence
            langevin (dict): dictionary with keys damp and temp. If None, no Langevin dynamics
            seed (int): Seed to use for random number generator

        Returns
            AFFSH: Class to propagate particle
        """
        self.model = model
        self.m = mass
        self.dt_c = dt_c
        self.t = t0

        # Bookkeeping variables
        self.debug = False
        self.i = 0
        self.deco = deco
        self.e_tol = e_tol
        # Track switches with dict {old_state, new_state, position, coeff, delta_v, success}
        self.switches = []

        # Initialize electronic state
        self.num_states = model.num_states
        self.lam = state0
        self.state0 = state0
        if state0 >= self.num_states:
            raise ValueError(
                "ground state must be less than total number of states")
        if coeff is None:
            self.coeff = np.zeros(self.num_states, dtype=complex)
            self.coeff[state0] = 1
        else:
            self.coeff = coeff

        # Initialize position and ensure matches model
        if not hasattr(r0, "__len__"):
            self.r = np.array([r0])
        if not hasattr(v0, "__len__"):
            self.v = np.array([v0])
            self.v0 = np.array([v0])
        self.r = r0
        self.v = v0
        self.v0 = v0

        self.dim = len(r0)
        if self.dim != model.dim:
            raise ValueError(
                "position vector dimension must match model dimension")

        # Initialize moments of position and velocity for decoherence (if applicable)
        if deco != None:
            self.deco = True
            self.delta_R = deco.get('delta_R')
            self.delta_P = deco.get('delta_P')
        else:
            self.deco = False
            self.delta_R = np.zeros(self.num_states)
            self.delta_P = np.zeros(self.num_states)
        self.torque = None
        if len(self.delta_R) != self.num_states or len(self.delta_P) != self.num_states:
            raise ValueError(
                "moment vector dimensions must match model dimension")

        # Track acceleration to save time when calculating trajectory
        self.a = np.zeros(self.dim)

        # Initialize seed if given
        self.seed = seed
        if seed != None:
            rand.seed(seed)

        # Handle Langevin if applicable
        if langevin != None:
            self.damping = langevin.get('damp')
            self.temp = langevin.get('temp')
            self.langevin = True
        else:
            self.langevin = False

        # Initialize constants
        self.HBAR = 1
        self.BM = 3.1668e-6

    def calc_traj(self, r0, v0, del_t, a0):
        """
        Propagates position, velocity using velocity Verlet algorithm (with half step velocity).
        Uses self.lam for current PES
        If langevin flag active, add damping (friction) and random motion (solvent)

        Returns:
            r: new position at time t0 + del_t
            v: new velocity at time t0 + del_t
            a: new acceleration at time t0 + del_t
        """
        half_v = v0 + .5*a0*del_t
        r = r0 + half_v*del_t
        a = -self.model.get_d_adiabatic_energy(r)[self.lam]/self.m
        if self.langevin:
            a = a - self.damping*half_v + (self.rand_force(del_t)/self.m)
        v = half_v + .5*a*del_t
        return r, v, a

    def rand_force(self, dt):
        """
        Returns random Langevin force from gaussian distribution times temperature and damping constant
        """
        sigma = np.sqrt(2*self.damping*self.m*self.BM*self.temp/dt)
        return np.random.normal(loc=0, size=self.dim, scale=sigma)

    def calc_overlap_mtx(self, r0, r1):
        """
        Calculates overlap matrix U, which measures overlap between adiabatic wave functions at r0 and r1.
        Has phase correction from http://dx.doi.org/10.1021/acs.jctc.9b00952
        **ONLY HAS PHASE CORRECTION FOR REAL HAMILTONIAN REGIME. NEED TO IMPLEMENT FOR COMPLEX**

        Parameters
            r0 (ndarray): initial particle position
            r1 (ndarray): final particle position

        Returns
            u_mtx (ndarray): overlap matrix with dimensions (num_states, num_states)
        """
        # Calculate phi(t0) and phi(t0 + tau)
        ev0 = self.model.get_wave_function(r0)
        ev1 = self.model.get_wave_function(r1)

        # Calculate u_mtx, ensuring positive diagonal (eq. 5)
        U = np.einsum("ij,ik->jk", ev0, ev1)
        # Old method with loop
        # U = np.zeros((self.num_states, self.num_states))
        # for i in range(self.num_states):
        #     for j in range(self.num_states):
        #         U[i, j] = ev0[:, i]@ev1[:, j]

        # If determinant of U is -1, flip phase of first eigenvector for r1
        # (flip first column of U)
        if int(linalg.det(U)) == -1:
            U[:, 0] *= -1

        # Enforce second condition
        def jacobi_sweep():
            converged = True

            for i in range(self.num_states):
                for j in range(i+1, self.num_states):
                    d = 3*((U[i, i]**2) + (U[j, j]**2))
                    d += 6*U[i, j]*U[j, i]
                    d += 8*(U[i, i] + U[j, j])
                    d -= 3 * \
                        (np.einsum("i,i->", U[i, :], U[:, i]) +
                         np.einsum("i,i->", U[j, :], U[:, j]))
                    if d < 0:
                        U[:, i] *= -1
                        U[:, j] *= -1
                        converged = False
            return converged

        converged = False
        while not converged:
            converged = jacobi_sweep()

        # Orthogonalize U
        U = self.orthogonalize(U)
        return U

    def calc_t_mtx(self, u_mtx, dt_c):
        """
        Calculate time derivative mtx. at dt_c/2 using eq. 29 and a proper U mtx.

        Parameters
            u_mtx (ndarray): overlap matrix for time=dt_c
            dt_c (int): classical time step

        Returns
            t_mtx (ndarray): Time derivative matrix for t=dt_c/2
        """
        return (1/dt_c)*linalg.logm(u_mtx)

    def calc_coeff(self, t0, t1, V, T, coeff0):
        """
        Calculate quantum coefficients (eq. 11)

        Parameters:
            t0 (int): initial time
            t1 (int): end time
            V (func): function that takes in single argument t and returns potential matrix at time t.
            Paper suggests linear interpolation between potential at times t0 and t1
            T (ndarray): time derivative matrix
            coeff0: intial coefficients at time t0

        Returns
            coeff (ndarray): coefficients integrated from t0 to t1
        """
        def f(t, c):
            # old (slower) method: np.sum(T*np.tile(c, (self.num_states, 1)), axis=1)
            return (V(t)*c/1j/self.HBAR) - np.einsum("j,ij->i", c, T)

        result = integrate.solve_ivp(f, (t0, t1), coeff0)
        return result.y[:, -1]

    def calc_hop_probabilities(self, coeff, t_mtx, dt_q, lam):
        """
        Calculate probability of hopping from surface lam to every other surface

        Parameters
            coeff (ndarray): wave function coefficients
            t_mtx (ndarray): Time density matrix
            dt_q (int): Quantim time step
            lam (int): current PES index

        Returns
            hop_vec (ndarray): Vector denoting hop probabilities. v[i] = probability from hopping from lam to i
        """
        c_vec = coeff/coeff[lam]
        t_vec = t_mtx[:, lam]
        result = -2*dt_q*(c_vec.real)*t_vec
        return np.maximum(result, np.zeros(len(c_vec)))

    def calc_KE(self, v):
        """
        Calculate kinetic energy given velocity. Uses class mass

        Parameters
            v (ndarray): velocity of particle

        Returns
            KE (float): Kinetic energy of particle
        """
        return .5*self.m*(mag(v)**2)

    def log_switch(self, state0, state1, r, v, c, delta_v, success=True):
        """
        Add dict object to state switches for logging purposes

        Parameters
            state0 (int): old state
            state1 (int): new state
            r (ndarray): position vector
            v (ndarray): velocity vector
            c (ndarray): function coefficients
            delta_v (float): difference between old PES and new PES at r
            success (bool): whether state switch was successful

        Returns
            None
        """
        log = {
            "old_state": state0,
            "new_state": state1,
            "position": r,
            "velocity": v,
            "coefficients": c,
            "delta_v": delta_v,
            "success": success
        }
        self.switches.append(log)

        return None

    def calc_torque(self, r):
        F = -self.model.get_d_adiabatic_energy(r)
        F[self.lam, :] *= 0
        return np.diag(F)

    def propagate_moments(self, delta_R0, delta_P0, torque0, dt_c, u_mtx, coeff, r):
        """
        unimplemented
        """
        sigma = coeff*np.conj(coeff)
        U_sq = np.sum(u_mtx**2, axis=1)

        delta_R = delta_R0 + (delta_P0/self.m) + \
            (.5/self.m)*torque0*sigma*(dt_c**2)

        torque1_ad = self.calc_torque(r)
        torque1 = U_sq*sum(torque1_ad)
        delta_P = delta_P0 + .5*(torque0 + torque1)*sigma*dt_c

        delta_R_ad = U_sq*sum(delta_R)
        delta_P_ad = U_sq*sum(delta_P)

        return delta_R_ad, delta_P_ad, torque1_ad

    def calc_deco_rate(self, delta_F, delta_R, T, pot, v):
        """
        Returns 1/tau
        """
        T_row = T[self.lam, :]
        term1 = (1/2/self.HBAR)*delta_F*(delta_R - delta_R[self.lam])
        term2 = np.tile(T_row*(-1*pot + pot[self.lam])*(
            delta_R - delta_R[self.lam]), (self.dim, 1)).T
        term2 *= np.tile(v, (self.num_states, 1))
        term2 = mag(term2, axis=1)
        term2 *= (2/self.HBAR/np.dot(v, v))
        result = term1 - term2

        return result

    def calc_reset_rate(self, delta_F, delta_R):
        """
        Returns 1/tau_reset
        """
        result = (-1/2/self.HBAR)*delta_F*(delta_R - delta_R[self.lam])
        return result

    def collapse_reset(self, deco_rate, reset_rate, coeff, delta_R, delta_P, dt_c):
        nu = rand.random()
        nu_v = np.zeros(self.num_states) + nu
        should_collapse = nu_v < deco_rate*dt_c
        should_reset = np.logical_or(should_collapse, nu_v < reset_rate*dt_c)

        for i in range(self.num_states):
            if should_collapse[i]:
                c_lam = coeff[self.lam]
                coeff[self.lam] = (c_lam/abs(c_lam)) * \
                    math.sqrt((abs(c_lam)**2) + (abs(coeff[i])**2))
                coeff[i] = 0

            if should_reset[i]:
                delta_R[i] = 0
                delta_P[i] = 0

        return coeff, delta_R, delta_P

    def orthogonalize(self, mtx):
        """
        Orthogonalize hermitian matrix using Lowdin orthogonalization scheme

        Parameters
            mtx (ndarray): Hermitian matrix to be orthogonalized

        Returns
            ortho_mtx (ndarray): orthogonalized matrix
        """
        return mtx@linalg.fractional_matrix_power((mtx.T@mtx), -.5)

    def prop_quantum(self, r0, r, u0, u1, dt_c, t0):
        # Calculate overlap matrix and time density mtx
        u_mtx = self.calc_overlap_mtx(r0, r)
        t_mid = self.calc_t_mtx(u_mtx, dt_c)

        # Determine dt_q (eq. 20, 21)
        u = self.model.get_adiabatic_energy((r + r0)/2)
        dt_q_prime = min(dt_c, .02/np.max(np.absolute(t_mid)))
        # Definition of bold V under eq. 16
        dt_q_prime = min(dt_q_prime, .02*self.HBAR /
                         np.max(np.absolute(u - np.average(u))))

        dt_q = dt_c/int(round(dt_c/dt_q_prime))

        # Carry out quantum time steps from t0 -> t0 + dt_c by dt_q
        # T is constant, V is varied) linearly from u0, u1
        n_q = int(dt_c/dt_q)
        def u_interp(t): return u0 + ((t - t0)/dt_c)*(u1-u0)
        c0 = self.coeff
        hop_attempted = False
        new_PES = self.lam

        for k in range(1, n_q + 1):
            t1 = t0 + (k-1)*dt_q
            t2 = t0 + k*dt_q
            c = self.calc_coeff(t1, t2, u_interp, t_mid, c0)

            if not hop_attempted:
                hop_vector = self.calc_hop_probabilities(
                    c, t_mid, dt_q, self.lam)
                delta = rand.random()
                # Try to hop to any state
                for i in range(self.num_states):
                    if delta < hop_vector[i] and i != self.lam:
                        hop_attempted = True
                        new_PES = i
                        break
            c0 = c

        return c, new_PES, hop_attempted, u_mtx, t_mid

    def step(self, dt_c, classical=False):
        """
        Advances algorithm one step by time dt_c

        Parameters
            dt_c (int): classical time step by which to advance algorithm

        Returns
            success (bool): Whether energy was conserved. If False, run step again with smaller dt_c
        """
        # Save initial values for later steps
        t0 = self.t
        r0 = self.r
        v0 = self.v
        a0 = self.a
        c0 = self.coeff

        # Nuclear classical evolution using classical time step dt_c
        r, v, a = self.calc_traj(r0, v0, dt_c, a0)
        u0 = self.model.get_adiabatic_energy(r0)
        u1 = self.model.get_adiabatic_energy(r)

        # Propagate quantum effects (if applicable)
        if not classical:
            result = self.prop_quantum(r0, r, u0, u1, dt_c, t0)
            c, new_PES, hop_attempted, u_mtx, t_mid = result
        else:
            hop_attempted = False
            new_PES = self.lam
            u_mtx = None
            t_mid = None
            c = self.coeff

        # Check energy conservation (if applicable)
        if self.e_tol != None:
            energy0 = u0[self.lam] + self.calc_KE(v0)
            energy1 = u1[self.lam] + self.calc_KE(v)
            if abs(energy1 - energy0) > self.e_tol:
                if hop_attempted:
                    # If above energy tolerance, use eq. 22 to update v
                    F0 = -self.model.get_d_adiabatic_energy(r0)[self.lam]
                    F1 = -self.model.get_d_adiabatic_energy(r)[new_PES]
                    v = v0 + (1/2/self.m)*dt_c*F0 + F1

                    # Check if this conserves energy. If so, no need to adjust velocity
                    energy_new = u1[new_PES] + self.calc_KE(v)
                    if abs(energy_new - energy0) < self.e_tol:
                        self.r = r
                        self.v = v
                        self.a = a
                        self.coeff = c
                        self.log_switch(self.lam, new_PES, r, v, c, 0)
                        self.lam = new_PES
                        self.delta_R = 0
                        self.delta_P = 0
                        self.t += dt_c
                        return True
                # If energy not conserved using force of hopped surface (or no hop attempted),
                # retry algorithm with smaller step size
                return False

        # Adjust velocity if necessary
        if hop_attempted:
            diff = u1[new_PES] - u1[self.lam]
            ev = self.model.get_wave_function(r)
            d_ev = self.model.get_d_wave_functions(r)
            dlj = ev[:, self.lam]@d_ev[:, new_PES]

            if self.calc_KE(v) <= diff:
                # Frustrated hop
                F = self.model.get_d_adiabatic_energy(r)
                check1 = np.dot(F[self.lam], dlj) * \
                    np.dot(F[new_PES], dlj) < 0
                check2 = np.dot(F[new_PES], dlj)*np.dot(r, dlj) < 0
                if check1 and check2:
                    correction = -np.sum(v0*dlj)/np.sum(np.square(dlj))
                    v = v0 + correction
                self.log_switch(self.lam, self.lam, r, v, c, diff, False)
            else:
                # Carry out correction and set moments to 0
                # If dlj == 0, denotes trivial crossing (WHY???).
                # At trivial crossing dlj -> infinity, correction -> 0, therefore do nothing
                if dlj != 0:
                    c_a = np.sum(np.square(dlj))
                    c_b = np.sum(2*dlj*v0)
                    c_c = (2/self.m)*diff
                    factors = quadratic(c_a, c_b, c_c)
                    corrections = factors*dlj

                    # Choose velocity correction that minimizes angle between v0 and v
                    v1 = v0 + corrections[0]
                    v2 = v0 + corrections[1]
                    v = min(v1, v2, key=lambda v: angle(v, v0))

                self.delta_R = 0
                self.delta_P = 0
                self.log_switch(self.lam, new_PES, r, v, c, diff)
                self.lam = new_PES

        # Decoherence calculations (if applicable) **Still not working**
        if self.deco:
            # Evolve moments
            torque0 = self.calc_torque(r0)
            del_R0 = self.delta_R
            del_P0 = self.delta_P

            # Check for collapse/reset
            result = self.propagate_moments(
                del_R0, del_P0, torque0, dt_c, u_mtx, c0, r)
            del_R, del_P, torque1 = result
            deco_rate = self.calc_deco_rate(torque1, del_R, t_mid, u1, v)
            reset_rate = self.calc_reset_rate(torque1, del_R)

            result = self.collapse_reset(
                deco_rate, reset_rate, c, del_R, del_P, dt_c)
            c, self.delta_P, self.delta_R = result

        # Update parameters
        self.t += dt_c
        self.v = v
        self.r = r
        self.a = a
        self.coeff = c
        return True

    def run(self, max_iter, stopping_fcn, debug=False, callback=lambda _: _):
        for i in range(max_iter):
            self.i = i
            if not self.step(self.dt_c):
                # Need to rerun step with smaller step size
                self.step(int(self.dt_c/2))
            callback(self)
            if (debug and i % 100 == 0):
                print(self.r, self.v, self.calc_KE(self.v), self.lam, self.t)

            if stopping_fcn(self):
                return
