import numpy as np
import scipy.special as spec
import math as math

KB = 3.1668e-6
PI = math.pi


def v_cumulative(v, T, m):
    return .5*np.sqrt((2*KB*T*PI)/m)*np.sqrt((KB*T*PI)/(2*m)) * \
        spec.erf(np.sqrt(m/(2*KB*T))*v)


def v_cumulative_inv(nu, T, m):
    return spec.erfinv((2*nu)/(np.sqrt((2*KB*T*PI)/m)*np.sqrt((KB*T*PI)/(2*m))))*(1/np.sqrt(m/(2*KB*T)))


def v_sample(T, m, shape):
    # Choose large v as max because cumulative function is not normalized
    nu = np.random.rand(shape)*v_cumulative(1e20, T, m)
    return v_cumulative_inv(nu, T, m)
