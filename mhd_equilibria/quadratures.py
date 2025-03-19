import jax.numpy as jnp
import jax
from jax import vmap
from mhd_equilibria.forms import *
import numpy as np

__all__ = [
    "quadrature_grid",
    "get_quadrature_periodic",
    "get_quadrature_spectral",
    "get_quadrature_composite",
]

def quadrature_grid(x, y, z):
    x_x, w_x = x
    x_y, w_y = y
    x_z, w_z = z
    
    x_s = [x_x, x_y, x_z]
    w_s = [w_x, w_y, w_z]
    d = 3
    N = w_x.size * w_y.size * w_z.size
    
    x_hat = jnp.array(jnp.meshgrid(*x_s)) # shape d, n1, n2, n3, ...
    x_hat = x_hat.transpose(*range(1, d+1), 0).reshape(N, d)
    w_q = jnp.array(jnp.meshgrid(*w_s)).transpose(*range(1, d+1), 0).reshape(N, d)
    w_q = jnp.prod(w_q, 1)
        
    return x_hat, w_q
    
def get_quadrature_periodic(n):
    def _get_quadrature(a, b):
        h = (b - a) / n
        _x = jnp.linspace(a, b - h, n)
        w_q = h * jnp.ones(n)
        return _x, w_q
    return _get_quadrature

def get_quadrature_spectral(n):
    def _get_quadrature(a, b):
        q = np.polynomial.legendre.leggauss(n)
        # Weights
        w_q = q[1] * (b - a) / 2
        # Points
        _x = (q[0] + 1) / 2 * (b - a) + a
        return _x, w_q
    return _get_quadrature

def get_quadrature_composite(T, n):
    # T is the knot vector without multiplicty
    # using the low order (Gauss) integration here
    def _get_quadrature(a, b):
        #  n is the number of points and weights
        q = np.polynomial.legendre.leggauss(n)
        # Transform to interval we want
        w_q = q[1][1::2] * (b - a) / 2
        x_q = (q[0][1::2] + 1) / 2 * (b - a) + a
        return x_q, w_q
    a_s = T[:-1]
    b_s = T[1:]
    x_q, w_q = vmap(_get_quadrature)(a_s, b_s)
    x_q = x_q.reshape(len(a_s)*(n//2))
    w_q = w_q.reshape(len(a_s)*(n//2))
    return x_q, w_q











