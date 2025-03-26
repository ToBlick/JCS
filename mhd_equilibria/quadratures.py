import jax.numpy as jnp
import jax
from jax import vmap
from mhd_equilibria.forms import *
import numpy as np
import quadax

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
        q = quadax.GaussKronrodRule(n,2)
        # Weights
        w_q = q._wh * (b - a) / 2
        # Points
        _x = (q._xh + 1) / 2 * (b - a) + a
        return _x, w_q
    return _get_quadrature

def get_quadrature_composite(T, n):
    # T is the knot vector without multiplicty
    # using the low order (Gauss) integration here
    def _get_quadrature(a, b):
        #  n is the number of points and weights
        q = quadax.GaussKronrodRule(n,2)
        # Transform to interval we want
        w_q = q._wh[1::2] * (b - a) / 2
        x_q = (q._xh[1::2] + 1) / 2 * (b - a) + a
        return x_q, w_q
    a_s = T[:-1]
    b_s = T[1:]
    x_q, w_q = vmap(_get_quadrature)(a_s, b_s)
    x_q = x_q.reshape(len(a_s)*(n//2))
    w_q = w_q.reshape(len(a_s)*(n//2))
    return x_q, w_q


# Testing a function to rederive points and weights. I will write in explicitly what the Legendre polynomials are 
# and then use root finder to find zeros

def quadrature(n):

    if n==1:
        points = jnp.array([0])
        weights = jnp.array([2])
    elif n==2:
        coeffs = jnp.array([3/2, 0, -1/2])
        points = jnp.roots(coeffs)
        weights = jnp.array([1,1])
    elif n==3:
        coeffs = jnp.array([5/2, 0, -3/2, 0])
        points = jnp.roots(coeffs)
        # Order is 0, 0.77, -0.77
        weights = jnp.array([8/9, 5/9, 5/9])
    elif n==4:
        coeffs = jnp.array([35/8, 0, -30/8, 0,3/8])
        points = jnp.roots(coeffs)
        # Order is 0.86, -0.86, -0.34, 0.34
        a = (jnp.sqrt(30) + 18 )/36
        b =(-jnp.sqrt(30) + 18 )/36
        weights = jnp.array([b, b, a,a])
    elif n==5:
        coeffs = jnp.array([63/8, 0, -70/8, 0,15/8,0])
        points = jnp.roots(coeffs)
        # Order is 0, -0.91, -0.54, 0.91, 0.54
        a = 128/225
        b = (322 + 13*jnp.sqrt(70))/900
        c = (322 - 13*jnp.sqrt(70))/900
        weights = jnp.array([a,c,b,c,b])
    elif n==6:
        coeffs = jnp.array([231/16, 0, -315/16, 0, 105/16, 0 ,-5/16])
        points = jnp.roots(coeffs)
        # Order is 0.93, 0.66, -0.93, -0.66, -0.24, 0.24
        d_coeffs = jnp.polyder(coeffs)
        w = jnp.polyval(d_coeffs, points)
        weights = 2/((1-points**2)*(w**2))

    elif n==7:
        coeffs = jnp.array([429/16, 0, -693/16, 0, 315/16, 0 ,-35/16,0])
        points = jnp.roots(coeffs)
        d_coeffs = jnp.polyder(coeffs)
        w = jnp.polyval(d_coeffs,points)
        weights = 2/((1-points**2)*(w**2))

    elif n==8:
        coeffs = jnp.array([6435/128, 0, -12012/16, 0, 6930/16, 0 ,-1260/16,0, 35/128])
        points = jnp.roots(coeffs)
        d_coeffs = jnp.polyder(coeffs)
        w = jnp.polyval(d_coeffs,points)
        weights = 2/((1-points**2)*(w**2))

    elif n==9:
        coeffs = jnp.array([12155/128, 0, -25740/128, 0, 18018/128, 0 ,-4620/128,0, 315/128,0])
        points = jnp.roots(coeffs)
        d_coeffs = jnp.polyder(coeffs)
        w = jnp.polyval(d_coeffs,points)
        weights = 2/((1-points**2)*(w**2))

    # n=10
    else:
        coeffs = jnp.array([46189/256, 0, -109395/256, 0, 90090/256, 0 ,-30030/256,0, 3465/256,0,-63/256])
        points = jnp.roots(coeffs)
        d_coeffs = jnp.polyder(coeffs)
        w = jnp.polyval(d_coeffs,points)
        weights = 2/((1-points**2)*(w**2))

    return points, weights






