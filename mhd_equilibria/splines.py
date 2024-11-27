import jax
from jax import numpy as jnp

from functools import partial

# TODO: Splines right now only work on [0, 1]

def indicator(x, i, T, p, n, m):
    return jnp.where(jnp.logical_and(T[i] <= x, x < T[i+1]), 
        1.0, 
        0.0
        )
    
def indicator_clamped(x, i, T, p, n, m):
    return jnp.where(jnp.logical_and(T[i] <= x, x < T[i+1]), 
        1.0, 
        jnp.where(jnp.logical_and(x == T[i+1], i == n-1),
            1.0,
            0.0)
        )

def safe_divide(numerator, denominator):
        return jnp.where(denominator == 0, 
            jnp.where(numerator == 0, 
                1.0, 
                0.0),
            numerator / denominator)

#TODO: the value of eps maybe should be a parameter
@partial(jax.jit, static_argnums=(3,6))
def spline(x, i, T, p, n, m, type):
    eps = 1e-16
    if p == 0:
        if type == 'clamped':
            return indicator_clamped(x, i, T, p, n, m)
        else:
            return indicator(x, i, T, p, n, m)
    else:
        w_1 = safe_divide(x - T[i] + eps,T[i + p] - T[i] + eps)
        w_2 = safe_divide(T[i + p + 1] - x + eps, T[i + p + 1] - T[i + 1] + eps)
        return w_1 * spline(x, i, T, p - 1, n, m, type) \
            + w_2 * spline(x, i + 1, T, p - 1, n, m, type)
            
def knot_vector(n, p, type='open'):
    if type == 'periodic':
        T = jnp.concatenate([jnp.linspace(0, 1, n-p+1)[-(p+1):-1] - 1, 
            jnp.linspace(0, 1, n-p+1), 
            jnp.linspace(0, 1, n-p+1)[1:(p+2)] + 1])
    elif type == 'clamped':
        T = jnp.concatenate([jnp.zeros(p),
            jnp.linspace(0, 1, n-p+1),
            jnp.ones(p)])
    else:
        print("Not implemented")
    return T

#TODO: Check the n -> n+p stuff in the periodic spline case
def get_spline(n, p, type='clamped'):
    T = knot_vector(n, p, type)
    m = n - p + 1
    if type == 'periodic':
        n = n + p
        def _spline(x, i):
            i = i + p
            return jax.lax.cond(i > n - 2*p,
                lambda _: spline(x, i, T, p, n, m, 'periodic') \
                        + spline(x, i - n + p, T, p, n, m, 'periodic'),
                lambda _: spline(x, i, T, p, n, m, 'periodic'),
                operand=None)
    elif type == 'clamped':
        def _spline(x, i):
            return spline(x, i, T, p, n, m, 'clamped')
    return _spline



