import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, grad
from functools import partial

### ! TODO:
# There is some bug going on with Legendre bases under autodiff

# 1D trig basis function ψ number i at point x in [0, L]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def _trig_fn_x(x, k, a, b):
    L = b - a
    k_half = (k+1)//2
    x = 2 * jnp.pi * k_half * (x - a) / L
    # k = 0:    k_half = 0 -> 1
    # k = 1:    k_half = 1 -> sin(2*pi*x/L)
    # k = 2:    k_half = 1 -> cos(2*pi*x/L)
    # k = 3:    k_half = 2 -> sin(4*pi*x/L)
    # k = 4:    k_half = 2 -> cos(4*pi*x/L)
    # k = 5:    k_half = 3 -> sin(6*pi*x/L)
    _val = jax.lax.cond(k % 2 == 0, jnp.cos, jnp.sin, x)
    r1 = lambda x: jnp.sqrt(1/L)
    r2 = lambda x: jnp.sqrt(2/L)
    _val2 = jax.lax.cond(k == 0, r1, r2, x)
    return _val * _val2

def get_trig_fn_x(K, a, b):
    return lambda x, k: _trig_fn_x(x, k, a, b)

def _binom(n, m):
  return jnp.exp(gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1))

def _get_legendre_coeff(k, n):
    return jnp.round(_binom(n, k) * _binom(n + k, k))

def __get_legendre_coeffs(n, N):
    _k = jnp.arange(n + 1, dtype=jnp.int32)
    non_zero_coeffs = vmap(_get_legendre_coeff, (0, None))(_k, n)
    return jnp.concatenate([non_zero_coeffs, jnp.zeros(N - n)])

def _get_legendre_coeffs(N):
    _n = jnp.arange(N + 1, dtype=jnp.int32)
    return jnp.array([__get_legendre_coeffs(n, N) for n in _n])
    # return vmap(_get_legendre_coeffs, (0, None), axis_size=N+1)(_n, N)

# 1D legendre basis function ψ number i at point x in [0, L]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def get_legendre_fn_x(K, a, b):
    coeffs = _get_legendre_coeffs(K)
    def _legendre_fn_x(x, k, coeffs):
        _n = jnp.arange(len(coeffs[k]), dtype=jnp.int32)
        _x = (-(x - a) / (b-a) )**_n
        return (-1)**k * jnp.sqrt((2*k + 1)/(b-a)) * jnp.dot(coeffs[k, :], _x) 
    return lambda x, i: _legendre_fn_x(x, i, coeffs)

def get_polynomial_basis_fn(coeffs, a, b):
    def _basis_fn(x, k):
        _n = jnp.arange(len(coeffs[k]), dtype=jnp.int32)
        _x = ( -(x-a)/(b-a) )**_n
        return jnp.dot(coeffs[k], _x)
    return _basis_fn

# converts a linear index i to a cartesian index (i,j)
def _lin_to_cart(i, shape):
    return jnp.unravel_index(i, shape)

def get_basis_fn(bases, shape):
    # TODO: vmap?
    def basis_fn(x, k):
        d = len(x)
        _k = jnp.array(_lin_to_cart(k, shape), dtype=jnp.int32)
        # return jnp.prod( vmap(_basis_fn, (None, 0, 0))(x, _d, _k) ) 
        return jnp.prod( jnp.array( [ bases[j](x[j], _k[j]) for j in range(d)] ) )

    return basis_fn
    


    