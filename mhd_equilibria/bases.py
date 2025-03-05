import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, grad
from functools import partial
import orthax

__all__ = ['_trig_fn', 'get_trig_fn', '_binom',
    '_get_legendre_coeff', '__get_legendre_coeffs', '_get_legendre_coeffs',
    'get_legendre_fn_x', 'get_chebyshev_fn_x', 'get_polynomial_basis_fn',
    '_lin_to_cart', 'get_tensor_basis_fn',
    'get_u_h_vec', 'get_u_h']

### ! TODO:
# There is some bug going on with Legendre bases under autodiff

# TODO: Fix this int32 int64 business

###
# Fourier Basis
###

# 1D trig basis function ψ number i at point x in [0, L]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def _trig_fn(x, k, a, b):
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

def get_trig_fn(K, a, b):
    return lambda x, k: _trig_fn(x, k, a, b)


###
# Legendre Basis
###

def _binom(n, m):
  return jnp.exp(gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1))

def _get_legendre_coeff(k, n):
    return jnp.round(_binom(n, k) * _binom(n + k, k))

def __get_legendre_coeffs(n, N):
    _k = jnp.arange(n + 1)
    non_zero_coeffs = vmap(_get_legendre_coeff, (0, None))(_k, n)
    return jnp.concatenate([non_zero_coeffs, jnp.zeros(N - n)])

def _get_legendre_coeffs(N):
    _n = jnp.arange(N + 1)
    return jnp.array([__get_legendre_coeffs(n, N) for n in _n])
    # return vmap(_get_legendre_coeffs, (0, None), axis_size=N+1)(_n, N)

# 1D legendre basis function ψ number i at point x in [0, L]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def get_legendre_fn_x(K, a, b):
    coeffs = _get_legendre_coeffs(K)
    def _legendre_fn_x(x, k):
        _n = jnp.arange(len(coeffs[k]))
        _x = (-(x - a) / (b-a) )**_n
        return (-1)**k * jnp.sqrt((2*k + 1)/(b-a)) * jnp.dot(coeffs[k, :], _x)
        # _x = (2 * x - a - b) / (b - a)
        # return jnp.sqrt((2*k + 1)/(b-a)) * orthax.legendre.legval(_x, jnp.zeros(k+1).at[k].set(1))
    return _legendre_fn_x

def get_chebyshev_fn_x(K, a, b):
    coeffs = jnp.zeros(K+1)
    def _cheb_fn_x(x, k):
        _x = (2 * x - a - b) / (b - a)
        # k = jnp.int32(k)
        return jnp.sqrt(1/(b-a)) * orthax.chebyshev.chebval(_x, coeffs.at[k].set(1))
    return _cheb_fn_x

def get_polynomial_basis_fn(coeffs, a, b):
    def _basis_fn(x, k):
        _n = jnp.arange(len(coeffs[k]))
        _x = ( -(x-a)/(b-a) )**_n
        return jnp.dot(coeffs[k], _x)
    return _basis_fn

###
# Tensor Basis
###

# converts a linear index i to a cartesian index (i,j)
def _lin_to_cart(i, shape):
    """
    Convert a linear index to a Cartesian index.

    Parameters:
    i (int): The linear index to convert.
    shape (tuple of int): The shape of the array to which the index applies.

    Returns:
    tuple of int: The Cartesian index corresponding to the linear index.
    """
    return jnp.unravel_index(i, shape)

def get_tensor_basis_fn(bases, shape):
    """
    Generates a tensor basis function from a list of basis functions and a given shape.

    Args:
        bases (list of callables): A list of basis functions, where each function takes two arguments (x, k) and returns a value.
        shape (tuple of int): The shape of the tensor, used to convert linear indices to Cartesian coordinates.

    Returns:
        callable: A function that takes two arguments (x, k) and returns the product of the basis functions evaluated at the given coordinates.
    """
    d = len(bases)
    # TODO: vmap?
    def basis_fn(x, k):
        _k = jnp.array(_lin_to_cart(k, shape), dtype=jnp.int32)
        return jnp.prod( jnp.array( [ bases[j](x[j], _k[j]) for j in range(d)] ) )
    return basis_fn

# def construct_tensor_basis(shape, Omega):
#     n_r, n_θ, n_φ = shape
#     bases = (get_legendre_fn_x(n_r, *Omega[0]), 
#              get_trig_fn(n_θ, *Omega[1]),
#              get_trig_fn(n_φ, *Omega[2]))
#     basis_fn = get_tensor_basis_fn(bases, shape)
#     return basis_fn

###
# Discrete functions
###

def get_u_h_vec(u_hat, basis_fn):
    _k = jnp.arange(len(u_hat))
    def u_h(x):
        return vmap(basis_fn, (None, 0), out_axes=1)(x, _k) @ u_hat
    return u_h

def get_u_h(u_hat, basis_fn):
    _k = jnp.arange(len(u_hat))
    def u_h(x):
        return jnp.sum(u_hat * vmap(basis_fn, (None, 0))(x, _k))
    return u_h
