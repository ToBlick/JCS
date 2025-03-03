import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, grad
from functools import partial
import orthax

__all__ = [
            '_fringe_idx', 
            '_ansi_idx', 
            '_unravel_ansi_idx', 
            'get_radial_zernike_coeff', 
            '_get_radial_zernike_coeff',
            '_get_radial_zernike_coeffs', 
            '__get_radial_zernike_coeffs',
            'get_zernike_fn_x', 
            'get_zernike_fn_radial_derivative', 
            'get_zernike_tensor_basis_fn']

###
# Zernike Basis
###

def _fringe_idx(n, l):
    return int( (1 + (n + jnp.abs(l)) / 2)**2 - 2 * jnp.abs(l) + jnp.floor( (1 - jnp.sign(l)) / 2 ) )

def _ansi_idx(n, m):
    return int( n * (n + 2) + m ) // 2

def _unravel_ansi_idx(j):
    n = (jnp.ceil( (-3 + jnp.sqrt(9 + 8 * j)) / 2 )).astype(int)
    m = 2 * j - n * (n + 2)
    return n, m

# Zernike polynomial number j, kth coefficient, ansi indices
def get_radial_zernike_coeff(k, j):
    n, l = _unravel_ansi_idx(j)
    m = jnp.abs(l)
    return _get_radial_zernike_coeff(k, n, m)

def _get_radial_zernike_coeff(k, n, m):
    def _nonzero_coeff(n, m, k):
        return jnp.round( (-1)**k * _binom(n - k, k) * _binom(n - 2*k, (n - m) / 2 - k) )
    def _zero_coeff(n, m, k):
        return k * 0.0
    return jax.lax.cond( (n - m) % 2 == 0, _nonzero_coeff, _zero_coeff, n, m, k)

# get all coefficients corresponding to index j, up to J
def __get_radial_zernike_coeffs(j, J):
    n, m = _unravel_ansi_idx(j)
    n_max, m_max = _unravel_ansi_idx(J)
    k_max = n_max // 2 - 1
    _k = jnp.arange((n-jnp.abs(m))//2 + 1)
    non_zero_coeffs = vmap(get_radial_zernike_coeff, (0, None))(_k, j)
    return jnp.concatenate([non_zero_coeffs, jnp.zeros(k_max - _k[-1] + 1)])

def _get_radial_zernike_coeffs(J):
    _j = jnp.arange(J + 1)
    return jnp.array([__get_radial_zernike_coeffs(j, J) for j in _j])

# 2D Zernike basis function ψ number j at point x = (r,theta) in [a, b] x [c, d]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def get_zernike_fn_x(J, a, b, c, d):
    coeffs = _get_radial_zernike_coeffs(J)
    Lθ = d - c
    Lr = b - a
    r1 = lambda x: jnp.sqrt(1/Lθ)
    r2 = lambda x: jnp.sqrt(2/Lθ)
    def _zernike_fn_x(x, j):
        r, θ = x
        n, l = _unravel_ansi_idx(j)
        m = jnp.abs(l)
        _k = jnp.arange(len(coeffs[j]))
        _r = ( (r - a) / Lr )**(n - 2 * _k)
        mθ = (θ - c) / Lθ * 2 * jnp.pi * m
        angular_term = jax.lax.cond(l < 0, jnp.sin, jnp.cos, mθ)
        neumann_factor = jax.lax.cond(m == 0, r1, r2, m)
        return jnp.sqrt((2*n + 2)/Lr) * jnp.dot(coeffs[j, :], _r) * angular_term * neumann_factor
    return _zernike_fn_x

def get_zernike_fn_radial_derivative(N, a, b, c, d):
    coeffs = _get_radial_zernike_coeffs(N)
    Lθ = d - c
    Lr = b - a
    r1 = lambda x: jnp.sqrt(1/Lθ)
    r2 = lambda x: jnp.sqrt(2/Lθ)
    r3 = lambda x: 1.0
    r4 = lambda x: 0.0
    def _zernike_fn_x(x, j):
        r, θ = x
        n, l = _unravel_ansi_idx(j)
        m = jnp.abs(l)
        _k = jnp.arange(len(coeffs[j]))
        _r = ( (r - a) / Lr )**(n - 2 * _k - 1) * (n - 2 * _k) / Lr
        mθ = (θ - c) / Lθ * 2 * jnp.pi * m
        angular_term = jax.lax.cond(l < 0, jnp.sin, jnp.cos, mθ)
        neumann_factor = jax.lax.cond(m == 0, r1, r2, m)
        constant_offset = jax.lax.cond(n == 0, r3, r4, m)
        return jnp.sqrt((2*n + 2)/Lr) * jnp.dot(coeffs[j, :], _r) * angular_term * neumann_factor + constant_offset
    return _zernike_fn_x

###
# Tensor Basis
###

def get_zernike_tensor_basis_fn(bases, shape):
    # TODO: vmap?
    def basis_fn(x, k):
        rθ, z = x[:-1], x[-1]
        _k = jnp.array(_lin_to_cart(k, shape), dtype=jnp.int32)
        poloidal_plane = bases[0](rθ, _k[0])
        toroidal = bases[1](z, _k[1])
        return poloidal_plane * toroidal
    return basis_fn

def construct_zernike_tensor_basis(shape, Omega):
    n_r_times_n_θ, n_φ = shape
    bases = (get_zernike_fn_x(n_r_times_n_θ, *Omega[0], *Omega[1]), 
             get_trig_fn(n_φ, *Omega[2]))
    basis_fn = get_zernike_tensor_basis_fn(bases, shape)
    return basis_fn