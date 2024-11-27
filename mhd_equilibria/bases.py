import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, grad
from functools import partial
import orthax

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
    _k = jnp.arange(n + 1, dtype=jnp.int64)
    non_zero_coeffs = vmap(_get_legendre_coeff, (0, None))(_k, n)
    return jnp.concatenate([non_zero_coeffs, jnp.zeros(N - n)])

def _get_legendre_coeffs(N):
    _n = jnp.arange(N + 1, dtype=jnp.int64)
    return jnp.array([__get_legendre_coeffs(n, N) for n in _n])
    # return vmap(_get_legendre_coeffs, (0, None), axis_size=N+1)(_n, N)

# 1D legendre basis function ψ number i at point x in [0, L]:
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def get_legendre_fn_x(K, a, b):
    coeffs = _get_legendre_coeffs(K)
    def _legendre_fn_x(x, k):
        _n = jnp.arange(len(coeffs[k]), dtype=jnp.int64)
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
        _n = jnp.arange(len(coeffs[k]), dtype=jnp.int64)
        _x = ( -(x-a)/(b-a) )**_n
        return jnp.dot(coeffs[k], _x)
    return _basis_fn
    
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
    _k = jnp.arange((n-jnp.abs(m))//2 + 1, dtype=jnp.int64)
    non_zero_coeffs = vmap(get_radial_zernike_coeff, (0, None))(_k, j)
    return jnp.concatenate([non_zero_coeffs, jnp.zeros(k_max - _k[-1] + 1)])

def _get_radial_zernike_coeffs(J):
    _j = jnp.arange(J + 1, dtype=jnp.int64)
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
        _k = jnp.arange(len(coeffs[j]), dtype=jnp.int64)
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
        _k = jnp.arange(len(coeffs[j]), dtype=jnp.int64)
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

# converts a linear index i to a cartesian index (i,j)
def _lin_to_cart(i, shape):
    return jnp.unravel_index(i, shape)

def get_tensor_basis_fn(bases, shape):
    d = len(bases)
    # TODO: vmap?
    def basis_fn(x, k):
        _k = jnp.array(_lin_to_cart(k, shape), dtype=jnp.int64)
        return jnp.prod( jnp.array( [ bases[j](x[j], _k[j]) for j in range(d)] ) )
    return basis_fn

def get_zernike_tensor_basis_fn(bases, shape):
    # TODO: vmap?
    def basis_fn(x, k):
        rθ, z = x[:-1], x[-1]
        _k = jnp.array(_lin_to_cart(k, shape), dtype=jnp.int64)
        poloidal_plane = bases[0](rθ, _k[0])
        toroidal = bases[1](z, _k[1])
        return poloidal_plane * toroidal
    return basis_fn

def construct_tensor_basis(shape, Omega):
    n_r, n_θ, n_φ = shape
    bases = (get_legendre_fn_x(n_r, *Omega[0]), 
             get_trig_fn(n_θ, *Omega[1]),
             get_trig_fn(n_φ, *Omega[2]))
    basis_fn = get_tensor_basis_fn(bases, shape)
    return basis_fn

def construct_zernike_tensor_basis(shape, Omega):
    n_r_times_n_θ, n_φ = shape
    bases = (get_zernike_fn_x(n_r_times_n_θ, *Omega[0], *Omega[1]), 
             get_trig_fn(n_φ, *Omega[2]))
    basis_fn = get_zernike_tensor_basis_fn(bases, shape)
    return basis_fn

###
# Discrete functions
###

def get_u_h_vec(u_hat, basis_fn):
    _k = jnp.arange(len(u_hat), dtype=jnp.int32)
    def u_h(x):
        return vmap(basis_fn, (None, 0), out_axes=1)(x, _k) @ u_hat
    return u_h

def get_u_h(u_hat, basis_fn):
    _k = jnp.arange(len(u_hat), dtype=jnp.int32)
    def u_h(x):
        return jnp.sum(u_hat * vmap(basis_fn, (None, 0))(x, _k))
    return u_h
