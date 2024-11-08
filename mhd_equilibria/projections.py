from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

def cyl_jacobian(x):
    r, θ, z = x
    return r

def integral(f, x_q, w_q):
    return jnp.sum(vmap(f)(x_q) * w_q)

def l2_product(f, g, x_q, w_q):
    def integrand(x):
        return jnp.sum(f(x) * g(x))
    return integral(integrand, x_q, w_q)

def get_l2_projection(basis_fn, x_q, w_q, n):
    # TODO: This assumes orthonormal bases
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_projection(f):
        _k = jnp.arange(n, dtype=jnp.int32)
        return vmap(lambda i: l2_product(f, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

def get_u_h(u_hat, basis_fn):
    _k = jnp.arange(len(u_hat), dtype=jnp.int32)
    def u_h(x):
        return jnp.sum(u_hat * vmap(basis_fn, (None, 0))(x, _k))
    return u_h

def get_u_h_vec(u_hat, basis_fn):
    # u_hat: (n, d)
    _k = jnp.arange(u_hat.shape[0], dtype=jnp.int32)
    _d = jnp.arange(u_hat.shape[1], dtype=jnp.int32)
    def u_h(x):
        return jnp.array([ jnp.sum(u_hat[:,i] * vmap(basis_fn, (None, 0))(x, _k)) for i in _d ])
    return u_h

def get_mass_matrix_lazy(basis_fn, x_q, w_q, n):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def M_ij(i, j):
        return l2_product(get_basis(i), get_basis(j), x_q, w_q)
    return M_ij