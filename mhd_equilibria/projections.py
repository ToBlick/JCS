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


# TODO n_s should really be stored in the basis_fns
def get_l2_projection_vec(basis_fns, x_q, w_q, ns):
    def get_basis(j, k):
        return lambda x: basis_fns[j](x, k)
    def _l2_projection(f, j):
        def fj(x):
            return f(x)[j]
        _k = jnp.arange(ns[j], dtype=jnp.int32)
        return vmap(lambda i: l2_product(fj, get_basis(j, i), x_q, w_q))(_k) 
    def l2_projection(f):
        _d = jnp.arange(len(ns), dtype=jnp.int32)
        return tuple([_l2_projection(f, j) for j in _d])
    return l2_projection

