from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

from mhd_equilibria.bases import *
from mhd_equilibria.projections import *

def get_mass_matrix_lazy(basis_fn, x_q, w_q, n):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def M_ij(i, j):
        return l2_product(get_basis(i), get_basis(j), x_q, w_q)
    return M_ij

def get_curl_operator(one_form_bases, two_form_bases, x_q, w_q, ns):
    def get_one_form_basis(j, k):
        return lambda x: one_form_bases[j](x, k)
    def get_two_form_basis(j, k):
        return lambda x: two_form_bases[j](x, k)
    def _curl_operator(f, j):
        def fj(x):
            return f(x)[j]
        _k = jnp.arange(ns[j], dtype=jnp.int32)
        return vmap(lambda i: l2_product(fj, get_one_form_basis(j, i), x_q, w_q) - l2_product(grad(fj), get_two_form_basis(j, i), x_q, w_q))(_k) 
    def curl_operator(f):
        _d = jnp.arange(len(ns), dtype=jnp.int32)
        return tuple([_curl_operator(f, j) for j in _d])
    return curl_operator

# def get_curl_form(basis_fn):
#     def _C(i,j,x):
#         # compute the inner product of the Laplacian of the ith basis function
#         # with the jth basis function
#         gradϕ_j = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, j)
#         gradϕ_i = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, i)
#         return -jnp.sum(gradϕ_i * gradϕ_j) / x.shape[0]

# # assemble
# K = vmap(vmap(_K, (0, None, None)), (None, 0, None))(n_s, n_s, x_q)