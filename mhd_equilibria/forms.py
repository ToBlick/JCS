from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.pullbacks import *
from mhd_equilibria.operators import curl

def get_mass_matrix_lazy(basis_fn, x_q, w_q, F):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def M_ij(i, j):
        return l2_product(get_basis(i), get_basis(j), x_q, w_q)
    return M_ij

def get_mass_matrix_lazy_0(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def f(k):
        return lambda x: basis_fn(x, k)
    def g(k):
        return lambda x: basis_fn(x, k) * jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(f(i), g(j), x_q, w_q)
    return M_ij

def get_mass_matrix_lazy_1(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def A(k):
        return lambda x: inv33(DF(x)).T @ basis_fn(x, k)
    def E(k):
        return lambda x: inv33(DF(x)).T @ basis_fn(x, k) * jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(A(i), E(j), x_q, w_q)
    return M_ij

def get_mass_matrix_lazy_2(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def B(k):
        return lambda x: DF(x) @ basis_fn(x, k)
    def S(k):
        return lambda x: DF(x) @ basis_fn(x, k) / jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(B(i), S(j), x_q, w_q)
    return M_ij

def get_mass_matrix_lazy_3(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def f(k):
        return lambda x: basis_fn(x, k)
    def g(k):
        return lambda x: basis_fn(x, k) / jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(f(i), g(j), x_q, w_q)
    return M_ij

# TODO
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

def get_curl_matrix_lazy(basis_fn, x_q, w_q, F):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def C_ij(i, j):
        return l2_product(get_basis(i), curl(get_basis(j)), x_q, w_q)
    return C_ij

def get_1_form_trace_lazy(basis_fn, x_q, w_q, F):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_product(A, E, x_q, w_q):
        def integrand(x):
            return (jnp.sum(A(x) * jnp.array([0,1,1])) 
                    + jnp.sum(E(x) * jnp.array([0,1,1])))/2
        return integral(integrand, x_q, w_q)
    def T_ij(i, j):
        return l2_product(get_basis(i), get_basis(j), x_q, w_q)
    return T_ij

def get_2_form_trace_lazy(basis_fn, x_q, w_q, F):
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_product(B, S, x_q, w_q):
        def integrand(x):
            return (jnp.sum(B(x) * jnp.array([1,0,0])) 
                    + jnp.sum(S(x) * jnp.array([1,0,0])))
        return integral(integrand, x_q, w_q)
    def T_ij(i, j):
        return l2_product(get_basis(i), get_basis(j), x_q, w_q)
    return T_ij

# def get_curl_form(basis_fn):
#     def _C(i,j,x):
#         # compute the inner product of the Laplacian of the ith basis function
#         # with the jth basis function
#         gradϕ_j = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, j)
#         gradϕ_i = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, i)
#         return -jnp.sum(gradϕ_i * gradϕ_j) / x.shape[0]

# # assemble
# K = vmap(vmap(_K, (0, None, None)), (None, 0, None))(n_s, n_s, x_q)

def assemble(M_lazy, ns, ms):
    return jax.vmap(lambda i: jax.vmap(lambda j: M_lazy(i, j))(ns))(ms)