# %%
import jax

from mhd_equilibria.vector_bases import get_vector_basis_fn
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from functools import partial
import numpy.testing as npt

import jax.experimental.sparse

from mhd_equilibria.bases import *
from mhd_equilibria.splines import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.operators import *

import matplotlib.pyplot as plt

Omega = ((0, 1), (0, 1), (0, 1))

# %%
def A(x):
    x1, x2, x3 = x
    r = (jnp.sum((x1-0.5)**2 + (x2-0.5)**2))
    return jnp.array([0, 0, 1]) * jnp.exp(-20 * r) + jnp.ones(3)

B = curl(A)

# %%
nx = 64
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3)) # shape 3, n_x, n_x, 1
_x = _x.transpose(1, 2, 3, 0).reshape((nx)**2, 3)

# %%

# Bases
def F(x):
    return x
F_inv = F

n_r, p_r = 12, 3
n_θ, p_θ = 12, 3
n_ζ, p_ζ = 1, 1

Omega = ((0, 1), (0, 1), (0, 1))
        
basis_r = jit(get_spline(n_r, p_r, 'clamped'))
basis_θ = jit(get_spline(n_θ, p_θ, 'clamped'))
basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))

basis_r0 = lambda x, i: basis_r(x, i+1)
basis_θ0 = lambda x, i: basis_θ(x, i+1)

basis_dr = jit(get_spline(n_r - 1, p_r - 1, 'clamped'))
basis_dθ = jit(get_spline(n_θ - 1, p_θ - 1, 'clamped'))
basis_dζ = basis_ζ

x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
            get_quadrature_periodic(1)(*Omega[2]))
# %%

# Zero-forms
basis0 = get_tensor_basis_fn(
                    (basis_r0, basis_θ0, basis_ζ), 
                    (n_r - 2, n_θ - 2, n_ζ))
N0 = (n_r - 2) * (n_θ - 2) * n_ζ

# One-forms
basis1_1 = get_tensor_basis_fn(
            (basis_dr, basis_θ0, basis_ζ), 
            (n_r - 1, n_θ - 2, n_ζ))
N1_1 = (n_r - 1) * (n_θ - 2) * n_ζ
basis1_2 = get_tensor_basis_fn(
            (basis_r0, basis_dθ, basis_ζ), 
            (n_r - 2, n_θ - 1, n_ζ))
N1_2 = (n_r - 2) * (n_θ - 1) * n_ζ
basis1_3 = get_tensor_basis_fn(
            (basis_r0, basis_θ0, basis_dζ), 
            (n_r - 2, n_θ - 2, n_ζ))
N1_3 = (n_r - 2) * (n_θ - 2) * (n_ζ)
basis1 = get_vector_basis_fn(
            (basis1_1, basis1_2, basis1_3), 
            (N1_1, N1_2, N1_3))
N1 = N1_1 + N1_2 + N1_3

# Two-forms
basis2_1 = get_tensor_basis_fn(
            (basis_r0, basis_dθ, basis_dζ), 
            (n_r - 2, n_θ - 1, n_ζ))
N2_1 = (n_r - 2) * (n_θ - 1) * (n_ζ)
basis2_2 = get_tensor_basis_fn(
            (basis_dr, basis_θ0, basis_dζ), 
            (n_r - 1, n_θ - 2, n_ζ))
N2_2 = (n_r - 1) * (n_θ - 2) * (n_ζ)
basis2_3 = get_tensor_basis_fn(
            (basis_dr, basis_dθ, basis_ζ), 
            (n_r - 1, n_θ - 1, n_ζ))
N2_3 = (n_r - 1) * (n_θ - 1) * n_ζ
basis2 = get_vector_basis_fn(
            (basis2_1, basis2_2, basis2_3), 
            (N2_1, N2_2, N2_3))
N2 = N2_1 + N2_2 + N2_3

# Three-forms
basis3 = get_tensor_basis_fn(
            (basis_dr, basis_dθ, basis_dζ), 
            (n_r - 1, n_θ - 1, n_ζ))
N3 = (n_r - 1) * (n_θ - 1) * (n_ζ)
# %%
# Project B to 2-forms
proj0 = get_l2_projection(basis0, x_q, w_q, N0)
proj1 = get_l2_projection(basis1, x_q, w_q, N1)
proj2 = get_l2_projection(basis2, x_q, w_q, N2)
proj3 = get_l2_projection(basis3, x_q, w_q, N3)

# %%
_M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
M0 = get_sparse_operator(_M0, jnp.arange(N0), jnp.arange(N0))

_M1 = jit(get_mass_matrix_lazy_1(basis1, x_q, w_q, F))
M1_1 = assemble(_M1, jnp.arange(N1_1), jnp.arange(N1_1))
M1_2 = assemble(_M1, jnp.arange(N1_1, N1_1+N1_2), jnp.arange(N1_1, N1_1+N1_2))
M1_3 = assemble(_M1, jnp.arange(N1_1+N1_2, N1), jnp.arange(N1_1+N1_2, N1))
M1 = jax.experimental.sparse.bcsr_fromdense(jax.scipy.linalg.block_diag(M1_1, M1_2, M1_3))
# %%

_M2 = jit(get_mass_matrix_lazy_2(basis2, x_q, w_q, F))
M2_1 = assemble(_M2, jnp.arange(N2_1), jnp.arange(N2_1))
M2_2 = assemble(_M2, jnp.arange(N2_1, N2_1 + N2_2), jnp.arange(N2_1, N2_1 + N2_2))
M2_3 = assemble(_M2, jnp.arange(N2_1 + N2_2, N2), jnp.arange(N2_1 + N2_2, N2))
M2 = jax.experimental.sparse.bcsr_fromdense(jax.scipy.linalg.block_diag(M2_1, M2_2, M2_3))

# _M3 = jit(get_mass_matrix_lazy_3(basis3, x_q, w_q, F))
# M3 = get_sparse_operator(_M3, jnp.arange(N3), jnp.arange(N3))

rhs = get_double_crossproduct_projection(basis1, x_q, w_q, N1, F)

# %%
B_hat = jax.experimental.sparse.linalg.spsolve(M2.data, M2.indices, M2.indptr, proj2(B))
B_h = jit(get_u_h_vec(B_hat, basis2))

# %%

plt.quiver(_x[:, 0], _x[:, 1], vmap(B)(_x)[:, 0], vmap(B)(_x)[:, 1], color = 'k', alpha=0.5)
plt.quiver(_x[:, 0], _x[:, 1], vmap(B_h)(_x)[:, 0], vmap(B_h)(_x)[:, 1], color = 'c', alpha=0.5)

# %%
_M12 = jit(get_mass_matrix_lazy_12(basis1, basis2, x_q, w_q, F))
M12 = get_sparse_operator(_M12, jnp.arange(N1), jnp.arange(N2))

# %%
M21 = jax.experimental.sparse.bcsr_fromdense(jax.experimental.sparse.csr_todense(M12).T)
# %%
def solve(M, b):
    return jax.experimental.sparse.linalg.spsolve(M.data, M.indices, M.indptr, b)

H_hat = solve(M1, M12 @ B_hat)
H_h = jit(get_u_h_vec(H_hat, basis1))
# %%
plt.quiver(_x[:, 0], _x[:, 1], vmap(H_h)(_x)[:, 0], vmap(H_h)(_x)[:, 1], color = 'k', alpha=0.5)
plt.quiver(_x[:, 0], _x[:, 1], vmap(B_h)(_x)[:, 0], vmap(B_h)(_x)[:, 1], color = 'c', alpha=0.5)
# %%
_C = get_curl_matrix_lazy(basis1, x_q, w_q, F)
C = get_sparse_operator(_C, jnp.arange(N1), jnp.arange(N1))
# %%
J_hat = solve(M1, C @ H_hat)
J_h = jit(get_u_h_vec(J_hat, basis1))
# %%
plt.contourf(_x1, _x2, vmap(J_h)(_x)[:, 2].reshape(nx, nx))
plt.colorbar()

# %%
E_hat = solve(M1, rhs(J_h, H_h, H_h))
E_h = jit(get_u_h_vec(E_hat, basis1))
# %%
plt.contourf(_x1, _x2, vmap(E_h)(_x)[:, 2].reshape(nx, nx))
plt.colorbar()

# %%
def get_curl_matrix_lazy_12(basis1, basis2, x_q, w_q, F):
    def get_basis1(k):
        return lambda x: basis1(x, k)
    def get_basis2(k):
        return lambda x: basis2(x, k)
    def C_ij(j, i):
        return l2_product(get_basis2(i), curl(get_basis1(j)), x_q, w_q)
    return C_ij

_C12 = jit(get_curl_matrix_lazy_12(basis1, basis2, x_q, w_q, F))
C12 = get_sparse_operator(_C12, jnp.arange(N2), jnp.arange(N1))
# %%
dB_hat = solve(M2, C12 @ E_hat)
dB_h = jit(get_u_h_vec(dB_hat, basis2))
# %%
plt.quiver(_x[:, 0], _x[:, 1], vmap(dB_h)(_x)[:, 0], vmap(dB_h)(_x)[:, 1], color = 'k', alpha=0.5)
# %%
