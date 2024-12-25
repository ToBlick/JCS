# %%
from jax import vmap
from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax.experimental.sparse import bcsr_fromdense
from jax.experimental.sparse.linalg import spsolve
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.pullbacks import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.vector_bases import get_vector_basis_fn

import matplotlib.pyplot as plt 
# %%

# Map to physical domain: unit cube to [-3,3]^3
def F(x):
    return 6*x - jnp.ones_like(x)*3

def F_inv(x):    
    return (x + jnp.ones_like(x)*3) / 6

def B_hopf(x, s=1, ω1=1, ω2=1):
    x1, x2, x3 = x
    r = jnp.sqrt(jnp.sum(x**2))
    prefactor = 4 * jnp.sqrt(s) / (jnp.pi * (1 + r**2)**3 * jnp.sqrt(ω1**2 + ω2**2) )
    return jnp.array([ 2 * (ω2 * x2 - ω1 * x1 * x3),
                     - 2 * (ω2 * x1 + ω1 * x2 * x3),
                      ω1 * (-1 + x1**2 + x2**2 - x3**2)]) * prefactor
    
B_hopf_ref = pullback_2form(B_hopf, F)
H_hopf_ref = pullback_1form(B_hopf, F)

def energy(x):
    return jnp.sum(B_hopf(x)**2)

e_ref = pullback_3form(energy, F)
    
# Bases
n_r, p_r = 8, 2
n_θ, p_θ = 8, 2
n_ζ, p_ζ = 8, 2

Omega = ((0, 1), (0, 1), (0, 1))
        
basis_r = jit(get_spline(n_r, p_r, 'clamped'))
basis_θ = jit(get_spline(n_θ, p_θ, 'clamped'))
basis_ζ = jit(get_spline(n_ζ, p_ζ, 'clamped'))

basis_dr = jit(get_spline(n_r - 1, p_r - 1, 'clamped'))
basis_dθ = jit(get_spline(n_θ - 1, p_θ - 1, 'clamped'))
basis_dζ = jit(get_spline(n_ζ - 1, p_ζ - 1, 'clamped'))

# %%
for i in range(n_r):
    plt.plot(vmap(basis_r, (0, None))(jnp.linspace(0, 1, 100), i))

# %%
x_q, w_q = quadrature_grid(
            # get_quadrature_spectral(31)(0, 1),
            # get_quadrature_spectral(31)(0, 1),
            # get_quadrature_spectral(31)(0, 1))
            get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, n_ζ - p_ζ + 1), 15))

# Zero-forms
basis0 = get_tensor_basis_fn(
                    (basis_r, basis_θ, basis_ζ), 
                    (n_r, n_θ, n_ζ))
N0 = n_r * n_θ * n_ζ

# One-forms
basis1_1 = get_tensor_basis_fn(
            (basis_dr, basis_θ, basis_ζ), 
            (n_r - 1, n_θ, n_ζ))
N1_1 = (n_r - 1) * n_θ * n_ζ
basis1_2 = get_tensor_basis_fn(
            (basis_r, basis_dθ, basis_ζ), 
            (n_r, n_θ - 1, n_ζ))
N1_2 = n_r * (n_θ - 1) * n_ζ
basis1_3 = get_tensor_basis_fn(
            (basis_r, basis_θ, basis_dζ), 
            (n_r, n_θ, n_ζ - 1))
N1_3 = n_r * n_θ * (n_ζ - 1)
basis1 = get_vector_basis_fn(
            (basis1_1, basis1_2, basis1_3), 
            (N1_1, N1_2, N1_3))
N1 = N1_1 + N1_2 + N1_3

# Two-forms
basis2_1 = get_tensor_basis_fn(
            (basis_r, basis_dθ, basis_dζ), 
            (n_r, n_θ - 1, n_ζ - 1))
N2_1 = (n_r) * (n_θ - 1) * (n_ζ - 1)
basis2_2 = get_tensor_basis_fn(
            (basis_dr, basis_θ, basis_dζ), 
            (n_r - 1, n_θ, n_ζ - 1))
N2_2 = (n_r - 1) * (n_θ) * (n_ζ - 1)
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
            (n_r - 1, n_θ - 1, n_ζ - 1))
N3 = (n_r - 1) * (n_θ - 1) * (n_ζ - 1)

# %%
# Mass matrices
_M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
import time
start = time.time()
M0 = assemble(_M0, jnp.arange(N0), jnp.arange(N0))
M0[0,0]
end = time.time()
print(end - start)

# %%
start = time.time()
M02 = sparse_assemble_3d(_M0, (n_r, n_θ, n_ζ), 3)
M02[0,0]
end = time.time()
print(end - start)

M0 = bcsr_fromdense(M0)
# print(jnp.linalg.norm(M0 - M02))

# %%
shapes_1forms = ( (n_r - 1, n_θ, n_ζ), 
                  (n_r, n_θ - 1, n_ζ), 
                  (n_r, n_θ, n_ζ - 1) )

# %%
_M1 = jit(get_mass_matrix_lazy_1(basis1, x_q, w_q, F))
M1_1 = assemble(_M1, jnp.arange(N1_1), jnp.arange(N1_1))
M1_2 = assemble(_M1, jnp.arange(N1_1, N1_1+N1_2), jnp.arange(N1_1, N1_1+N1_2))
M1_3 = assemble(_M1, jnp.arange(N1_1+N1_2, N1), jnp.arange(N1_1+N1_2, N1))
M1 = bcsr_fromdense(block_diag(M1_1, M1_2, M1_3))
# M1 = jnp.array(block_diag(M1_1, M1_2, M1_3))

# %%
M1_smart_rowI = sparse_assemble_row_3d_vec(0, _M1, shapes_1forms, 3)

# %%
jnp.linalg.norm(M1[0] - M1_smart_rowI)
# %%
# M1_smart = sparse_assemble_3d_vec(_M1, shapes_1forms, 3)
# %%
_M2 = jit(get_mass_matrix_lazy_2(basis2, x_q, w_q, F))
M2_1 = assemble(_M2, jnp.arange(N2_1), jnp.arange(N2_1))
M2_2 = assemble(_M2, jnp.arange(N2_1, N2_1 + N2_2), jnp.arange(N2_1, N2_1 + N2_2))
M2_3 = assemble(_M2, jnp.arange(N2_1 + N2_2, N2), jnp.arange(N2_1 + N2_2, N2))
M2 = bcsr_fromdense(block_diag(M2_1, M2_2, M2_3))
# %%
_M3 = jit(get_mass_matrix_lazy_3(basis3, x_q, w_q, F))
M3 = get_sparse_operator(_M3, jnp.arange(N3), jnp.arange(N3))
# %%
# Projections
proj0 = get_l2_projection(basis0, x_q, w_q, N0)
proj1 = get_l2_projection(basis1, x_q, w_q, N1)
proj2 = get_l2_projection(basis2, x_q, w_q, N2)
proj3 = get_l2_projection(basis3, x_q, w_q, N3)
# %%
B_hat = spsolve(M2.data, M2.indices, M2.indptr, proj2(H_hopf_ref))
B_h = jit(get_u_h_vec(B_hat, basis2))

H_hat = spsolve(M1.data, M1.indices, M1.indptr, proj1(B_hopf_ref))
H_h = jit(get_u_h_vec(H_hat, basis1))
# %%
e_hat = spsolve(M0.data, M0.indices, M0.indptr, proj0(e_ref))
e_h = jit(get_u_h(e_hat, basis0))
# %%
nx = 8
_x1 = jnp.linspace(-3, 3, nx)
_x2 = jnp.linspace(-3, 3, nx)
_x3 = jnp.linspace(-3, 3, nx)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3)) # shape 3, n_x, n_x, 1
_x = _x.transpose(1, 2, 3, 0).reshape((nx)**3, 3)

H_h_x = vmap( pullback_1form(H_h, F_inv) )(_x)

# # %%
# from mpl_toolkits.mplot3d import axes3d

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.quiver(_x[:,0], _x[:,1], _x[:,2], 
#           H_h_x[:,0], H_h_x[:,1], H_h_x[:,2],
#           length=10)
# plt.show()
# %%
import plotly.graph_objects as go
fig = go.Figure(data=go.Streamtube(x=_x[:,0], y=_x[:,1], z=_x[:,2],
                                   u=H_h_x[:,0], v=H_h_x[:,1], w=H_h_x[:,2],
    ))
fig.show()
# %%
nx = 512
_x1 = jnp.linspace(-3, 3, nx)
_x2 = jnp.array([0])
_x3 = jnp.array([0])
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3)) # shape 3, n_x, n_x, 1
_x = _x.transpose(1, 2, 3, 0).reshape((nx), 3)

e_h_x = vmap(pullback_0form(e_h, F_inv))(_x)
# %%
# plt.contourf(_x1, _x2, e_h_x.reshape(nx, nx))
# plt.colorbar()
plt.plot(_x1, e_h_x)

# %%
