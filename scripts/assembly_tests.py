# %%
import jax
from jax import jit
import jax.numpy as jnp
import jax.experimental.sparse
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.vector_bases import get_vector_basis_fn

import matplotlib.pyplot as plt 

import time

# %%
# Map to physical domain: unit cube to [-3,3]^3
def F(x):
    return 6*x - jnp.ones_like(x)*3

def F_inv(x):    
    return (x + jnp.ones_like(x)*3) / 6

# Bases
n_r, p_r = 6, 2
n_θ, p_θ = 6, 2
n_ζ, p_ζ = 6, 2

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
assemble_M0 = jit(lambda : assemble(_M0, jnp.arange(N0), jnp.arange(N0)))
sparse_assemble_M0 = jit(lambda: sparse_assemble_3d(_M0, (n_r, n_θ, n_ζ), p_r))
vmap_assemble_M0 = jit(lambda: assemble_full_vmap(_M0, jnp.arange(N0), jnp.arange(N0)))
# %%
start = time.time()
M0 = assemble_M0()
M0[0,0]
end = time.time()
print("Assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = assemble_M0()
    M0[0,0]
end = time.time()
print("Assemble: " , (end - start)/3)
# %%
start = time.time()
M0 = sparse_assemble_M0()
M0[0,0]
end = time.time()
print("Sparse assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = sparse_assemble_M0()
    M0[0,0]
end = time.time()
print("Sparse assemble: " , (end - start)/3)
# %%
start = time.time()
M0 = vmap_assemble_M0()
M0[0,0]
end = time.time()
print("Vmap assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = vmap_assemble_M0()
    M0[0,0]
end = time.time()
print("Vmap assemble: " , (end - start)/3)
# %%
