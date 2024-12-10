# %%
import jax
import jax.experimental
import time
from mhd_equilibria.vector_bases import get_vector_basis_fn
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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

### Analytical solutions

# S1 = 0.176
# S2 = 0.5
# S3 = -0.496
# S4 = 0.198
# S5 = 0.011

# def psi_analytic(x):
#     R, Z, phi = x
#     return (-S1/8 * R**2 - S2/2 * Z**2 + S3 + S4 * R**2 + S5 * (R**4 - 4*R**2*Z**2))

κ = 1.5
q = 1.5
B0 = 1.0
R0 = 1.5
μ0 = 1.0
        
c0 = B0 / (R0**2 * κ * q)
c1 = B0 * ( κ**2 + 1) / ( R0**2 * κ * q )
c2 = 0.0

def F(x_hat):
    R_hat, Z_hat, phi_hat = x_hat
    R = R_hat * R0 * 1.2 + R0 * 0.2
    Z = Z_hat * 3 - 1.5
    phi = phi_hat * 2 * jnp.pi
    return jnp.array([R, Z, phi])

def F_inv(x):
    R, Z, phi = x
    R_hat = (R/R0 - 0.2) / 1.2
    Z_hat = (Z + 1.5) / 3
    phi_hat = phi / (2 * jnp.pi)
    return jnp.array([R_hat, Z_hat, phi_hat])
        
p0 = 1.8

def psi_analytic(x):
    R, Z, φ = x
    return B0/(2 * R0**2 * κ * q) * ( R**2 * Z**2 + κ**2/4 * (R**2 - R0**2)**2 )
def lambda_analytic(x):
    R, Z, φ = x
    return (psi_analytic(x)/R**2)
lambda_analytic_hat = pullback_0form(lambda_analytic, F)
psi_analytic_hat = pullback_0form(psi_analytic, F)

# %%
# Bases
errors = []
ns = [5, 8, 10, 16]
ps = [1, 2, 3, 4]
import time
for n in ns:
    for p in ps:
        n_r, p_r = n, p
        n_θ, p_θ = n, p
        n_ζ, p_ζ = 1, 1
        
        basis_r = jit(get_spline(n_r, p_r, 'clamped'))
        basis_θ = jit(get_spline(n_θ, p_θ, 'clamped'))
        basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))

        basis_r0 = lambda x, i: basis_r(x, i+1)
        basis_θ0 = lambda x, i: basis_θ(x, i+1)

        basis_dr = jit(get_spline(n_r - 1, p_r - 1, 'clamped'))
        basis_dθ = jit(get_spline(n_θ - 1, p_θ - 1, 'clamped'))
        basis_dζ = basis_ζ

        basis0_0_shape = (n_r - 2, n_θ - 2, n_ζ)
        basis0_0 = get_tensor_basis_fn(
                            (basis_r0, basis_θ0, basis_ζ), 
                            basis0_0_shape)
        N0_0 = (n_r - 2) * (n_θ - 2) * n_ζ

        basis0_shape = (n_r, n_θ, n_ζ)
        basis0 = get_tensor_basis_fn(
                            (basis_r, basis_θ, basis_ζ), 
                            basis0_shape)
        N0 = n_r * n_θ * n_ζ

        x_q, w_q = quadrature_grid(
                    get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
                    get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
                    get_quadrature_periodic(1)(*Omega[2]))

        # # Plotting stuff
        nx = 100
        _R_hat = jnp.linspace(0, 1, nx)
        _Z_hat = jnp.linspace(0, 1, nx)
        _phi_hat = jnp.array([0])
        _x_hat = jnp.array(jnp.meshgrid(_R_hat, _Z_hat, _phi_hat)) # shape 3, n_x, n_x, 1
        _x_hat = _x_hat.transpose(1, 2, 3, 0).reshape((nx)**2, 3)

        # plt.contourf(_R_hat, _Z_hat, vmap(psi_analytic_hat)(_x_hat).reshape(nx, nx), levels=50)
        # plt.colorbar()
        # plt.xlabel(r'$\hat{R}$')
        # plt.ylabel(r'$\hat{Z}$')
        
        _R = jnp.linspace(0.2*R0, 1.4*R0, nx)
        _Z = jnp.linspace(-1.5, 1.5, nx)
        _phi = jnp.array([0])
        _x = jnp.array(jnp.meshgrid(_R, _Z, _phi)) # shape 3, n_x, n_x, 1
        _x = _x.transpose(1, 2, 3, 0).reshape((nx)**2, 3)

        # plt.contourf(_R, _Z, vmap(psi_analytic)(_x).reshape(nx, nx), levels=50)
        # plt.colorbar()
        # plt.xlabel(r'$R$')
        # plt.ylabel(r'$Z$')

        start = time.time()
        proj0 = get_l2_projection(basis0, x_q, w_q, N0)
        _M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
        # M0 = sparse_assemble_3d(_M0, basis0_shape, 3)
        # M0 = jax.experimental.sparse.bcsr_fromdense(M0)
        
        proj0_0 = get_l2_projection(basis0_0, x_q, w_q, N0_0)
        _M0_0 = jit(get_mass_matrix_lazy_0(basis0_0, x_q, w_q, F))
        # M0_0 = sparse_assemble_3d(_M0_0, basis0_0_shape, 3)
        # M0_0 = jax.experimental.sparse.bcsr_fromdense(M0_0)
        start = time.time()
        M0 = get_sparse_operator(_M0, jnp.arange(N0), jnp.arange(N0))
        M0_0 = get_sparse_operator(_M0_0, jnp.arange(N0_0), jnp.arange(N0_0))
        end = time.time()
        print("Time to assemble mass matrices: ", end - start)
        def solve(M, b):
            return jax.experimental.sparse.linalg.spsolve(M.data, M.indices, M.indptr, b)

        lambda0_hat = solve(M0, proj0(pullback_3form(lambda_analytic, F)))
        lambda0_h = jit(get_u_h(lambda0_hat, basis0))

        plt.contourf(_R_hat, _Z_hat, (vmap(lambda0_h)(_x_hat)).reshape(nx, nx), levels=50)
        plt.colorbar()
        plt.xlabel(r'$\hat{R}$')
        plt.ylabel(r'$\hat{Z}$')

        def weak_gs_operator(f, g, F):
            DF = jacfwd(F)
            def A(x):
                return inv33(DF(x)).T @ grad(f)(x)
            def E(x):
                R = F(x)[0]
                return inv33(DF(x)).T @ grad(g)(x) * jnp.linalg.det(DF(x)) * R**3
            return l2_product(A, E, x_q, w_q)

        @jit
        def L_lazy(i, j):
            f = lambda x: basis0_0(x, i)
            g = lambda x: basis0_0(x, j)
            return weak_gs_operator(f, g, F)

        start = time.time()
        # L = sparse_assemble_3d(L_lazy, basis0_0_shape, 3)
        # L = jax.experimental.sparse.bcsr_fromdense(L)
        L = get_sparse_operator(L_lazy, jnp.arange(N0_0), jnp.arange(N0_0))
        end = time.time()
        print("Time to assemble L: ", end - start)

        def boundary_operator(i):
            f = lambda x: basis0_0(x, i)
            return weak_gs_operator(f, lambda_analytic_hat, F)

        def f_lazy(i):
            DF = jacfwd(F)
            σ = lambda x: basis0_0(x, i)
            # F F' and P'
            def r(x):
                R = F(x)[0]
                return (- c2 * R0**2 - c1 * R**2 / μ0) * jnp.linalg.det(DF(x)) * R
            return l2_product(r, σ, x_q, w_q)

        f_hat = vmap(f_lazy)(jnp.arange(N0_0))
        b_hat = vmap(boundary_operator)(jnp.arange(N0_0))
        
        rhs = f_hat - b_hat
        start = time.time()
        lambda_hat = solve(L, rhs)
        end = time.time()
        print("Time to solve: ", end - start)

        @jit
        def lambda_h(x):
            return get_u_h(lambda_hat, basis0)(x) + lambda0_h(x)

        def psi_h(x):
            R = F(x)[0]
            return R**2 * lambda_h(x)

        # # %%
        # plt.contourf(_R_hat, _Z_hat, vmap(lambda_h)(_x_hat).reshape(nx, nx), levels=50)
        # plt.colorbar()
        # # %%
        # plt.contourf(_R_hat, _Z_hat, vmap(lambda0_h)(_x_hat).reshape(nx, nx), levels=50)
        # plt.colorbar()
        # # %%
        plt.contourf(_R_hat, _Z_hat, vmap(psi_h)(_x_hat).reshape(nx, nx), levels=50)
        plt.colorbar()
        # # %%
        # plt.contourf(_R_hat, _Z_hat, vmap(psi_analytic_hat)(_x_hat).reshape(nx, nx), levels=50)
        # plt.colorbar()
        # # %%

        # psi_proj_hat = solve(M0, proj0(pullback_3form(psi_analytic, F)))
        # psi_proj_h = jit(get_u_h(psi_proj_hat, basis0))
        # # %%
        # plt.contourf(_R_hat, _Z_hat, vmap(psi_proj_h)(_x_hat).reshape(nx, nx), levels=50)
        # plt.colorbar()

        def err(f, g):
            def d(x):
                return f(x) - g(x)
            return l2_product(d, d, x_q, w_q) / l2_product(g, g, x_q, w_q)

        print("n = ", n, "p = ", p)
        print(err(psi_h, psi_analytic_hat))
        # print(err(psi_proj_h, psi_analytic_hat))
        # print(err(psi_proj_h, psi_h))
        
        errors.append(err(psi_h, psi_analytic_hat))

# %%
errors = jnp.array(errors).reshape((len(ns), len(ps)))

plt.plot(ns, errors[:, 0], label='p = 1')
plt.plot(ns, errors[:, 1], label='p = 2')
plt.plot(ns, errors[:, 2], label='p = 3')
plt.plot(ns, errors[:, 3], label='p = 4')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('n')
plt.ylabel(r'Relative error of $\psi$')

# %%
ns = jnp.array(ns) * 1.0
# Plot the errors with markers
plt.plot(ns, errors[:, 0], label='p = 1', marker='v', color='k')
plt.plot(ns, errors[:, 1], label='p = 2', marker='o', color='r')
plt.plot(ns, errors[:, 2], label='p = 3', marker='s', color='g')
plt.plot(ns, errors[:, 3], label='p = 4', marker='^', color='b')

# Add polynomial decay reference lines
# Adjust the scaling factor if needed so they appear nicely on the plot.
plt.plot(ns[3:], 3e-2 * ns[3:]**(-2), 'k--', label=r'$n^{-2}$')
plt.plot(ns[3:], 3e-1 * ns[3:]**(-4), 'r--', label=r'$n^{-4}$')
plt.plot(ns[3:], 3e-0 * ns[3:]**(-6), 'g--', label=r'$n^{-6}$')
plt.plot(ns[3:], 7e1 * ns[3:]**(-8), 'b--', label=r'$n^{-8}$')

# Set log scales
plt.yscale('log')
plt.xscale('log')

# Configure grid (including minor ticks)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=1)
plt.grid(which='minor', linestyle='--', linewidth=0.5)

# Labels and legend
plt.xlabel('n')
plt.ylabel(r'Relative error of $\psi$')
plt.legend()

plt.tight_layout()
plt.show()

# %%
