# %%
from mhd_equilibria.vector_bases import get_vector_basis_fn
from mhd_equilibria.pullbacks import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, config, jacfwd
config.update("jax_enable_x64", True)
from functools import partial
import numpy.testing as npt
from jax.experimental.sparse.linalg import spsolve

from mhd_equilibria.bases import *
from mhd_equilibria.splines import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.operators import *

import matplotlib.pyplot as plt

Omega = ((0, 1), (0, 1), (0, 1))

η = 0.3315
ρ = 4
μ0P2 = 0.277
ψ0 = 0.1
r0 = 1.04

def F(x_hat):
    R_hat, Z_hat, phi_hat = x_hat
    R = R_hat * r0 + 0.001
    Z = Z_hat
    phi = phi_hat * 2 * jnp.pi
    return jnp.array([R, Z, phi])

def F_inv(x):
    R, Z, phi = x
    R_hat = R / r0
    Z_hat = Z
    phi_hat = phi / (2 * jnp.pi)
    return jnp.array([R_hat, Z_hat, phi_hat])

# %%
n = 16
p = 3

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

basis0_shape = (n_r - 2, n_θ - 2, n_ζ)
basis0 = get_tensor_basis_fn(
                    (basis_r, basis_θ0, basis_ζ), 
                    basis0_shape)
N0 = (n_r - 1) * (n_θ - 2) * n_ζ

x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
            get_quadrature_periodic(1)(*Omega[2]))

def psi_analytic(x):
    R, Z, φ = x
    return (1 - R)**4
def lambda_analytic(x):
    R, Z, φ = x
    return (psi_analytic(x)/R**2)
lambda_analytic_hat = pullback_0form(lambda_analytic, F)
psi_analytic_hat = pullback_0form(psi_analytic, F)

# # Plotting stuff
nx = 100
_R_hat = jnp.linspace(0, 1, nx)
_Z_hat = jnp.linspace(0, 1, nx)
_phi_hat = jnp.array([0])
_x_hat = jnp.array(jnp.meshgrid(_R_hat, _Z_hat, _phi_hat)) # shape 3, n_x, n_x, 1
_x_hat = _x_hat.transpose(1, 2, 3, 0).reshape((nx)**2, 3)
        
proj0 = get_l2_projection(basis0, x_q, w_q, N0)
_M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
# %%
M0 = assemble(_M0, jnp.arange(N0), jnp.arange(N0))
lambda0_hat = jnp.linalg.solve(M0, proj0(pullback_3form(lambda_analytic, F)))
lambda0_h = jit(get_u_h(lambda0_hat, basis0))

# def solve(M, b):
#     return jax.experimental.sparse.linalg.spsolve(M.data, M.indices, M.indptr, b)

# %%
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
    
@jit
def gs_mass_operator(i, j):
    DF = jacfwd(F)
    def f(x): 
        return basis0_0(x, i)
    def g(x):
        R = F(x)[0]
        return basis0_0(x, j) * jnp.linalg.det(DF(x)) * R**5 * 2*μ0P2 / ψ0**2
    return l2_product(f, g, x_q, w_q)

def flux(x):
    return 0.0

@jit
def f_lazy(i):
    DF = jacfwd(F)
    σ = lambda x: basis0_0(x, i)
    # F F'
    def r(x):
        R = F(x)[0]
        return flux(F(x)) * jnp.linalg.det(DF(x)) * R
    return l2_product(r, σ, x_q, w_q)
f_hat = vmap(f_lazy)(jnp.arange(N0_0))

# %%
def boundary_operator(i):
    f = lambda x: basis0_0(x, i)
    return weak_gs_operator(f, lambda_analytic_hat, F)
b_hat = vmap(boundary_operator)(jnp.arange(N0_0))
# %%
plt.contourf(_R_hat, _Z_hat, vmap(lambda0_h)(_x_hat).reshape(nx, nx), levels=50)
plt.colorbar()
# %%
L = assemble(L_lazy, jnp.arange(N0_0), jnp.arange(N0_0))

# %%
Q = assemble(gs_mass_operator, jnp.arange(N0_0), jnp.arange(N0_0))

# %%
jnp.linalg.cond(L - Q)


# %%
plt.imshow(L - Q)
plt.colorbar()

# %%
lambda_hat = jnp.linalg.solve(L - Q, b_hat)

# %%
def lambda_h(x):
    return get_u_h(lambda_hat, basis0_0)(x)

def psi_h(x):
    R = F(x)[0]
    return R**2 * lambda_h(x)

nx = 100
_R = jnp.linspace(1e-6, r0, nx)
_Z = jnp.linspace(0, 1, nx)
_phi = jnp.array([0])
_x = jnp.array(jnp.meshgrid(_R, _Z, _phi)) # shape 3, n_x, n_x, 1
_x = _x.transpose(1, 2, 3, 0).reshape((nx)**2, 3)

psi_h = jit(pullback_0form(psi_h, F_inv))
plt.contourf(_R, _Z, vmap(psi_h)(_x).reshape(nx, nx), levels=25)
plt.colorbar()
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')

# %%

def Δstar(psi):
    def Δstar_psi(x):
        R = x[0]
        J_inv = jnp.eye(3).at[0,0].set(1/R)
        J = jnp.eye(3).at[0,0].set(R)
        scaled_grad_psi = lambda x: J_inv @ grad(psi)(x)
        return jnp.trace(J @ jacfwd(scaled_grad_psi)(x))
        # return jnp.trace(jax.hessian(psi)(x)) - 1/R * jax.grad(psi)(x)[0]
    return Δstar_psi

# %%
lhs = vmap(Δstar(psi_h))(_x)
# %%
def Ppsi(x):
    R = x[0]
    return 2 * μ0P2 * R**2 * psi_h(x) / ψ0**2 

rhs = - vmap(Ppsi)(_x) - vmap(flux)(_x)
# %%
plt.plot(lhs, label=r'$\Delta^* \psi$')
plt.plot(rhs, label=r'$P \psi$')
plt.legend()
# %%
plt.contourf(_R, _Z, rhs.reshape(nx, nx), levels=25)
plt.colorbar()
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
# %%
plt.contourf(_R, _Z, (lhs - rhs).reshape(nx, nx), levels=25)
plt.colorbar()
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')

# %%
plt.plot(_R, (lhs - rhs).reshape(nx, nx)[nx//2, :])

# %%
plt.plot(_Z, (lhs - rhs).reshape(nx, nx)[:, nx//2])
# %%
