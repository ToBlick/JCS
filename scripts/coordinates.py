# %%
from jax import jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

import numpy as np

from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.plotting import *
from mhd_equilibria.operators import *
from mhd_equilibria.pullbacks import *
from mhd_equilibria.quadratures import *

import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

import time

# %%
### Constants
κ = 1.5
q = 1.5
B0 = 1.0
R0 = 2.0
μ0 = 1.0

χ = 1.0 # friction coefficient
γ = 1.3 # adiabatic index
        
c0 = B0 / (R0**2 * κ * q)
c1 = B0 * ( κ**2 + 1) / ( R0**2 * κ * q )
c2 = 0.0
        
p0 = 1.8

Φ = jnp.pi/4
        
### Analytical solutions
def Psi(x):
    R, φ, Z = x
    return B0/(2 * R0**2 * κ * q) * ( R**2 * Z**2 + κ**2/4 * (R**2 - R0**2)**2 )

# Define the outer flux surface
def f(x):
    T = Psi(jnp.array([R0 - 0.7, Φ, 0.0]))
    return Psi(x) - T

f_sq = lambda x: f(x)**2

### Quadrature grid
Omega = ((1., 3.), (-1.5, 1.5))
nx = 256
_R = jnp.linspace(*Omega[0], nx)
_Φ = jnp.array([Φ])
_Z = jnp.linspace(*Omega[1], nx)
hR = (_R[-1] - _R[0]) / nx
hZ = (_Z[-1] - _Z[0]) / nx
w_q = hR * hZ * 2 * jnp.pi
        
x = jnp.array(jnp.meshgrid(_R, _Φ, _Z)) # shape 3, n_x, n_x, 1
x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)

# # %%
# _r = jnp.linspace(0, 1.0, nx)
# _θ = jnp.linspace(0, 2*jnp.pi, nx)
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.contourf(_R, _Z, vmap(lambda y: jnp.abs(f(y)))(x).reshape(nx, nx).T, 100)
# ax.contour(_R, _Z, vmap(f)(x).reshape(nx, nx).T, levels = [0], colors='white')    
# ax.plot(R0 + 0.9 * jnp.cos(_θ), 0.9 * jnp.sin(_θ), color = 'red')
# ax.set_xlabel(r'$R$')
# ax.set_ylabel(r'$Z$')
# ax.set_aspect('auto')
# plt.show()

# %%
### Basis
n = 32
_bases = (get_trig_fn_x(n, 0, 2*jnp.pi), )
shape = (n,)
basis_fn = get_tensor_basis_fn(_bases, shape) # basis_fn(x, k)
_ns = jnp.arange(n, dtype=jnp.int32)

ς_hat = jnp.zeros(n)
ς_hat = ς_hat.at[0].set(1.0)

def ς(θ):
    # special care here since the u_h fct is not written with scalar functions in mind.
    return get_u_h(ς_hat, basis_fn)(θ)

# %%
_r = jnp.linspace(0, 1.0, nx)
_θ = jnp.linspace(0, 2*jnp.pi, nx)
fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(_R, _Z, vmap(lambda y: jnp.abs(f(y)))(x).reshape(nx, nx).T, 100)
ax.contour(_R, _Z, vmap(f)(x).reshape(nx, nx).T, levels = [0], colors='white')    
ax.plot(R0 + vmap(ς)(_θ[:, None]) * jnp.cos(_θ), 
        vmap(ς)(_θ[:, None]) * jnp.sin(_θ), 
        color = 'red')
ax.set_xlabel(r'$R$')
ax.set_ylabel(r'$Z$')
ax.set_aspect('equal')
plt.show()
# %%
@jit
def residual(ς_hat):
    ς = get_u_h(ς_hat, basis_fn)
    def res(θ):
        R = R0 + ς(θ) * jnp.cos(θ)
        Z = ς(θ) * jnp.sin(θ)
        x = jnp.array([R, Φ * jnp.ones(1), Z])
        return f_sq(x)
    return jnp.mean(vmap(res)(_θ[:, None]))

# %%
import optax
from jax import value_and_grad

solver = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(
                        max_backtracking_steps=50,
                        store_grad=True
                        )
                    )
# have to jit this for performance
opt_state = solver.init(ς_hat)
value_and_grad = optax.value_and_grad_from_state(residual)

params = [ ς_hat ]
# optimization loop 
for _ in range(100):
    value, grad = value_and_grad(ς_hat, state=opt_state)
    if jnp.sum( grad**2 ) < 1e-9:
        break
    updates, opt_state = solver.update(
        grad, opt_state, ς_hat, value=value, 
        grad=grad, value_fn=residual
    )
    ς_hat = optax.apply_updates(ς_hat, updates)
    params.append( ς_hat )


# %%
cmap = plt.get_cmap("inferno")

_r = jnp.linspace(0, 1.0, nx)
_θ = jnp.linspace(0, 2*jnp.pi, nx)
fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(_R, _Z, vmap(lambda y: jnp.abs(f(y)))(x).reshape(nx, nx).T, 100)
ax.contour(_R, _Z, vmap(f)(x).reshape(nx, nx).T, levels = [0], colors='white')    
for (i, p) in enumerate(params):
    ς = get_u_h(p, basis_fn)
    ax.plot(R0 + vmap(ς)(_θ[:, None]) * jnp.cos(_θ), 
            vmap(ς)(_θ[:, None]) * jnp.sin(_θ), 
            color = cmap(i/len(params)))
ax.set_xlabel(r'$R$')
ax.set_ylabel(r'$Z$')
ax.set_aspect('equal')
plt.show()

# %%
for (i, p) in enumerate(params):
    ς = get_u_h(p, basis_fn)
    plt.plot(_θ, vmap(ς)(_θ[:, None]), color = cmap(i/len(params)))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\varsigma(\theta)$')
plt.show()


# %%

def F(x):
    r, θ, z = x
    θ = jnp.array([θ])
    R = R0 + ς(θ) * r * jnp.cos(θ)
    phi = -z/R0
    Z = ς(θ) * r * jnp.sin(θ)
    R = jnp.squeeze(R)
    Z = jnp.squeeze(Z)
    return jnp.array([R, phi, Z])
    
def F_inv(x):
    R, phi, Z = x
    z = - R0 * phi
    θ = jnp.arctan2(Z, R - R0)
    _θ = jnp.array([θ])
    r = jnp.sqrt( (R - R0)**2 + Z**2) / ς(_θ)
    r = jnp.squeeze(r)
    return jnp.array([r, θ, z])

def f_hat(x):
    r, θ = x
    return 2 * jnp.exp(-r**2/2/0.3**2) * r**2 * (1 + jnp.cos(θ)**2)

def f_hat_3d(x):
    r, θ, z = x
    return f_hat(jnp.array([r, θ]))

f = pullback_0form(f_hat_3d, F_inv)

# %%
nx = 256
_r = np.linspace(1e-2, 1, nx)
_θ = np.linspace(0, 2*np.pi, nx)
d = 2
x_hat = jnp.array(jnp.meshgrid(_r, _θ)).reshape(d, nx**2).T
__r, __θ = np.meshgrid(_r, _θ)

# Convert tokamak coordinates to Cylindrical ones for plotting
x_hat_ext = jnp.column_stack([x_hat, jnp.zeros(x_hat.shape[0])])
x = vmap(F)(x_hat_ext)
R = x[:, 0].reshape(nx, nx)
Z = x[:, 2].reshape(nx, nx)

# %%
f_vals = jax.vmap(f)(x).reshape(nx, nx)
# f_vals = jax.vmap(f_hat)(x_hat)
f_vals_reshaped = f_vals.reshape(nx, nx)

# %%
# Plot the function on the unit disk
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.pcolormesh(R, Z, f_vals, shading='auto', cmap='viridis')
_x0 = jnp.column_stack( [_r, jnp.zeros_like(_r)] )
plt.plot(R0 + ς(jnp.zeros(1)) * _r, vmap(f_hat)(_x0), color='w')
plt.arrow(R0, 0, 0.9 * ς(jnp.zeros(1)), 0, color='w', head_width=0.02)
_x0 = jnp.column_stack( [_r, jnp.pi * jnp.ones_like(_r)] )
plt.plot(R0 - ς(jnp.array([jnp.pi])) * (_r), vmap(f_hat)(_x0), color='w')
plt.arrow(R0, 0, -0.9 * ς(jnp.array([jnp.pi])), 0, color='w', head_width=0.02)
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()
# %%

grad_f_hat = jax.grad(f_hat_3d)
grad_f = pullback_1form(grad_f_hat, F_inv)
grad_f_vals = jax.vmap(grad_f)(x)
grad_f_vals_reshaped = grad_f_vals.reshape(nx, nx, 3)

# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.pcolormesh(R, Z, f_vals_reshaped, shading='auto', cmap='viridis')
plt.quiver( x[::300, 0], 
            x[::300, 2],
            grad_f_vals[::300,0],
            grad_f_vals[::300,2],
            color = 'w',
            )
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()
# %%
Omega = ((0, 1), (0, 2*jnp.pi), (0, 2*jnp.pi))
n_r, n_θ, n_φ = 5, 5, 1
N = n_r * n_θ * n_φ
basis_fn = construct_tensor_basis((n_r, n_θ, n_φ), Omega)
lowres_basis_fn = construct_tensor_basis((1, 1, 1), Omega)

x_q, w_q = quadrature_grid(get_quadrature(15)(*Omega[0]),
                           get_quadrature_periodic(16)(*Omega[1]),
                           get_quadrature_periodic(1)(*Omega[2]))

basis_fns = (basis_fn, basis_fn, lowres_basis_fn)
ns = (N, N, 1)

def get_mass_matrices(bases, x_q, w_q, ns):
    Ms = []
    for i, basis_fn in enumerate(bases):
        N = ns[i]
        M_ij = get_mass_matrix_lazy(basis_fn, x_q, w_q, N)
        M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)
        M = jnp.where(M < 1e-10, 0.0, M)
        Ms.append(M)
    return Ms
# %%
Ms = get_mass_matrices(basis_fns, x_q, w_q, ns)
# %%
𝚷1 = get_l2_projection_vec(basis_fns, x_q, w_q, ns)

# %%
grad_f_hat_dofs = 𝚷1(grad_f_hat)
# %%
def get_u_h_vec(u_hat, basis_fns):
    # u_hat: d-tuple with n_j elements
    _d = jnp.arange(len(u_hat), dtype=jnp.int32)
    def u_h(x):
        return jnp.array([ jnp.sum(u_hat[i] * vmap(basis_fns[i], (None, 0))(x, jnp.arange(len(u_hat[i]), dtype=jnp.int32))) for i in _d ])
    return u_h

grad_f_hat_h = get_u_h_vec(grad_f_hat_dofs, basis_fns)
grad_f_h = pullback_1form(grad_f_hat_h, F_inv)
grad_f_h_vals = jax.vmap(grad_f_h)(x)
# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.pcolormesh(R, Z, f_vals_reshaped, shading='auto', cmap='viridis')
plt.quiver( x[::300, 0], 
            x[::300, 2],
            grad_f_h_vals[::300,0],
            grad_f_h_vals[::300,2],
            color = 'r',
            alpha = 0.5
            )
plt.quiver( x[::300, 0], 
            x[::300, 2],
            grad_f_vals[::300,0],
            grad_f_vals[::300,2],
            color = 'w',
            alpha = 0.5
            )
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()
# %%
def error_1forms(u, v, F):
    def _err(x):
        return u(x) - v(x)
    _int1 = inner_product_1form(_err, _err, F)
    _int2 = inner_product_1form(v, v, F)
    return jnp.sqrt( integral(_int1, x_q, w_q) ) / jnp.sqrt( integral(_int2, x_q, w_q) )

print(error_1forms(grad_f_hat_h, grad_f_hat, F))
