# %%
from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

import numpy as np

from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.plotting import *
from mhd_equilibria.operators import *

import matplotlib.pyplot as plt

import time

# %%
### Constants
κ = 1.9
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
ax.set_aspect('auto')
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
# (this takes forever because of the laplacian, we only do 100 iterations)
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
ax.set_aspect('auto')
plt.show()

# %%
for (i, p) in enumerate(params):
    ς = get_u_h(p, basis_fn)
    plt.plot(_θ, vmap(ς)(_θ[:, None]), color = cmap(i/len(params)))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\varsigma(\theta)$')
plt.show()

