# %%
from jax import jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

import numpy as np

from mhd_equilibria.bases import *
from mhd_equilibria.vector_bases import *
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
κ = 1.6
q = 1.6
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
    T = Psi(jnp.array([R0 - 0.75, Φ, 0.0]))
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
_bases = (get_trig_fn(n, 0, 2*jnp.pi), )
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
    if jnp.sum( grad**2 ) < 1e-16:
        break
    updates, opt_state = solver.update(
        grad, opt_state, ς_hat, value=value, 
        grad=grad, value_fn=residual
    )
    ς_hat = optax.apply_updates(ς_hat, updates)
    params.append( ς_hat )


# %%
cmap = plt.get_cmap("inferno")

Omega = ((0.2, 1), (0, 2*jnp.pi), (0, 2*jnp.pi))

_r = jnp.linspace(*Omega[0], nx)[1:]
_θ = jnp.linspace(*Omega[1], nx)[1:]
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
    X = R * jnp.cos(phi)
    Y = R * jnp.sin(phi)
    return jnp.array([X, Y, Z])
    
def F_inv(x):
    X, Y, Z = x
    R = jnp.sqrt(X**2 + Y**2)
    phi = jnp.arctan2(Y, X)
    # R, phi, Z = x
    z = - R0 * phi
    θ = jnp.arctan2(Z, R - R0)
    _θ = jnp.array([θ])
    r = jnp.sqrt( (R - R0)**2 + Z**2) / ς(_θ)
    r = jnp.squeeze(r)
    return jnp.array([r, θ, z])

def f_hat(x):
    r, θ = x
    return 2 * jnp.exp(-r**2/2/0.4**2) * r**2 * (1 * jnp.cos(θ)**2)

def f_hat_3d(x):
    r, θ, z = x
    return f_hat(jnp.array([r, θ]))

f = pullback_0form(f_hat_3d, F_inv)

# %%
nx = len(_r)
d = 2
x_hat = jnp.array(jnp.meshgrid(_r, _θ)).reshape(d, nx**2).T

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

grad_f_direct = jax.grad(pullback_0form(f_hat, F_inv))
grad_f_direct_vals = jax.vmap(grad_f)(x)

# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.pcolormesh(R, Z, f_vals_reshaped, shading='auto', cmap='viridis')
plt.quiver(           x[::80, 0], 
                      x[::80, 2],
            grad_f_vals[::80,0],
            grad_f_vals[::80,2],
            color = 'w',
            )
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()
# %%
n_r, n_θ, n_φ = 28, 1, 1
N = n_r * n_θ * n_φ
basis_fn = construct_zernike_tensor_basis((n_r * n_θ, n_φ), Omega)
lowres_basis_fn = construct_zernike_tensor_basis((1, 1), Omega)
# basis_fn = construct_tensor_basis((n_r, n_θ, n_φ), Omega)
# lowres_basis_fn = construct_tensor_basis((1, 1, 1), Omega)

x_q, w_q = quadrature_grid(get_quadrature_spectral(61)(*Omega[0]),
                           get_quadrature_periodic(64)(*Omega[1]),
                           get_quadrature_periodic(1)(*Omega[2]))

basis_fns = (basis_fn, basis_fn, basis_fn)
ns_1forms = jnp.array((N, N, N))
N1 = jnp.sum(ns_1forms)

basis_fn_1forms = get_vector_basis_fn(basis_fns, ns_1forms)
basis_fn_0forms = basis_fn

# %%
_basis = lambda x: basis_fn(x, 12)
x_hat_ext = jnp.column_stack([x_hat, jnp.zeros(x_hat.shape[0])])
_vals = jax.vmap(_basis)(x_hat_ext)
_vals_reshaped = _vals.reshape(nx, nx)
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.pcolormesh(R, Z, _vals_reshaped, shading='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()

# %%
M0 = get_mass_matrix_lazy_0(basis_fn_0forms, x_q, w_q, F)
M0_assembled = vmap(vmap(M0, (0, None)), (None, 0))(jnp.arange(N), jnp.arange(N))
M0 = jnp.where(jnp.abs(M0_assembled) > 1e-16, M0_assembled, 0.0)
print(jnp.linalg.cond(M0_assembled))
print(jnp.sum(M0_assembled > 1e-16), jnp.sum(M0_assembled > 1e-16)/N, 100 *jnp.sum(M0_assembled > 1e-16) / N**2)
# %%
plt.imshow(M0_assembled)
plt.colorbar()

# %%
M0_hat = get_mass_matrix_lazy_0(basis_fn_0forms, x_q, w_q, lambda x: x)
M0_hat_assembled = vmap(vmap(M0_hat, (0, None)), (None, 0))(jnp.arange(N), jnp.arange(N))
𝚷0 = get_l2_projection(basis_fn_0forms, x_q, w_q, N)
f_hat_dofs = 𝚷0(pullback_0form(f, F))
f_hat_dofs = jnp.linalg.solve(M0_hat_assembled, f_hat_dofs)
f_hat_h = get_u_h(f_hat_dofs, basis_fn_0forms)
f_h = jit(pullback_0form(f_hat_h, F_inv))
f_h_vals = jax.vmap(f_h)(x).reshape(nx, nx)

# %%

def error_0forms(u, v, F):
    @jit
    def _err(x):
        return u(x) - v(x)
    _int1 = inner_product_0form(_err, _err, F)
    _int2 = inner_product_0form(v, v, F)
    return jnp.sqrt( integral(_int1, x_q, w_q) ) / jnp.sqrt( integral(_int2, x_q, w_q) )

print(error_0forms(f_hat_h, f_hat_3d, F))

cm = plt.get_cmap('viridis')
plt.figure(figsize=(10, 10))
plt.pcolormesh(R, Z, f_vals, shading='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.show()
# %%
M1_ref = get_mass_matrix_lazy(basis_fn_1forms, x_q, w_q, F)
M1_ref_assembled = vmap(vmap(M1_ref, (0, None)), (None, 0))(jnp.arange(N1), jnp.arange(N1))

# %%
M1 = get_mass_matrix_lazy_1(basis_fn_1forms, x_q, w_q, F)
# assemble
Ns = jnp.arange(N1)
M1_assembled = vmap(vmap(M1, (0, None)), (None, 0))(Ns, Ns)
M1_assembled = jnp.where(jnp.abs(M1_assembled) > 1e-16, M1_assembled, 0.0)
# M_assembled = [ M(i, j) for i in range(N1) for j in range(N1) ]
# %%
print(jnp.linalg.cond(M1_assembled[:ns_1forms[0],:ns_1forms[0]]), 
      jnp.linalg.cond(M1_assembled[ns_1forms[0]:ns_1forms[0]+ns_1forms[1],ns_1forms[0]:ns_1forms[0]+ns_1forms[1]]), 
      jnp.linalg.cond(M1_assembled[ns_1forms[0]+ns_1forms[1]:,ns_1forms[0]+ns_1forms[1]:]))
print(jnp.sum(M1_assembled > 1e-16), jnp.sum(M1_assembled > 1e-16)/N1, 100 * jnp.sum(M1_assembled > 1e-16) / N1**2)
# %%
plt.imshow(M1_assembled[:ns_1forms[0],:ns_1forms[0]])
plt.colorbar()
# %%
plt.imshow(M1_assembled[ns_1forms[0]:ns_1forms[0]+ns_1forms[1],ns_1forms[0]:ns_1forms[0]+ns_1forms[1]])
plt.colorbar()
# %%
𝚷1 = get_l2_projection(basis_fn_1forms, x_q, w_q, N1)

# %%
grad_f_hat_dofs = 𝚷1(pullback_1form(grad_f, F))
grad_f_hat_dofs = jnp.linalg.solve(M1_ref_assembled, grad_f_hat_dofs)

grad_f_hat_h = get_u_h_vec(grad_f_hat_dofs, basis_fn_1forms)
grad_f_h = jit(pullback_1form(grad_f_hat_h, F_inv))
grad_f_h_vals = jax.vmap(grad_f_h)(x)

# %%
grad_f_hat_dofs

# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(12, 12))
plt.pcolormesh(R, Z, f_vals_reshaped, shading='auto', cmap='viridis')
plt.quiver(             x[::80,0], 
                        x[::80,2],
            grad_f_h_vals[::80,0],
            grad_f_h_vals[::80,2],
            color = 'r',
            alpha = 0.5
            )
plt.quiver(           x[::80,0], 
                      x[::80,2],
            grad_f_vals[::80,0],
            grad_f_vals[::80,2],
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
    @jit
    def _err(x):
        return u(x) - v(x)
    _int1 = inner_product_1form(_err, _err, F)
    _int2 = inner_product_1form(v, v, F)
    return jnp.sqrt( integral(_int1, x_q, w_q) ) / jnp.sqrt( integral(_int2, x_q, w_q) )

print(error_1forms(grad_f_hat_h, grad_f_hat, F))

# %%
def naive_error_1forms(u, v, F):
    @jit
    def _err(x):
        return u(x) - v(x)
    _int1 = l2_product(_err, _err, x_q, w_q)
    _int2 = l2_product(v, v, x_q, w_q)
    return jnp.sqrt( _int1 ) / jnp.sqrt( _int2 )

print(naive_error_1forms(grad_f_hat_h, grad_f_hat, F))

# %%
C = get_curl_matrix_lazy(basis_fn_1forms, x_q, w_q, F)
C_assembled = vmap(vmap(C, (None, 0)), (0, None))(jnp.arange(N1), jnp.arange(N1))

# %%

def A(x):
    def _Psi(x):
        X, Y, Z = x
        R = jnp.sqrt(X**2 + Y**2)
        phi = jnp.arctan2(Y, X)
        _κ = κ
        _q = q
        return B0/(2 * R0**2 * _κ * _q) * ( R**2 * Z**2 + _κ**2/4 * (R**2 - R0**2)**2 )
    return jnp.array( [0, _Psi(x), 0] )

A_ref = pullback_1form(A, F)
B_ref = curl(A_ref)
J_ref = curl(B_ref)
B = pullback_2form(B_ref, F_inv)
J = pullback_2form(J_ref, F_inv)
B_vals = jax.vmap(B)(x)
J_vals = jax.vmap(J)(x)

# %%
_helicity_0 = inner_product_1form(A_ref, B_ref, F)
helicity_0 = integral(_helicity_0, x_q, w_q)

# %%
basis_fn_2forms = basis_fn_1forms
𝚷2 = get_l2_projection(basis_fn_2forms, x_q, w_q, N1)
B_ref_dofs = 𝚷2(pullback_2form(B, F))
H_ref_dofs = jnp.linalg.solve(M1_assembled, M1_ref_assembled @ B_ref_dofs)

# %%
H_ref_h = get_u_h_vec(H_ref_dofs, basis_fn_1forms)
B_ref_h = get_u_h_vec(B_ref_dofs, basis_fn_2forms)
H_h = jit(pullback_1form(H_ref_h, F_inv))
B_h = jit(pullback_2form(B_ref_h, F_inv))
H_h_vals = jax.vmap(H_h)(x)
B_h_vals = jax.vmap(B_h)(x)

# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.quiver(        x[::180,0], 
                   x[::180,2],
            H_h_vals[::180,0],
            H_h_vals[::180,2],
            color = 'k',
            label = 'H'
            )
plt.quiver(        x[::180,0], 
                   x[::180,2],
            B_h_vals[::180,0],
            B_h_vals[::180,2],
            color = 'c',
            label = 'B'
            )
plt.xlabel('R')
plt.ylabel('Z')
plt.legend()
plt.axis('equal')
plt.show()

# %%
J_ref_dofs = jnp.linalg.solve(M1_assembled, C_assembled @ H_ref_dofs)
J_ref_h = get_u_h_vec(J_ref_dofs, basis_fn_1forms)
rhs = get_double_crossproduct_projection(basis_fn_1forms, x_q, w_q, N1, F)
E_ref_dofs = jnp.linalg.solve(M1_assembled, rhs(J_ref_h, H_ref_h, H_ref_h))
E_ref_h = get_u_h_vec(E_ref_dofs, basis_fn_1forms)
E_h = jit(pullback_1form(E_ref_h, F_inv))
E_h_vals = jax.vmap(E_h)(x)
# %%
cm = plt.get_cmap('viridis')
plt.figure(figsize=(12, 12))
plt.pcolormesh(R, Z, E_h_vals[:,1].reshape(nx, nx), shading='auto', cmap='viridis')
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.colorbar()
plt.show()

# %%
def dB_hat(x):
    return curl(E_ref_h)(x)

dB = jit(pullback_2form(dB_hat, F_inv))
dB_vals = jax.vmap(dB)(x)

# %%
dB_ref_dofs = jnp.linalg.solve(M1_ref_assembled, C_assembled @ E_ref_dofs)
dB_ref_h = jit(get_u_h_vec(dB_ref_dofs, basis_fn_2forms))
dB_h = jit(pullback_2form(dB_ref_h, F_inv))
dB_h_vals = jax.vmap(dB_h)(x)

# %%

def plot_vector_form(dofs, basis, F_inv, pullback, x, plot_every=180):
    u_ref_h = jit(get_u_h_vec(dofs, basis))
    u_h = jit(pullback(u_ref_h, F_inv))
    u_h_vals = jax.vmap(u_h)(x)
    plt.figure(figsize=(6, 6))
    plt.quiver(        x[::plot_every,0], 
                       x[::plot_every,2],
                u_h_vals[::plot_every,0],
                u_h_vals[::plot_every,2],
                color = 'k',
                )
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.legend()
    plt.axis('equal')
    plt.show()

# %%
plot_vector_form(B_ref_dofs, basis_fn_2forms, F_inv, pullback_2form, x)

# %%
plot_vector_form(H_ref_dofs, basis_fn_1forms, F_inv, pullback_1form, x)

# %%
plot_vector_form(jnp.zeros_like(H_ref_dofs).at[0].set(1.0), basis_fn_1forms, F_inv, pullback_1form, x)

# %%
plot_vector_form(jnp.zeros_like(B_ref_dofs).at[5].set(1.0), basis_fn_2forms, F_inv, pullback_2form, x)

# %%
B_ref_dofs_0 = B_ref_dofs
B_ref_h_0 = jit(get_u_h_vec(B_ref_dofs_0, basis_fn_2forms))
B_h_0 = jit(pullback_2form(B_ref_h, F_inv))
B_h_0_vals = jax.vmap(B_h)(x)

E_values = []
dE_values = []

M11 = M1_assembled
M12 = M1_ref_assembled

# %%
x_q_bdy, w_q_bdy = quadrature_grid((jnp.ones(1), jnp.ones(1)),
                           get_quadrature_periodic(64)(*Omega[1]),
                           get_quadrature_periodic(1)(*Omega[2]))

T1 = get_1_form_trace_lazy(basis_fn_1forms, x_q_bdy, w_q_bdy, F)
T2 = get_2_form_trace_lazy(basis_fn_2forms, x_q_bdy, w_q_bdy, F)
# %%
T1_assembled = vmap(vmap(T1, (0, None)), (None, 0))(jnp.arange(N1), jnp.arange(N1))
T2_assembled = vmap(vmap(T2, (0, None)), (None, 0))(jnp.arange(N1), jnp.arange(N1))
# %%
eta = 1e-3

for _ in range(100):
    H_ref_dofs = jnp.linalg.solve(M11, M12 @ B_ref_dofs)
    H_ref_h = jit(get_u_h_vec(H_ref_dofs, basis_fn_1forms))
    J_ref_dofs = jnp.linalg.solve(M11, C_assembled @ H_ref_dofs)
    J_ref_h = jit(get_u_h_vec(J_ref_dofs, basis_fn_1forms))
    E_ref_dofs = jnp.linalg.solve(M11, rhs(J_ref_h, H_ref_h, H_ref_h))
    dB_ref_dofs = jnp.linalg.solve(M12, C_assembled @ E_ref_dofs)
    dH_ref_dofs = jnp.linalg.solve(M11, M12 @ dB_ref_dofs)
    dE_values.append(dB_ref_dofs @ M12.T @ H_ref_dofs)
    E_values.append(B_ref_dofs @ M12.T @ H_ref_dofs)
    B_ref_dofs += eta * dB_ref_dofs
print("Force squared = ", -rhs(J_ref_h, H_ref_h, H_ref_h) @ J_ref_dofs)   
print("dE = ", dE_values[-1])

fig, ax1 = plt.subplots()
# Primary y-axis
ax1.plot(E_values, label='Energy E')
ax1.set_xlabel('iterations')

ax2 = ax1.twinx()  # Create a twin y-axis
ax2.plot(dE_values, '--', label='delta E')

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))
plt.show()

# %%
B_ref_h = jit(get_u_h_vec(B_ref_dofs, basis_fn_2forms))
B_h = jit(pullback_2form(B_ref_h, F_inv))
B_h_vals = jax.vmap(B_h)(x)

cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.quiver(        x[::80,0], 
                   x[::80,2],
            B_h_vals[::80,0],
            B_h_vals[::80,2],
            color = 'k',
            label = 'B(t=1)'
            )
plt.quiver(          x[::80,0], 
                     x[::80,2],
            B_h_0_vals[::80,0],
            B_h_0_vals[::80,2],
            color = 'c',
            label = 'B(t=0)'
            )
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.legend()
plt.show()
# %%
dB_ref_h = jit(get_u_h_vec(dB_ref_dofs, basis_fn_2forms))
dB_h = jit(pullback_2form(dB_ref_h, F_inv))
dB_h_vals = jax.vmap(dB_h)(x)
cm = plt.get_cmap('viridis')
plt.figure(figsize=(6, 6))
plt.quiver(         x[::80,0], 
                    x[::80,2],
            dB_h_vals[::80,0],
            dB_h_vals[::80,2],
            color = 'k',
            label = 'dB'
            )
plt.xlabel('R')
plt.ylabel('Z')
plt.legend()
plt.axis('equal')
plt.show()
# %%
