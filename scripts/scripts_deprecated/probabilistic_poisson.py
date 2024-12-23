# %%
import jax
from jax import jit
import jax.numpy as jnp
import jax.experimental.sparse
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import laplacian
from mhd_equilibria.vector_bases import get_vector_basis_fn

import matplotlib.pyplot as plt 
jax.config.update("jax_enable_x64", True)
import time

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%
# Map to physical domain: unit cube to [-3,3]^3
def F(x):
    return x
F_inv = F

errors = []
n = 32
p = 3
ns = (n, 1, 1)
ps = (p, 0, 0)
types = ('clamped', 'fourier', 'fourier')
boundary = ('free', 'periodic', 'periodic')
basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)
basis1, shapes1, N1 = get_zero_form_basis((n-1, 1, 1), (p-1, 0, 0), types, boundary)

x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
            get_quadrature_periodic(1)(0,1),
            get_quadrature_periodic(1)(0,1))

nx = 512
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.zeros(1)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape((nx)*1*1, 3)

# for i in range(N0):
#     plt.plot(_x[:, 0], vmap(basis0, (0, None))(_x, i), color='black')
# for i in range(N1):
#     plt.plot(_x[:, 0], vmap(basis1, (0, None))(_x, i), color='c')
# plt.show()
    
# %%

def derivative_matrix_lazy(i, j):
    return l2_product(lambda x: grad(basis0)(x, i), lambda x: basis1(x, j), x_q, w_q)

G = assemble_full_vmap(derivative_matrix_lazy, jnp.arange(N0), jnp.arange(N1)).T

# %%
def mass_matrix_lazy_1(i, j):
    def get_basis(k):
        return lambda x: basis1(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)
M1 = assemble_full_vmap(mass_matrix_lazy_1, jnp.arange(N1), jnp.arange(N1))

def mass_matrix_lazy_0(i, j):
    def get_basis(k):
        return lambda x: basis0(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)
M0 = assemble_full_vmap(mass_matrix_lazy_0, jnp.arange(N0), jnp.arange(N0))

# %%
n_modes = 61
key = jax.random.PRNGKey(0)

def get_coeffs(key, n):
    a = jax.random.normal(key, (n,))
    weights = jnp.array([ 1/(i+1)**2 for i in range(n//2) ])
    a = a * jnp.sort(jnp.concatenate([jnp.zeros(1), weights, weights]), descending=True)
    return a

fourier_basis = get_trig_fn(n_modes, 0, 1)
_f = get_u_h(get_coeffs(key, n_modes), fourier_basis)
def f(x):
    return _f(x[0])

plt.plot(_x[:, 0], vmap(f)(_x), color='black', label='ψ_0')


###
# Gσ = p(f)
# Mσ = G.T u
# -> σ = M⁻¹ G.T u
# -> G M⁻¹ G.T u = p(f)
###

# %%
kappa = 5e-2
L = kappa * G @ jnp.linalg.solve(M0, G.T)

proj = get_l2_projection(basis1, x_q, w_q, N1)
# f_hat = jnp.linalg.solve(M, proj(f))
# f_h = get_u_h(f_hat, basis0)
u_hat = jnp.linalg.solve(L, proj(f))
f_hat = jnp.linalg.solve(M1, proj(f))
u_h = get_u_h(u_hat, basis1)
f_h = get_u_h(f_hat, basis1)

# %%
plt.plot(_x[:, 0], vmap(u_h)(_x), color='c', label='u_h')
plt.plot(_x[:, 0], vmap(f)(_x), color='grey', label='f')
plt.plot(_x[:, 0], -kappa * vmap(laplacian(u_h))(_x), color='pink', label='-Δu_h')   
plt.legend()
plt.show()
# %%
print(jnp.linalg.norm(L @ u_hat - M1 @ f_hat))
# %%
print((0.5 * u_hat.T @ L @ u_hat - u_hat.T @ M1 @ f_hat))
print(-(0.5 * u_hat.T @ M1 @ f_hat))

# %%
def get_sol_pair(key):
    _f = get_u_h(get_coeffs(key, n_modes), fourier_basis)
    def f(x):
        return _f(x[0])
    f_hat = jnp.linalg.solve(M1, proj(f))
    u_hat = jnp.linalg.solve(L, proj(f))
    return u_hat, f_hat

n_s = 32
key, _ = jax.random.split(key)
sol_pairs = vmap(get_sol_pair)(jax.random.split(key, n_s))
eps = 0.0
# %%
us, fs = sol_pairs

def add_noise(key, u, eps=0.01):
    _f = get_u_h(eps * get_coeffs(key, n_modes), fourier_basis)
    def f(x):
        return _f(x[0])
    f_hat = jnp.linalg.solve(M1, proj(f))
    return u + f_hat

us_noisy = vmap(add_noise, (0, 0, None))(jax.random.split(jax.random.PRNGKey(1), n_s), us, eps)

# %%
def L2_distance(u1, u2):
    return jnp.sqrt((u1 - u2) @ M1 @ (u1 - u2)) / 2

def H1_distance(u1, u2):
    return jnp.sqrt((u1 - u2) @ L @ (u1 - u2)) / 2

def energy(u, f):
    return 0.5 * u.T @ L @ u - u.T @ M1 @ f

def residual(u, f):
    return (L @ u - M1 @ f) @ (L @ u - M1 @ f) / 2

C_L2 = vmap(vmap(L2_distance, (0, None)), (None, 0))(us, fs)
C_H1 = vmap(vmap(H1_distance, (0, None)), (None, 0))(us, fs)
C_E = vmap(vmap(energy,       (0, None)), (None, 0))(us, fs)
C_R = vmap(vmap(residual,     (0, None)), (None, 0))(us, fs)
# %%
def plot_diff(u_hat, f_hat):
    u_h = get_u_h(u_hat, basis1)
    f_h = get_u_h(f_hat, basis1)
    plt.plot(_x[:, 0], vmap(u_h)(_x), color='c', label='u_h')
    plt.plot(_x[:, 0], vmap(f_h)(_x), color='grey', label='f')
    # plt.plot(_x[:, 0], -kappa * vmap(laplacian(u_h))(_x), color='pink', label='-Δu_h')   
    plt.legend()
    plt.show()
# %%
C_L2 = vmap(vmap(L2_distance, (0, None)), (None, 0))(us_noisy, fs)
C_H1 = vmap(vmap(H1_distance, (0, None)), (None, 0))(us_noisy, fs)
C_E =  vmap(vmap(energy,      (0, None)), (None, 0))(us_noisy, fs)
C_R =  vmap(vmap(residual,    (0, None)), (None, 0))(us_noisy, fs)
# %%
plt.imshow(C_L2)
plt.colorbar()
# %%
plt.imshow(C_H1)
plt.colorbar()
# %%
plt.imshow(C_E)
plt.colorbar()
# %%
plt.imshow(C_R)
plt.colorbar()
# %%
for i in range(n_s):
    plt.plot(C_E[i] - C_E[i,i], marker='o', linestyle='', markersize=2) 
# %%
for i in range(n_s):
    plt.plot(C_R[i] - C_R[i,i], marker='o', linestyle='', markersize=2)
# %%
for i in range(n_s):
    plt.plot(C_H1[i] - C_H1[i,i], marker='o', linestyle='', markersize=2)
# %%
for i in range(n_s):
    plt.plot(C_L2[i] - C_L2[i,i] + 1e-3, marker='o', linestyle='', markersize=2)

# %%

###
# Sinkhorn
###
epsilon = 0.1
def sinkhorn(a, b, C, epsilon=0.1, n_iter=100):
    K = jnp.exp(-C / epsilon)
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    for i in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return jnp.diag(u) @ K @ jnp.diag(v), u, v

a = jnp.ones(n_s) / n_s
b = jnp.ones(n_s) / n_s


# %%
P, u, v = sinkhorn(a, b, C_E, epsilon)
plt.imshow(P)
plt.colorbar()
# %%
plt.plot(u)
plt.plot(v)

def plot_diff(u_hat, f_hat, u_e_hat):
    u_h = get_u_h(u_hat, basis1)
    u_e = get_u_h(u_e_hat, basis1)
    f_h = get_u_h(f_hat, basis1)
    plt.plot(_x[:, 0], vmap(u_h)(_x), color='c', label='u_h')
    plt.plot(_x[:, 0], vmap(f_h)(_x), color='grey', label='f')
    plt.plot(_x[:, 0], vmap(u_e)(_x), color='pink', label='u_exact')
    plt.legend()
    plt.show()

# %%
key, _key = jax.random.split(key)
u_t, f_t = get_sol_pair(key)
u_t_noisy = add_noise(_key, u_t, eps)

# %%
def L2_distance(u, f):
    return 0.5 * jnp.sqrt((u - f) @ M1 @ (u - f))
def H1_distance(u, f):
    return 0.5 * jnp.sqrt((u - f) @ L @ (u - f))
def energy(u, f):
    return 0.5 * u.T @ L @ u - u.T @ M1 @ f
def residual(u, f):
    return 0.5 * (L @ u - M1 @ f) @ jnp.linalg.solve(M1, (L @ u - M1 @ f))

for C in [L2_distance, H1_distance, energy, residual]:
    epsilon = 0.05
    C_mat = vmap(vmap(C, (0, None)), (None, 0))(us_noisy, fs)
    def grad_C(y): 
        return grad(C, 1)(y, f_t)
    P, u, v = sinkhorn(a, b, C_mat, epsilon, 1000)
    grad_C_vec = vmap(grad_C)(us_noisy) # shape (n_s, N1)
    C_vec = vmap(C, (0, None))(us_noisy, f_t) # shape (n_s, N1)
    denom = jnp.exp( - C_vec / epsilon) @ v
    nom = (jnp.exp( - C_vec / epsilon) * v) @ grad_C_vec
    grad_u = nom / denom
    if C == L2_distance:
        u_est = f_t - jnp.linalg.solve(M1, grad_u)
    elif C == H1_distance:
        u_est = f_t - jnp.linalg.solve(L, grad_u)
    elif C == energy:
        u_est = - jnp.linalg.solve(M1, grad_u)
    elif C == residual:
        u_est = jnp.linalg.solve(L, M1 @ f_t - M1 @ grad_u)
    
    plot_diff(u_est, f_t, u_t)
    print((u_est - u_t) @ M1 @ (u_est - u_t))
# %%
