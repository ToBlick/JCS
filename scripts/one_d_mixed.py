# %%
import jax
from jax import jit
import jax.experimental
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
n = 128
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
def f(x):
    return (4 * jnp.pi)**2 * jnp.sin(4 * jnp.pi * x[0])
def u(x):
    return jnp.sin(4 * jnp.pi * x[0])

###
# Gσ = p(f)
# Mσ = G.T u
# -> σ = M⁻¹ G.T u
# -> G M⁻¹ G.T u = p(f)
###

L = G @ jnp.linalg.solve(M0, G.T)
L = jnp.where(jnp.abs(L) > 1e-4, L, jnp.zeros_like(L))
L_sp = jax.experimental.sparse.bcsr_fromdense(L)

print(L_sp)
# %%
proj = get_l2_projection(basis1, x_q, w_q, N1)
# f_hat = jnp.linalg.solve(M, proj(f))
# f_h = get_u_h(f_hat, basis0)
u_hat = jnp.linalg.solve(L, proj(f))
u_h = get_u_h(u_hat, basis1)

def err(x):
    return jnp.sum((u_h(x) - u(x))**2)

error = integral(err, x_q, w_q)
# errors.append(error)
print(f'n = {n}, p = {p}, error = {error}')

# %%
L_sp = jax.experimental.sparse.bcsr_fromdense(jnp.where(jnp.abs(L) > 1e-6, L, jnp.zeros_like(L)))

# %%
plt.plot(_x[:, 0], vmap(u_h)(_x), color='c', label='u_h', alpha = 0.5)
plt.plot(_x[:, 0], vmap(u)(_x), color='grey', label='u', alpha = 0.5)   
plt.legend()
plt.show()

# %%
evd = jnp.linalg.eigh(jnp.linalg.solve(M1, L))
# %%

###
# Lv = λ Mv
# -> M⁻¹ L v = λ v
###
plt.plot(evd[0], marker='s', label='λ')
plt.plot( (jnp.pi * jnp.arange(1,N1))**2, linestyle='--', color='grey', label='(π n)²')
plt.xlabel('n')
plt.legend()
plt.yscale('log')
# %%
for i in range(5):
    phi_i = get_u_h(evd[1][:, i], basis1)
    plt.plot(_x[:,0], vmap(phi_i)(_x), label=f'i ={i}, λ/ᴨ² = {evd[0][i] / jnp.pi**2}')
# %%

_end = 127
true_evs = jnp.sort(jnp.array([ (i**2) for i in range(1, n) ]))
# ax.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='s', label='λ/ᴨ²')
plt.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='v', label='λ/ᴨ²')
plt.plot(jnp.arange(1,_end + 1), true_evs[:_end], marker='*', label='λ/ᴨ²')
plt.yscale('log')   
# %%
