# %%
from jax import jit, config, vmap, grad, jacfwd, jacrev
from jax.scipy.linalg import eigh
from jax.experimental.sparse import bcsr_fromdense
from jax.experimental.sparse.linalg import spsolve
import jax.numpy as jnp
from jax.lib import xla_bridge
import jax.experimental.sparse
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import div
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.pullbacks import *

import matplotlib.pyplot as plt 

### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
print(xla_bridge.get_backend().platform)

def get_mass_matrix_lazy_33(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def f(k):
        return lambda x: basis_fn(x, k)
    def g(k):
        return lambda x: basis_fn(x, k) / jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(f(i), g(j), x_q, w_q)
    return M_ij
def get_divergence_matrix_lazy_23(basis_fn2, basis_fn3, x_q, w_q, F):
    DF = jacfwd(F)
    def f(k):
        phi = lambda x: basis_fn2(x, k)
        return div(phi)
    def g(k):
        return lambda x: basis_fn3(x, k) / jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(f(i), g(j), x_q, w_q)
    return M_ij

def get_mass_matrix_lazy_22(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def B(k):
        return lambda x: DF(x) @ basis_fn(x, k)
    def S(k):
        return lambda x: DF(x) @ basis_fn(x, k) / jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(B(i), S(j), x_q, w_q)
    return M_ij






# %%
alpha = jnp.pi/2
# F maps the logical domain (unit cube) to the physical one
def F(x):
    return jnp.array([ [ jnp.cos(alpha), jnp.sin(alpha), 0],
                       [-jnp.sin(alpha), jnp.cos(alpha), 0],
                       [0              , 0             , 1] ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
def F_inv(x):
    return jnp.array([ [jnp.cos(alpha), -jnp.sin(alpha), 0],
                       [jnp.sin(alpha),  jnp.cos(alpha), 0],
                       [0             , 0              , 1]]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2

n = 20
p = 3
ns = (n, n, 1)
ps = (p, p, 1)
types = ('clamped', 'clamped', 'fourier')
boundary = ('free', 'free', 'periodic')
basis0, shape0, N0  = get_zero_form_basis( ns, ps, types, boundary)
basis1, shapes1, N1 = get_one_form_basis(  ns, ps, types, boundary)
basis2, shapes2, N2 = get_two_form_basis(  ns, ps, types, boundary)
basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

x_q, w_q = quadrature_grid(
    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
    get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
    get_quadrature_periodic(1)(0,1))

nx = 32
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)

# %%
D = assemble(get_divergence_matrix_lazy_23(basis2, basis3, (lambda x:x)(x_q), (lambda x:x)(w_q), F), jnp.arange(N2), jnp.arange(N3)).T
# %%
M22 = assemble(get_mass_matrix_lazy_22(basis2, (lambda x:x)(x_q), (lambda x:x)(w_q), F), jnp.arange(N2), jnp.arange(N2))

# %%
def f(x):
    return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
def u(x):
    return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])
proj = get_l2_projection(basis3, (lambda x:x)(x_q), (lambda x:x)(w_q), N3)

###
# D σ = p(f) in L2
# Mσ = D.T u in Hdiv
# -> σ = M⁻¹ D.T u
# -> Dσ = D M⁻¹ D.T u = p(f)
# -> (D M⁻¹ D.T) u = p(f) 
# or
#
# | M  -D.T | | σ | = |  0   |
# | D   0   | | u | = | p(f) |
#
###
Q = jnp.block([[M22, -D.T], 
               [D, jnp.zeros((N3, N3))]])
b = jnp.block([jnp.zeros(N2), proj(f)])

# %%
# def sparse_solve(A, b):
#     return spsolve(A.data, A.indices, A.indptr, b, tol=1e-12)
# Q_sp = bcsr_fromdense(Q)
# sigma_hat, u_hat = jnp.split(sparse_solve(Q_sp, b), [N2])

sigma_hat, u_hat = jnp.split(jnp.linalg.solve(Q, b), [N2])
# %%
u_h = pullback_3form(get_u_h(u_hat, basis3), F_inv)
sigma_h = get_u_h_vec(sigma_hat, basis2)

def err(x):
    return jnp.sum((u_h(x) + u(x))**2)

error = jnp.sqrt(integral(err, x_q, w_q))
print(f'n = {n}, p = {p}, error = {error}')

# %%
plt.contourf(_x1, _x2, (vmap(u_h)(_x)).reshape(nx, nx))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

# %%
_basis = lambda x: basis2(x, 66)
plt.quiver(_x1, _x2, 
          (vmap(sigma_h)(_x)[:,0]).reshape(nx, nx), 
          (vmap(sigma_h)(_x)[:,1]).reshape(nx, nx),
          scale=300)
plt.xlabel('x')
plt.ylabel('y')

# %%
M33 = assemble(get_mass_matrix_lazy_33(basis3, x_q, w_q, F), jnp.arange(N3), jnp.arange(N3))

# def sparse_solve(A, b):
#     return jax.experimental.sparse.linalg.spsolve(A.data, A.indices, A.indptr, b)

###
# Lv = λ Mv
# Q Q.T = M
# -> Q⁻¹ L v = λ Q.T v
# -> v.T L.T Q⁻¹.T = λ v.T Q
# -> Q⁻¹.T v.T L.T Q⁻¹.T = λ Q⁻¹.T v.T Q
# -> (v Q⁻¹).T L.T Q⁻¹.T = λ v
# -> Q⁻¹ L Q⁻¹.T v = λ v
###
def generalized_eigh(A, B):
    Q = jnp.linalg.cholesky(B)
    # Q_inv = jnp.linalg.inv(Q)
    # C = Q_inv @ A @ Q_inv.T
    ### Q B.T = A.T -> B Q.T = A
    ### Q C = B
    C = jnp.linalg.solve(Q, jnp.linalg.solve(Q, A.T).T)
    eigenvalues, eigenvectors_transformed = eigh(C)
    eigenvectors_original = jnp.linalg.solve(Q.T, eigenvectors_transformed)
    return eigenvalues, eigenvectors_original
# %%
L = D @ jnp.linalg.solve(M22, D.T)
evs, evecs = generalized_eigh(L, M33)
# %%

###
# Lv = λ Mv
# -> M⁻¹ L v = λ v
###
_end = 25
true_evs = jnp.sort(jnp.array([ i**2 + j**2 for i in range(1, n) for j in range(1, n)]))
fig, ax = plt.subplots()
ax.set_yticks((true_evs[:]))
ax.set_xticks(jnp.arange(1,_end + 1)[::2])
ax.yaxis.grid(True, which='both')
ax.xaxis.grid(True, which='both')
ax.set_ylabel('λ/ᴨ²')
ax.legend()
# ax.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='s', label='λ/ᴨ²')
ax.plot(jnp.arange(1,_end + 1), evs[:_end] / (jnp.pi**2), marker='v', label='λ/ᴨ²')
ax.plot(jnp.arange(1,_end + 1), true_evs[:_end], marker='*', label='λ/ᴨ²', linestyle='')
# ax.set_yscale('log')
ax.set_xlabel('n')
# %%
# plt.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='s', label='λ/ᴨ²')
# plt.plot(jnp.arange(1,_end + 1), evs[:_end] / (jnp.pi**2), marker='v', label='λ/ᴨ²')
# plt.plot(jnp.arange(1,_end + 1), true_evs[:_end], marker='*', label='λ/ᴨ²')
# plt.xlabel('n')
# plt.ylabel('λ/ᴨ²')
# %%
_i = 5
phi_i = get_u_h(evecs[:, _i], basis3)
plt.contourf(_x1, _x2, vmap(phi_i)(_x).reshape(nx, nx))
plt.colorbar()
plt.title(f'λ/ᴨ² = {evs[_i] / (jnp.pi**2)}')
# %%
plt.plot(jnp.abs(evs / (jnp.pi**2) - true_evs)/true_evs)
plt.yscale('log')
# %%
