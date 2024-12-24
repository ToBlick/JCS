# %%
from jax import jit, config, vmap
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

import matplotlib.pyplot as plt 

### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
print(xla_bridge.get_backend().platform)

# %%

# F maps the logical domain (unit cube) to the physical one
def F(x):
    return x
F_inv = F

n = 20
p = 3
ns = (n, n, 1)
ps = (p, p, 1)
types = ('clamped', 'clamped', 'fourier')
boundary = ('free', 'free', 'periodic')
basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)
basis1, shapes1, N1 = get_one_form_basis(ns, ps, types, boundary)
basis2, shapes2, N2 = get_two_form_basis(ns, ps, types, boundary)
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
def piecewise_assemble(f, ns, ms, split):
    n = len(ns)
    m = len(ms)
    _ns = jnp.split(ns, [(n * i)//split for i in range(1, split)])
    _ms = jnp.split(ms, [(m * i)//split for i in range(1, split)])
    # subarrays = [ [ assemble(f, _ns[i], _ms[j]) for j in range(3) ] for i in range(3) ]
    subarrays = []
    for i in range(split):
        row = []
        for j in range(split):
            row.append(assemble(f, _ns[i], _ms[j]))
        subarrays.append(row)
    return jnp.block(subarrays)

@jit
def divergence_matrix_lazy(i, j):
    def get_basis(k):
        return lambda x: basis2(x, k)
    return l2_product(lambda x: div(get_basis(i))(x), lambda x: basis3(x, j), x_q, w_q)
D = assemble(divergence_matrix_lazy, jnp.arange(N2), jnp.arange(N3)).T
D = bcsr_fromdense(D)


# %%
@jit
def mass_matrix_lazy_2(i, j):
    def get_basis(k):
        return lambda x: basis2(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)

M2 = assemble(mass_matrix_lazy_2, jnp.arange(N2), jnp.arange(N2))
# %%
M2 = bcsr_fromdense(M2)

# %%
def f(x):
    return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
def u(x):
    return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])
proj = get_l2_projection(basis3, x_q, w_q, N3)

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
Q = jnp.block([[M2.todense(), -D.todense().T], [D.todense(), jnp.zeros((N3, N3))]])
b = jnp.block([jnp.zeros(N2), proj(f)])

# %%
def sparse_solve(A, b):
    return spsolve(A.data, A.indices, A.indptr, b, tol=1e-12)

Q_sp = bcsr_fromdense(Q)
sigma_hat, u_hat = jnp.split(sparse_solve(Q_sp, b), [N2])


# %%
# D = D.todense()
# Q = jnp.array([sparse_solve(M2, D[i,:]) for i in range(N3)]).T
# L = bcsr_fromdense(D @ Q_sp)
# D = bcsr_fromdense(D)
# u_hat = sparse_solve(L, proj(f))

# %%
u_h = get_u_h(u_hat, basis3)
sigma_h = get_u_h_vec(sigma_hat, basis2)

def err(x):
    return jnp.sum((u_h(x) - u(x))**2)

error = jnp.sqrt(integral(err, x_q, w_q))
print(f'n = {n}, p = {p}, error = {error}')

# %%
plt.contourf(_x1, _x2, (vmap(u_h)(_x) - vmap(u)(_x)).reshape(nx, nx))
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
@jit
def mass_matrix_lazy_3(i, j):
    def get_basis(k):
        return lambda x: basis3(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)
# M3 = piecewise_assemble(mass_matrix_lazy_2, jnp.arange(N3), jnp.arange(N3), 4)
M3 = assemble(mass_matrix_lazy_2, jnp.arange(N3), jnp.arange(N3))
M3 = bcsr_fromdense(M3)

# def sparse_solve(A, b):
#     return jax.experimental.sparse.linalg.spsolve(A.data, A.indices, A.indptr, b)


# %%
L = L.todense()
M3 = M3.todense()
evs, evecs = jnp.linalg.eigh(jnp.linalg.solve(M3, L))
L = bcsr_fromdense(L)
M3 = bcsr_fromdense(M3)
# %%

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
L = L.todense()
M3 = M3.todense()
evs, evecs = generalized_eigh(L, M3)
L = bcsr_fromdense(L)
M3 = bcsr_fromdense(M3)
# %%

###
# Lv = λ Mv
# -> M⁻¹ L v = λ v
###
_end = 16
true_evs = jnp.sort(jnp.array([ i**2 + j**2 for i in range(1, n) for j in range(1, n)]))
fig, ax = plt.subplots()
ax.set_yticks((true_evs))
ax.set_xticks(jnp.arange(1,_end + 1))
ax.yaxis.grid(True, which='major')
ax.xaxis.grid(True, which='major')
ax.set_ylabel('λ/ᴨ²')
ax.legend()
# ax.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='s', label='λ/ᴨ²')
ax.plot(jnp.arange(1,_end + 1), evs[:_end] / (jnp.pi**2), marker='v', label='λ/ᴨ²')
ax.plot(jnp.arange(1,_end + 1), true_evs[:_end], marker='*', label='λ/ᴨ²')
# ax.set_yscale('log')
ax.set_xlabel('n')
# %%

# plt.plot(jnp.arange(1,_end + 1), evd[0][:_end] / (jnp.pi**2), marker='s', label='λ/ᴨ²')
# plt.plot(jnp.arange(1,_end + 1), evs[:_end] / (jnp.pi**2), marker='v', label='λ/ᴨ²')
# plt.plot(jnp.arange(1,_end + 1), true_evs[:_end], marker='*', label='λ/ᴨ²')
# plt.xlabel('n')
# plt.ylabel('λ/ᴨ²')
# %%
_i = 56
phi_i = get_u_h(evecs[:, _i], basis3)
plt.contourf(_x1, _x2, vmap(phi_i)(_x).reshape(nx, nx))
plt.colorbar()
plt.title(f'λ/ᴨ² = {evs[_i] / (jnp.pi**2)}')
# %%
