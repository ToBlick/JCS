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
from mhd_equilibria.operators import div

import matplotlib.pyplot as plt 

### Enable double precision
jax.config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
from jax.lib import xla_bridge
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
D = jax.experimental.sparse.bcsr_fromdense(D)


# %%
@jit
def mass_matrix_lazy_2(i, j):
    def get_basis(k):
        return lambda x: basis2(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)

M2 = assemble(mass_matrix_lazy_2, jnp.arange(N2), jnp.arange(N2))
# %%
M2 = jax.experimental.sparse.bcsr_fromdense(M2)

# %%
def f(x):
    return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
def u(x):
    return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])

###
# D σ = p(f) in L2
# Mσ = D.T u in Hdiv
# -> σ = M⁻¹ D.T u
# -> Dσ = D M⁻¹ D.T u = p(f)
# -> (D M⁻¹ D.T) u = p(f) 
###

# %%
def sparse_solve(A, b):
    return jax.experimental.sparse.linalg.spsolve(A.data, A.indices, A.indptr, b, tol=1e-12)

D = D.todense()
Q = jnp.array([sparse_solve(M2, D[i,:]) for i in range(N3)]).T
L = jax.experimental.sparse.bcsr_fromdense(D @ Q)
D = jax.experimental.sparse.bcsr_fromdense(D)

# %%
proj = get_l2_projection(basis3, x_q, w_q, N3)
# f_hat = jnp.linalg.solve(M, proj(f))
# f_h = get_u_h(f_hat, basis0)
u_hat = sparse_solve(L, proj(f))
u_h = get_u_h(u_hat, basis3)

def err(x):
    return jnp.sum((u_h(x) - u(x))**2)

error = jnp.sqrt(integral(err, x_q, w_q))
print(f'n = {n}, p = {p}, error = {error}')

# %%
# plt.contourf(_x1, _x2, (vmap(u_h)(_x) - vmap(u)(_x)).reshape(nx, nx))
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')

# %%
@jit
def mass_matrix_lazy_3(i, j):
    def get_basis(k):
        return lambda x: basis3(x, k)
    return l2_product(get_basis(i), get_basis(j), x_q, w_q)
# M3 = piecewise_assemble(mass_matrix_lazy_2, jnp.arange(N3), jnp.arange(N3), 4)
M3 = assemble(mass_matrix_lazy_2, jnp.arange(N3), jnp.arange(N3))
M3 = jax.experimental.sparse.bcsr_fromdense(M3)

# def sparse_solve(A, b):
#     return jax.experimental.sparse.linalg.spsolve(A.data, A.indices, A.indptr, b)


# %%
L = L.todense()
M3 = M3.todense()
evs, evecs = jnp.linalg.eigh(jnp.linalg.solve(M3, L))
L = jax.experimental.sparse.bcsr_fromdense(L)
M3 = jax.experimental.sparse.bcsr_fromdense(M3)
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
    eigenvalues, eigenvectors_transformed = jax.scipy.linalg.eigh(C)
    eigenvectors_original = jnp.linalg.solve(Q.T, eigenvectors_transformed)
    return eigenvalues, eigenvectors_original
# %%
L = L.todense()
M3 = M3.todense()
evs, evecs = generalized_eigh(L, M3)
L = jax.experimental.sparse.bcsr_fromdense(L)
M3 = jax.experimental.sparse.bcsr_fromdense(M3)
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
