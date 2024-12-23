#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, config
config.update("jax_enable_x64", True)
from functools import partial
import numpy.testing as npt

from jax.experimental.sparse import bcsr_fromdense
from jax.experimental.sparse.linalg import spsolve

from mhd_equilibria.bases import *
from mhd_equilibria.splines import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *

n = 8
p = 3
sp = jit(get_spline(n, p, "clamped"))
# %%

n_dx = n-1
dx_sp = jit(get_spline(n_dx, p-1, "clamped"))

# %%

_x = jnp.linspace(0, 1, 1000)

plt.figure()
for i in range(n):
    # plt.plot(_x, vmap(sp, (0, None))(_x, i), label='p = 3')
    # plt.plot(_x, 0.1 * vmap(grad(sp), (0, None))(_x, i), label='p = 3')
    plt.plot(_x, vmap(dx_sp, (0, None))(_x, i), label='p = 2')
plt.xlabel('x')
# %%
dx_sp_2 = grad(sp)

def sparse_assemble_row(i, _M, n2, p):
    M = jnp.zeros(n2)
    range = jnp.arange(-p, p + 1) + i
    row = vmap(_M, (0, None))(range, i)
    M = M.at[range].set(row)
    return M

def sparse_assemble_row_3d(I, _M, shape, p):
    i, j, k = jnp.unravel_index(I, shape)
    N = shape[0] * shape[1] * shape[2]
    M = jnp.zeros(N)
    range_i = jnp.clip(jnp.arange(-p, p + 1) + i, 0, shape[0] - 1)
    range_j = jnp.clip(jnp.arange(-p, p + 1) + j, 0, shape[1] - 1)
    range_k = jnp.clip(jnp.arange(-p, p + 1) + k, 0, shape[2] - 1)
    # Use jax.numpy.meshgrid to generate all combinations
    grid = jnp.meshgrid(range_i, range_j, range_k, indexing="ij")
    combinations = jnp.stack(grid, axis=-1).reshape(-1, len(grid))
    indices = vmap(jnp.ravel_multi_index, (0, None, None))(combinations, shape, 'clip')
    row = vmap(_M, (0, None))(indices, I)
    M = M.at[indices].set(row)
    return M

@partial(jit, static_argnums=(0,1,2,3))
def sparse_assemble(_M, n1, n2, p):
    return vmap(sparse_assemble_row, (0, None, None, None))(jnp.arange(n1), _M, n2, p)

@partial(jit, static_argnums=(0,1,2))
def sparse_assemble_3d(_M, shape, p):
    N = shape[0] * shape[1] * shape[2]
    return vmap(sparse_assemble_row_3d, (0, None, None, None))(jnp.arange(N), _M, shape, p)

def f(x):
    return jnp.cos(2 * x * 2 * jnp.pi) * jnp.exp(-10 * (x - 0.5)**2)
x_q_1d, w_q_1d = get_quadrature_composite(jnp.linspace(0, 1, n - p + 1), 61)
_M0 = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
M0 = assemble(_M0, jnp.arange(n), jnp.arange(n))
_M1 = get_mass_matrix_lazy(dx_sp, x_q_1d, w_q_1d, None)
M1 = assemble(_M1, jnp.arange(n_dx), jnp.arange(n_dx))

# %%
import time
start = time.time()
for i in range(10):
    M02 = sparse_assemble(_M0, n, n, p)
print(M02[0,0])
end = time.time()
print(end - start)

start = time.time()
for i in range(10):
    M0 = assemble(_M0, jnp.arange(n), jnp.arange(n))
print(M0[0,0])
end = time.time()
print(end - start)

# %%
import time
start = time.time()
for i in range(10):
    M03 = sparse_assemble_3d(_M0, (n,n,n), p)
print(M03[0,0])
end = time.time()
print(end - start)

# %%
print(jnp.linalg.norm(M0 - M02))
# print(jnp.linalg.norm(M03 - M02))

# %%
plt.figure()
plt.scatter(jnp.arange(n**3), M03[0,:])
# %%

proj0 = get_l2_projection(sp, x_q_1d, w_q_1d, n)
f_hat = jnp.linalg.solve(M0, proj0(f))
f_h = get_u_h(f_hat, sp)
# %%
proj1 = get_l2_projection(dx_sp, x_q_1d, w_q_1d, n_dx)
gradf_hat = jnp.linalg.solve(M1, proj1(grad(f_h)))
gradf_h = get_u_h(gradf_hat, dx_sp)

gradf_hat2 = jnp.linalg.solve(M0, proj0(grad(f)))
gradf_h2 = get_u_h(gradf_hat2, sp)
# %%
nx = 256
_x = jnp.linspace(0, 1, nx)    
plt.figure()
# plt.plot(_x, vmap(f)(_x), label='f')
plt.plot(_x, vmap(grad(f))(_x), label='∇f')
# plt.plot(_x, vmap(f_h)(_x), label='f_h')
plt.plot(_x, vmap(grad(f_h))(_x), label='∇fₕ')
plt.plot(_x, vmap(gradf_h2)(_x), label='(∇f)ₕ: n = 16, p = 3')
plt.plot(_x, vmap(gradf_h)(_x), label='(∇f)ₕ: n = 15, p = 2')
plt.xlabel('x')
plt.legend(fontsize=14)

# %%
plt.figure()
plt.plot(_x, jnp.abs(vmap(grad(f))(_x) - vmap(gradf_h2)(_x)), label='∇f - (∇f)ₕ: n = 16, p = 3')
plt.plot(_x, jnp.abs(vmap(grad(f))(_x) - vmap(gradf_h)(_x) ), label='∇f - (∇f)ₕ: n = 15, p = 2')
plt.plot(_x, jnp.abs(vmap(grad(f_h))(_x) - vmap(gradf_h2)(_x)), label='∇fₕ - (∇f)ₕ: n = 16, p = 3')
plt.plot(_x, jnp.abs(vmap(grad(f_h))(_x) - vmap(gradf_h)(_x) ), label='∇fₕ - (∇f)ₕ: n = 15, p = 2')
plt.yscale('log')
plt.xlabel('x')
plt.legend(fontsize=14)
plt.show()

# %%
print(jnp.linalg.norm(vmap(grad(f_h))(_x) - vmap(gradf_h)(_x)))
print(jnp.linalg.norm(vmap(grad(f))(_x) - vmap(gradf_h)(_x)))
print(jnp.linalg.norm(vmap(grad(f))(_x) - vmap(gradf_h2)(_x)))

# npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), atol=1e-15)
# %%

M0 = bcsr_fromdense(M0)
f_hat = spsolve(
            M0.data, M0.indices, M0.indptr, proj0(f))

# %%
