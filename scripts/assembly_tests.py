# %%
import jax
from jax import jit
import jax.numpy as jnp
import jax.experimental.sparse
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.vector_bases import get_vector_basis_fn

import matplotlib.pyplot as plt 

import time

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%
# Map to physical domain: unit cube to [-3,3]^3
def F(x):
    return 6*x - jnp.ones_like(x)*3

def F_inv(x):    
    return (x + jnp.ones_like(x)*3) / 6

# %%

ns = (6, 7, 3)
ps = (3, 3, 0)
types = ('clamped', 'periodic', 'fourier')
basis0, shape0, N0 = get_zero_form_basis( ns, ps, types)
basis1, shapes1, N1 = get_one_form_basis( ns, ps, types)

x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, 5 - 2 + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, 5 - 2 + 1), 15),
            get_quadrature_periodic(16)(0,1))

# %%
# Mass matrices
_M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
assemble_M0 = jit(lambda : assemble(_M0, jnp.arange(N0), jnp.arange(N0)))
sparse_assemble_M0 = jit(lambda: sparse_assemble_3d(_M0, ns, max(ps)))
vmap_assemble_M0 = jit(lambda: assemble_full_vmap(_M0, jnp.arange(N0), jnp.arange(N0)))
# %%
start = time.time()
M0a = assemble_M0()
M0a[0,0]
end = time.time()
print("Assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = assemble_M0()
    M0[0,0]
end = time.time()
print("Assemble: " , (end - start)/3)
# %%
start = time.time()
M0s = sparse_assemble_M0()
M0s[0,0]
end = time.time()
print("Sparse assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = sparse_assemble_M0()
    M0[0,0]
end = time.time()
print("Sparse assemble: " , (end - start)/3)
# %%
start = time.time()
M0v = vmap_assemble_M0()
M0v[0,0]
end = time.time()
print("Vmap assemble compilation: " , end - start)
start = time.time()
for _ in range(3):
    M0 = vmap_assemble_M0()
    M0[0,0]
end = time.time()
print("Vmap assemble: " , (end - start)/3)
# %%
print(jnp.sum((M0a - M0s)**2))
print(jnp.sum((M0v - M0s)**2))
print(jnp.sum((M0v - M0a)**2))
# %%
nx = 32
_r = jnp.linspace(0, 1, nx)
_θ = jnp.linspace(0, 1, nx)
_z = jnp.array([0.0])

x = jnp.array(jnp.meshgrid(_r, _θ, _z)) # shape 3, n_x, n_x, 1
x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)

# %%
_i = 80
plt.contourf(_r, _θ, vmap(basis0, (0, None))(x, _i).reshape(nx, nx))
plt.xlabel('r')
plt.ylabel('θ')
plt.colorbar()
plt.title('basis ' + str(_i))
plt.show()
# %%
_i = 80
_nrm = jnp.sqrt(jnp.sum(vmap(basis1, (0, None))(x, _i)**2, axis=1)).reshape(nx, nx)
plt.contourf(_r, _θ, _nrm)
plt.quiver(_r, _θ, vmap(basis1, (0, None))(x, _i)[:,0].reshape(nx, nx)/_nrm, vmap(basis1, (0, None))(x, _i)[:,1].reshape(nx, nx)/_nrm)
plt.xlabel('r')
plt.ylabel('θ')
plt.colorbar()
plt.title('basis ' + str(_i))
plt.show()

# %%
