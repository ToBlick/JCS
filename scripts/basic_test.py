# %%
import jax.experimental
import jax.experimental.sparse
from matplotlib import pyplot as plt
from mhd_equilibria.splines import *
from mhd_equilibria.forms import *
from mhd_equilibria.bases import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.bases import *
from mhd_equilibria.vector_bases import *
from mhd_equilibria.operators import *
from mhd_equilibria.projections import *

import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import quadax as quad
import chex

import matplotlib.pyplot as plt

n = 6
p = 3
type = "periodic"

sp = jit(get_spline(n, p, type))
def f(x):
    return jnp.cos(2*jnp.pi*x)
x_q_1d, w_q_1d = get_quadrature_composite(jnp.linspace(0, 1, n - p + 1), 15)
_M0 = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
M0 = assemble(_M0, jnp.arange(n), jnp.arange(n))
# %%
plt.imshow(M0)
# %%
        
proj = get_l2_projection(sp, x_q_1d, w_q_1d, n)
f_hat = jnp.linalg.solve(M0, proj(f))
print(f_hat)
# %%

f_h = get_u_h(f_hat, sp)
nx = 256
_x = jnp.linspace(0, 1, nx)

# %%
plt.plot(_x, f(_x) - vmap(f_h)(_x))
# %%

import time

_M = jit(get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None))

# %%
t0 = time.time()
for _ in range(10):
    M = assemble(_M, jnp.arange(n), jnp.arange(n))
    M[1,1]
t1 = time.time()
print("Assemble: ", (t1 - t0)/10)
# %%
t0 = time.time()
for _ in range(100):
    M = assemble_full_vmap(_M, jnp.arange(n), jnp.arange(n))
    M[1,1]
t1 = time.time()
print("Assemble vmapped: ", (t1 - t0)/10)
# %%
