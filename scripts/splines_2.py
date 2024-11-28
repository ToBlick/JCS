#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from functools import partial
import numpy.testing as npt

from mhd_equilibria.bases import *
from mhd_equilibria.splines import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *

import matplotlib.pyplot as plt

n = 8
p = 3
sp = jit(get_spline(n, p, "clamped"))
# %%

n_dx = n-1
dx_sp = jit(get_spline(n_dx, p-1, "clamped"))

# %%

_x = jnp.linspace(0, 1, 1000)

for i in range(n):
    # plt.plot(_x, vmap(sp, (0, None))(_x, i), label='p = 3')
    plt.plot(_x, 0.1 * vmap(grad(sp), (0, None))(_x, i), label='p = 3')
    plt.plot(_x, vmap(dx_sp, (0, None))(_x, i), label='p = 2')
plt.show()
# %%
dx_sp_2 = grad(sp)

def f(x):
    return dx_sp_2(x, 3)
x_q_1d, w_q_1d = get_quadrature_spectral(41)(0, 1)
_M = get_mass_matrix_lazy(dx_sp, x_q_1d, w_q_1d, None)
M = assemble(_M, 
             jnp.arange(n_dx , dtype=jnp.int32), 
             jnp.arange(n_dx , dtype=jnp.int32))
print(jnp.linalg.cond(M))
# %%
proj = get_l2_projection(dx_sp, x_q_1d, w_q_1d, n_dx)
f_hat = jnp.linalg.solve(M, proj(f))
# %%
f_h = get_u_h(f_hat, dx_sp)
# %%
nx = 256
_x = jnp.linspace(0, 1, nx)    
plt.plot(_x, vmap(f)(_x), label='f')
plt.plot(_x, vmap(f_h)(_x), label='f_h')

npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), atol=1e-15)
# %%

# %%
