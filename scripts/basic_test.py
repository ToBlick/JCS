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
# %%
def F(x):
    return x

n = 8
p = 3
ns = (n, 1, 1)
ns2 = (n+1, 1, 1)
ps = (p, 1, 1)

nx = 128
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.zeros(1)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx, 3)
        
### TODO: Projection and differentiation do not compute for periodic splines right now!
type = "periodic"
    
types = (type, 'fourier', 'fourier')
boundary = ('free', 'periodic', 'periodic')
basis0, shape0,  N0 = get_zero_form_basis( ns, ps, types, boundary)
basis1, shapes1, N1 = get_one_form_basis(  ns, ps, types, boundary)
basis2, shapes2, N2 = get_two_form_basis(  ns, ps, types, boundary)
basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

x_q, w_q = quadrature_grid(get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
                           get_quadrature_periodic(1)(0,1),
                           get_quadrature_periodic(1)(0,1))

def f(x):
    return jnp.sin(2 * jnp.pi * x[0])

proj = get_l2_projection(basis0, x_q, w_q, N0)
M00 = assemble(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
f_hat = jnp.linalg.solve(M00, proj(f))
f_h = get_u_h(f_hat, basis0)

M11 = assemble(get_mass_matrix_lazy_11(basis1, x_q, w_q, F), jnp.arange(N1), jnp.arange(N1))
D = assemble(get_gradient_matrix_lazy_01(basis0, basis1, x_q, w_q, F), jnp.arange(N0), jnp.arange(N1)).T
df_hat = jnp.linalg.solve(M11, D @ f_hat)
df_h = get_u_h_vec(df_hat, basis1)

# %%
plt.plot(_x1, vmap(df_h)(_x)[:, 0])
plt.plot(_x1, vmap(grad(f_h))(_x)[:, 0])
plt.plot(_x1, vmap(grad(f))(_x)[:, 0])

# %%
jnp.sqrt(jnp.sum((vmap(df_h)(_x) - vmap(grad(f_h))(_x))**2))

# %%
for i in range(N0):
    _basis = lambda x: basis0(x, i)
    plt.plot(_x1, vmap(_basis)(_x))

# %%
for i in range(N1):
    _basis = lambda x: basis1(x, i)
    plt.plot(_x1, vmap(_basis)(_x)[:, 0])
# %%
