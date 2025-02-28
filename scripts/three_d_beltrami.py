# %%
from jax import jit, config
from jax.experimental.sparse import bcsr_fromdense
from jax.experimental.sparse.linalg import spsolve
import jax.numpy as jnp
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.operators import div

import matplotlib.pyplot as plt 

### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%
def sparse_solve(A, b):
    return spsolve(A.data, A.indices, A.indptr, b, tol=1e-12)

# F maps the logical domain (unit cube) to the physical one
def F(x):
    return x
F_inv = F

# We are looking to solve  ∇ x B = νB


# Analytical solution
def B(x):
    return (jnp.true_divide(1,jnp.sqrt(2))*jnp.cos(jnp.true_divide(jnp.exp(x[0]+x[1]),jnp.sqrt(2))),jnp.true_divide(1,jnp.sqrt(2))*jnp.cos(jnp.true_divide(jnp.exp(x[0]+x[1]),jnp.sqrt(2))) ,  jnp.sin(jnp.true_divide(jnp.exp(x[0]+x[1]),jnp.sqrt(2))))



# Here, I'm thinking if there's a way to use our mappings to find the Beltrami field
# as in Camilla's thesis.


