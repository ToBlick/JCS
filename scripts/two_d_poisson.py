# %%
from jax import jit, config
import jax.numpy as jnp
from jax.experimental.sparse import bcsr_fromdense
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import *
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *

import matplotlib.pyplot as plt 
config.update("jax_enable_x64", True)

# Poisson's equation via vorticity: minus Delta phi = w (vortiicity) = f(phi)
# Impose on the unit square Omega = (0,1) x (0,1), with homogeneous 
# Dirichlet boundary conditions phi = 0 on delOmega. From Camilla thesis, can
# reduce to linear eigenvalue problem,  minus Delta phi = lambda phi
# Exact solution


def lam(n,m):
    return jnp.power(jnp.pi, 2)*(jnp.power(m, 2) + jnp.power(n, 2))
# test
# print(lam(1,1))


#We wish to simulate voriticy and phi, using the relation w = lambda phi 


Omega = ((0, 1), (0, 1))

# F maps the logical domain (unit square) to the physical one

def F(x):
    return x
F_inv = F

n = 10
p = 2
ns = (n, n, 1)
ps = (p, p, 1)


types = ('clamped', 'clamped', 'fourier')
boundary = ('free', 'free', 'periodic')
basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)
basis1, shapes1, N1 = get_one_form_basis(ns, ps, types, boundary)
basis2, shapes2, N2 = get_two_form_basis(ns, ps, types, boundary)
basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

# points and weights
x_q, w_q = quadrature_grid(
    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
    get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
    get_quadrature_periodic(1)(0,1))



#Set up unifrom grid
nx = 22
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)


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
# Taking normalization constant to be 1.
def u(x):
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])






















# %%