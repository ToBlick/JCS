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
import mhd_equilibria.forms as mforms
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import div
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.pullbacks import *

from functools import partial
# 
### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
print(xla_bridge.get_backend().platform)

# %%

# Had to termporarily add back in these definitions because forms wasn't importing; just seeing if new quadrature works

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
alpha = jnp.pi/2

@partial(jit, static_argnums=(1, 2))
def mismatch(alpha, n, p):
    # F maps the logical domain (unit cube) to the physical one
    def F(x):
        return jnp.array([ [ jnp.cos(alpha), jnp.sin(alpha), 0],
                        [-jnp.sin(alpha), jnp.cos(alpha), 0],
                        [0              , 0             , 1] ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
    def F_inv(x):
        return jnp.array([ [jnp.cos(alpha), -jnp.sin(alpha), 0],
                        [jnp.sin(alpha),  jnp.cos(alpha), 0],
                        [0             , 0              , 1]]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
        
    ns = (n, n, 1)
    ps = (p, p, 1)
    types = ('clamped', 'clamped', 'fourier')
    boundary = ('free', 'free', 'periodic')
    basis0, shape0, N0  = get_zero_form_basis( ns, ps, types, boundary)
    basis1, shapes1, N1 = get_one_form_basis(  ns, ps, types, boundary)
    basis2, shapes2, N2 = get_two_form_basis(  ns, ps, types, boundary)
    basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

    N2 = n * (n-1) * 2
    N3 = (n-1) * (n-1)
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

    # Assemble the mass matrix 
    D = assemble(get_divergence_matrix_lazy_23(basis2, basis3, (lambda x:x)(x_q), (lambda x:x)(w_q), F), jnp.arange(N2), jnp.arange(N3)).T
    M22 = assemble(get_mass_matrix_lazy_22(basis2, x_q, w_q, F), jnp.arange(N2), jnp.arange(N2))
    def f(x):
        return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
    def u(x):
        return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])
    proj3 = get_l2_projection(basis3, x_q, w_q, N3)

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
    L = D @ jnp.linalg.solve(M22, D.T)
    print(jnp.linalg.cond(L))
    u_hat = jnp.linalg.solve(L, proj3(f))
    u_h = pullback_3form(get_u_h(u_hat, basis3), F_inv)
    def err(x):
        return jnp.sum((u_h(x) + u(x))**2)
    error = jnp.sqrt(integral(err, x_q, w_q))
    print(f'n = {n}, p = {p}, error = {error}')
    return error

# %%

mismatch(alpha, 8, 2)
# %%
mismatch(alpha, 16, 2)

# %%
jax.grad(mismatch)(jnp.pi/6, 16, 2)
# %%
vals = jax.vmap(lambda alpha: jax.value_and_grad(mismatch)(alpha, 16, 2))(jnp.linspace(jnp.pi/2 - 0.01, jnp.pi/2 + 0.01, 32))
# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=jnp.linspace(0, jnp.pi, 64), y=vals[0], mode='lines', name='error'))
fig.add_trace(go.Scatter(x=jnp.linspace(0, jnp.pi, 64), y=vals[1], mode='markers', name='gradient'))
fig.show()
# %%
