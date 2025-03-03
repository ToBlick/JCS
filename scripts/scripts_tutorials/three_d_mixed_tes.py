# %%
from jax import jit, config
from functools import partial
import jax.numpy as jnp
from jax.experimental.sparse import bcsr_fromdense
from jax.experimental.sparse.linalg import spsolve
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import div
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *

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

def f(x):
    return 3 * (jnp.pi)**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])
def u(x):
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])

def get_error(n, p):
    ns = (n, n, n)
    ps = (p, p, p)
    types = ('clamped', 'clamped', 'clamped')
    boundary = ('free', 'free', 'free')
    basis0, shapes0, N0 = get_zero_form_basis(ns, ps, types, boundary)
    basis1, shapes1, N1 = get_one_form_basis(ns, ps, types, boundary)
    basis2, shapes2, N2 = get_two_form_basis(ns, ps, types, boundary)
    basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

    x_q, w_q = quadrature_grid(
        get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
        get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
        get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15))

    Dij = jit(get_divergence_matrix_lazy_23(basis2, basis3, x_q, w_q, F))
    D = assemble(Dij, jnp.arange(N2), jnp.arange(N3)).T
        
    M2ij = jit(get_mass_matrix_lazy_22(basis2, x_q, w_q, F))
    M2 = assemble(M2ij, jnp.arange(N2), jnp.arange(N2))
    
    proj = get_l2_projection(jit(basis3), x_q, w_q, N3)

    Q = jnp.block([[M2, -D.T], [D, jnp.zeros((N3, N3))]])
    Q = bcsr_fromdense(Q)
    b = jnp.block([jnp.zeros(N2), proj(f)])

    # sigma_hat, u_hat = jnp.split(jnp.linalg.solve(Q, b), [N2])
    sigma_hat, u_hat = jnp.split(sparse_solve(Q, b), [N2])
    
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
    ###

    u_h = get_u_h(u_hat, basis3)

    def err(x):
        return jnp.sum((u_h(x) - u(x))**2)

    error = jnp.sqrt(integral(err, x_q, w_q))
    # print(f'n = {n}, p = {p}, error = {error}')
    return error

# %%
_ns = [4, 6, 8, 10]
_ps = [1, 2, 3]

import time
errors = []
times = []
for n in _ns:
    for p in _ps:
        start = time.time()
        errors.append(get_error(n, p))
        end = time.time()
        times.append(end - start)
        print(f'n = {n}, p = {p}, time = {end - start}, error = {errors[-1]}')
# %%
### Convergence plot - the dashed lines are convergence with rate p

import plotly.graph_objects as go
arrerrors = jnp.array(errors).reshape((len(_ns), len(_ps)))
colors = ['red', 'blue', 'green', 'orange', 'purple']
fig = go.Figure()
for (i,p) in enumerate(_ps):
    fig.add_trace(go.Scatter
    (
        x=_ns,
        y=(arrerrors[:,i]),
        mode='lines+markers',
        name=f'p = {p}',
        marker=dict(symbol='square', color=colors[i]),
    ))
    fig.add_trace(go.Scatter
    (
        x=_ns[-2:],
        y=(arrerrors[-2,i]) * 1.3 * jnp.array([1, (_ns[-2]/_ns[-1])**(p)]),
        mode='lines',
        name=f'n^{-p}',
        line=dict(dash='dash', color=colors[i]),
        
    ))
fig.update_layout(
    xaxis_title='n',
    yaxis_title='error',
    yaxis_type='log',
    xaxis_type='log',
    showlegend=True
)
fig.show()
# %%
import plotly.graph_objects as go
arrtimes = jnp.array(times).reshape((len(_ns), len(_ps)))
fig = go.Figure()
for (i,p) in enumerate(_ps):
    fig.add_trace(go.Scatter
    (
        x=jnp.array(_ns)**3,
        y=(arrtimes[:,i]),
        mode='lines+markers',
        name=f'p = {p}',
        marker=dict(symbol='square', color=colors[i]),
    ))
fig.add_trace(go.Scatter
    (
        x=jnp.array(_ns)[-2:]**3,
        y=arrtimes[-2,-1] * 1.3 * (jnp.array(_ns)[-2:] / _ns[-2] )**(3 * 2.5),
        mode='lines',
        name=f'N^(2.5)',
        line=dict(dash='dash'),
        
    ))
fig.update_layout(
    xaxis_title='nx * ny * nz',
    yaxis_title='computation time [s]',
    yaxis_type='log',
    xaxis_type='log',
    showlegend=True
)
fig.show()# %%

# %%
