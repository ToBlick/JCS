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
nr, nχ, nζ = 4, 4, 4
pr, pχ, pζ = 3, 3, 1
basis_r = get_spline(nr, pr, 'clamped')
basis_χ = get_spline(nχ, pχ, 'periodic')
basis_ζ = get_trig_fn(nζ, 0, 1)
# %%
import timeit, statistics
nx = 32
_x1 = jnp.linspace(1e-6, 1, nx)
_x2 = jnp.linspace(1e-6, 1, nx)
_x3 = jnp.linspace(1e-6, 1, nx)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*nx, 3)
result = vmap(vmap(basis_r, (0, None)),(None, 0))(_x, jnp.arange(nr))

# %%
n_rep = 100
durations = timeit.Timer('vmap(vmap(basis_r, (0, None)),(None, 0))(_x1, jnp.arange(nr))', globals=globals()).repeat(repeat=n_rep, number=1)
print('clamped: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx
# %%
durations = timeit.Timer('vmap(vmap(basis_χ, (0, None)),(None, 0))(_x1, jnp.arange(nχ))', globals=globals()).repeat(repeat=n_rep, number=1)
print('periodic: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nχ/nx
# %%
durations = timeit.Timer('vmap(vmap(basis_ζ, (0, None)),(None, 0))(_x1, jnp.arange(nζ))', globals=globals()).repeat(repeat=n_rep, number=1)
print('fourier: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nζ/nx
# %%
jit_basis_r = jit(get_spline(nr, pr, 'clamped'))
jit_basis_χ = jit(get_spline(nχ, pχ, 'periodic'))
jit_basis_ζ = jit(get_trig_fn(nζ, 0, 1))
jit_basis_r(0, 0), jit_basis_ζ(0, 0), jit_basis_ζ(0, 0)
# %%
durations = timeit.Timer('vmap(vmap(jit_basis_r, (0, None)),(None, 0))(_x1, jnp.arange(nr))', globals=globals()).repeat(repeat=n_rep, number=1)
print('clamped, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx
# %%
durations = timeit.Timer('vmap(vmap(jit_basis_χ, (0, None)),(None, 0))(_x1, jnp.arange(nχ))', globals=globals()).repeat(repeat=n_rep, number=1)
print('periodic, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nχ/nx
# %%
durations = timeit.Timer('vmap(vmap(jit_basis_ζ, (0, None)),(None, 0))(_x1, jnp.arange(nζ))', globals=globals()).repeat(repeat=n_rep, number=1)
print('fourier, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nζ/nx
# %%

basis0, shape0, N0 = get_zero_form_basis((nr, nχ, nζ), (pr, pχ, pζ), ('clamped', 'periodic', 'fourier'), ('clamped', 'periodic', 'periodic'))

durations = timeit.Timer('vmap(vmap(basis0, (0, None)),(None, 0))(_x, jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx**3
# %%
jit_basis0 = jit(basis0)
jit_basis0(jnp.zeros(3), 0)
durations = timeit.Timer('vmap(vmap(jit_basis0, (0, None)),(None, 0))(_x, jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx**3
# %%
basis1, shape1, N1 = get_one_form_basis((nr, nχ, nζ), (pr, pχ, pζ), ('clamped', 'periodic', 'fourier'), ('clamped', 'periodic', 'periodic'))

# %%
durations = timeit.Timer('vmap(vmap(basis1, (0, None)),(None, 0))(_x, jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr
# %%
jit_basis1 = jit(basis1)
jit_basis1(jnp.zeros(3), 0)
durations = timeit.Timer('vmap(vmap(jit_basis1, (0, None)),(None, 0))(_x, jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr
# %%
