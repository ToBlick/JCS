# %%
import jax
from jax import jit
import jax.numpy as jnp
import jax.experimental.sparse
from mhd_equilibria.bases import *
from mhd_equilibria.forms import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import laplacian
from mhd_equilibria.vector_bases import get_vector_basis_fn

import matplotlib.pyplot as plt 
jax.config.update("jax_enable_x64", True)
import time

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%
# Map to physical domain: unit cube to [-3,3]^3
def F(x):
    return x
F_inv = F

errors = []
_ns = [8, 12, 16, 20, 32]
_ps = [1, 2, 3, 4, 5, 6]
for n in _ns:
    for p in _ps:
        ns = (n, 1, 1)
        ps = (p, 0, 0)
        types = ('clamped', 'fourier', 'fourier')
        boundary = ('dirichlet', 'periodic', 'periodic')
        
        basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)
        basis1, shapes1, N1 = get_one_form_basis(ns, ps, types, boundary)

        x_q, w_q = quadrature_grid(
                    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
                    get_quadrature_periodic(1)(0,1),
                    get_quadrature_periodic(1)(0,1))

        # nx = 512
        # _x1 = jnp.linspace(0, 1, nx)
        # _x2 = jnp.zeros(1)
        # _x3 = jnp.zeros(1)
        # _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
        # _x = _x.transpose(1, 2, 3, 0).reshape((nx)*1*1, 3)

        def stiffness_matrix_lazy(i, j):
            return l2_product(lambda x: grad(basis0)(x, i), lambda x: grad(basis0)(x, j), x_q, w_q)

        K = assemble_full_vmap(stiffness_matrix_lazy, jnp.arange(N0), jnp.arange(N0))
        # def mass_matrix_lazy(i, j):
        #     def get_basis(k):
        #         return lambda x: basis0(x, k)
        #     return l2_product(get_basis(i), get_basis(j), x_q, w_q)
        # M = assemble_full_vmap(mass_matrix_lazy, jnp.arange(N0), jnp.arange(N0))

        def f(x):
            return (4 * jnp.pi)**2 * jnp.sin(4 * jnp.pi * x[0])
        def u(x):
            return jnp.sin(4 * jnp.pi * x[0])

        proj = get_l2_projection(basis0, x_q, w_q, N0)
        # f_hat = jnp.linalg.solve(M, proj(f))
        # f_h = get_u_h(f_hat, basis0)

        u_hat = jnp.linalg.solve(K, proj(f))
        u_h = get_u_h(u_hat, basis0)

        def err(x):
            return jnp.sum((u_h(x) - u(x))**2)

        error = integral(err, x_q, w_q)
        errors.append(error)
        print(f'n = {n}, p = {p}, error = {error}')

# %%
arrerrors = jnp.array(errors).reshape((len(_ns), len(_ps)))
for (i,p) in enumerate(_ps):
    plt.plot(_ns, jnp.sqrt(arrerrors[:,i]), label=f'p = {p}', marker='s')
    plt.plot(_ns[-2:], jnp.sqrt(arrerrors[-2,i]) * 2 * jnp.array([1, (_ns[-2]/_ns[-1])**(2*p)]), linestyle='--', color='grey')
plt.xlabel('n')
plt.ylabel('error')
plt.yscale('log')
plt.xscale('log')
plt.grid('minor')
plt.legend()


# %%
