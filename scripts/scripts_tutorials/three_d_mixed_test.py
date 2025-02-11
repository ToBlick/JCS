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

def f(x):
    return 3 * (jnp.pi)**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])
def u(x):
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])

errors = []
_ns = [6, 8, 10]
_ps = [1, 2, 3, 4]
for n in _ns:
    for p in _ps:
        ns = (n, n, n)
        ps = (p, p, p)
        types = ('clamped', 'clamped', 'clamped')
        boundary = ('free', 'free', 'free')
        basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)
        basis1, shapes1, N1 = get_one_form_basis(ns, ps, types, boundary)
        basis2, shapes2, N2 = get_two_form_basis(ns, ps, types, boundary)
        basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

        x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15))
        
        @jit
        def divergence_matrix_lazy(i, j):
            def get_basis(k):
                return lambda x: basis2(x, k)
            return l2_product(lambda x: div(get_basis(i))(x), lambda x: basis3(x, j), x_q, w_q)
        D = assemble(divergence_matrix_lazy, jnp.arange(N2), jnp.arange(N3)).T

        @jit
        def mass_matrix_lazy_2(i, j):
            def get_basis(k):
                return lambda x: basis2(x, k)
            return l2_product(get_basis(i), get_basis(j), x_q, w_q)

        M2 = assemble(mass_matrix_lazy_2, jnp.arange(N2), jnp.arange(N2))
        proj = get_l2_projection(basis3, x_q, w_q, N3)

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
        #
        ###

        u_h = get_u_h(u_hat, basis3)

        def err(x):
            return jnp.sum((u_h(x) - u(x))**2)

        error = jnp.sqrt(integral(err, x_q, w_q))
        print(f'n = {n}, p = {p}, error = {error}')
        errors.append(error)
# %%
### Convergence plot - the dashed lines are convergence with rate p
arrerrors = jnp.array(errors).reshape((len(_ns), len(_ps)))
for (i,p) in enumerate(_ps):
    plt.plot(_ns, jnp.sqrt(arrerrors[:,i]), label=f'p = {p}', marker='s')
    plt.plot(_ns[-2:], jnp.sqrt(arrerrors[-2,i]) * 1.1 * jnp.array([1, (_ns[-2]/_ns[-1])**(p)]), linestyle='--', color='grey')
plt.xlabel('n')
plt.ylabel('error')
plt.yscale('log')
plt.xscale('log')
plt.grid('minor')
plt.legend()
# %%
