# %%
from jax import jit, config, grad
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
# Impose on the unit square Omega =  \{(x,y):  0<= x,y <=1\}, with homogeneous 
# Dirichlet boundary conditions phi = 0 on delOmega. We wish to handle a nonlinear f, so we consider 
#f(phi) = -2y^3+6x(1-x)y


def f(x):
    return -2*(x[1]**3) + 6*x[0]*(1-x[0])*x[1]

def u(x):
    return x[0]*(1-x[0])*(x[1]**3)


### Loop over _ns (number of basis functions) and _ps (degree of basis functions) and store L2 errors
errors = []
_ns = [8, 12, 16, 20]
_ps = [1, 2, 3, 4, 5, 6]
for n in _ns:
    for p in _ps:
        ### Everything is written in 3D, so to solve a 2D problem we use a constant basis in z
        ns = (n, n, 1)
        ### Same degree in x,y constant in z
        ps = (p, p, 0)
        ### basis_fucntion(x,y,z) = spline(x) * spline(y) * fourier(z)
        ### z direction is a single Fourier mode (constant)
        types = ('clamped', 'clamped', 'fourier')
        ### Boundary conditions: Only the 'dirichlet' boundary condition actually does something: It removes the first and last clamped spline from the basis.
        boundary = ('dirichlet', 'dirichlet', 'periodic')
        ### basis0 is the basis for a zero form (scalar field) - it is a function that takes (x, i) and returns the i-th basis function evaluated at x (this is a scalar)
        ### shape0 is an array with the number of x,y,z basis functions - in this case (n, n, 1)
        ### N0 is the total number of basis functions: n×n×1
        basis0, shape0, N0 = get_zero_form_basis(ns, ps, types, boundary)

        ### Quadrature grid: x_q is the quadrature points and w_q the weights
        ### There are a few options: 
        ### get_quadrature_spectral is Gauss quadrature
        ### get_quadrature_composite is Gauss quadrature on a composite grid (between spline knots)
        ### get_quadrature_periodic is trapezoidal, which converges exponentially for periodic functions
        ### The first two only work for the degrees supported by quadax, i.e. {15, 21, 31, 41, 51, 61}.
        x_q, w_q = quadrature_grid(
                    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
                    get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
                    get_quadrature_periodic(1)(0,1))

        ### Lazy function for the stiffness matrix, returns its i,j-th element
        ### Note: This is not mixed FEM, just a standard stiffness matrix with natural homogeneous Neumann BCs
        def stiffness_matrix_lazy(i, j):
            return l2_product(lambda x: grad(basis0)(x, i), lambda x: grad(basis0)(x, j), x_q, w_q)

        ### Assemble the stiffness matrix: full_vmap means that all elements are computed at once
        K = assemble_full_vmap(stiffness_matrix_lazy, jnp.arange(N0), jnp.arange(N0))
        # K = assemble(stiffness_matrix_lazy, jnp.arange(N0), jnp.arange(N0))

        ### function that given a function f returns a vector v: v_i = ∫ f(x) * basis_i(x) dx ∀i
        proj = get_l2_projection(basis0, x_q, w_q, N0)

        ### Solve the linear system K u_hat = v
        u_hat = jnp.linalg.solve(K, proj(f))
        ### Given the DoFs u_hat, return the function u_h(x) = ∑ u_hat_i basis_i(x)
        u_h = get_u_h(u_hat, basis0)

        def err(x):
            return jnp.sum((u_h(x) - u(x))**2)
        error = jnp.sqrt(integral(err, x_q, w_q))
        errors.append(error)
        print(f'n = {n}, p = {p}, error = {error}')

# %%

### Convergence plot - the dashed lines are convergence with rate p
arrerrors = jnp.array(errors).reshape((len(_ns), len(_ps)))
plt.figure()
plt.title('Convergence plots')
for (i,p) in enumerate(_ps):
    plt.plot(_ns, jnp.sqrt(arrerrors[:,i]), label=f'p = {p}', marker='s')
    plt.plot(_ns[-2:], jnp.sqrt(arrerrors[-2,i]) * 1.3 * jnp.array([1, (_ns[-2]/_ns[-1])**(p)]), linestyle='--', color='grey')
plt.xlabel('n')
plt.ylabel('error')
plt.yscale('log')
plt.xscale('log')
plt.grid('minor')
plt.legend()

# %%
