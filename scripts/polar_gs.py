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
from mhd_equilibria.quadratures import *
from mhd_equilibria.splines import *
from mhd_equilibria.operators import div
from mhd_equilibria.vector_bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.pullbacks import *
from mhd_equilibria.mappings import *

import matplotlib.pyplot as plt 

### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
print(xla_bridge.get_backend().platform)
# %%

### Get polar mapping

#TODO: Fix the R0 offset
a = 1
R0 = 2.0
Y0 = 0

def R(x):
    r, χ, z = x
    return R0 + a * r * jnp.cos(2 * jnp.pi * χ)
def Y(x):
    r, χ, z = x
    return Y0 + a * r * jnp.sin(2 * jnp.pi * χ)

ns = (4, 8, 1)
ps = (3, 3, 1)

### Project this mapping to the zero form basis
types = ('clamped', 'periodic', 'fourier')
boundary = ('free', 'periodic', 'periodic')
basis, shape, N = get_zero_form_basis(ns, ps, types, boundary)
basis = jit(basis)

# quadrature grid and projection
x_q, w_q = quadrature_grid(
    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
    get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
    get_quadrature_periodic(1)(0,1))
proj = get_l2_projection(basis, x_q, w_q, N)

# %%
M = assemble_full_vmap(get_mass_matrix_lazy_00(basis, x_q, w_q, lambda x: x), jnp.arange(N), jnp.arange(N))

# %%
# isogeometric mapping

R_hat = jnp.linalg.solve(M, proj(R))
R_h = get_u_h(R_hat, basis)
Y_hat = jnp.linalg.solve(M, proj(Y))
Y_h = get_u_h(Y_hat, basis)

cR = R_hat.reshape(ns[0], ns[1])
cY = Y_hat.reshape(ns[0], ns[1])

# %%

ΔR = cR[1,:] - R0
ΔY = cY[1,:] - Y0
# %%
τ = max([jnp.max(-2 * ΔR), jnp.max(ΔR - jnp.sqrt(3) * ΔY), jnp.max(ΔR + jnp.sqrt(3) * ΔY)])

 # %%
plt.scatter(R_hat, Y_hat, s=5)
plt.scatter([τ + R0, R0 - τ/2, R0 - τ/2], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2], s=10, c='k')
plt.plot([τ + R0, R0 - τ/2, R0 - τ/2, τ + R0], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2, 0], 'k:')
# %%
ξ00 = jnp.ones(ns[1]) / 3
ξ01 = 1/3 + 2/(3*τ) * ΔR
ξ10 = jnp.ones(ns[1]) / 3
ξ11 = 1/3 - 1/(3*τ) * ΔR + jnp.sqrt(3)/(3*τ) * ΔY
ξ20 = jnp.ones(ns[1]) / 3
ξ21 = 1/3 - 1/(3*τ) * ΔR - jnp.sqrt(3)/(3*τ) * ΔY

# %%
ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]]) # (3, 2, ns[1]) -> l, i, j

# %%
# f(r, χ, ζ) = ∑ijk [ ∑l f(lk)    ξ(lij)   +   f(ijk) ] φi(r) φj(χ) φk(ζ)
#                       (3, nζ) (3, 2, nχ)   (nr, nχ, nζ)
#
# two parts to the basis:
# - non cartesian part: 3 * nζ
# - cartesian part: (nr - 2) * nχ * nζ

def get_polar_zero_form_basis(ns, ps):

    basis_r = get_spline(ns[0], ps[0], 'clamped')
    basis_χ = get_spline(ns[1], ps[1], 'periodic')
    basis_ζ = get_trig_fn(ns[2], 0, 1)
    bases = (basis_r, basis_χ, basis_ζ)
    nr, nχ, nζ = ns

    def polar_basis(x, I):
        nr, nχ, nζ = ns
        basis_r, basis_χ, basis_ζ = bases
        outer_shape = (nr - 2, nχ, nζ)
        
        def inner_basis(x, I):
            # we are in the f(lk) part of the vector
            l, k = jnp.unravel_index(I, (3, nζ))
            # sum over all j and the first 2 i
            φ_r_i = vmap(basis_r, (None, 0))(x[0], jnp.arange(2))
            φ_χ_j = vmap(basis_χ, (None, 0))(x[1], jnp.arange(nχ))
            φ_ζ_k = basis_ζ(x[2], k)
            return ((ξ @ φ_χ_j) @ φ_r_i)[l] * φ_ζ_k
        
        def outer_basis(x, I):
            I -= 3 * nζ
            i, j, k = jnp.unravel_index(I, outer_shape)
            φ_r_i = basis_r(x[0], i + 2)
            φ_χ_j = basis_χ(x[1], j)
            φ_ζ_k = basis_ζ(x[2], k)
            return φ_r_i * φ_χ_j * φ_ζ_k
        
        return jax.lax.cond(I < 3 * nζ, inner_basis, outer_basis, x, I)
    
    return polar_basis, (nr, nχ, nζ), (3 + (nr - 2) * nχ) * nζ
# %%

basis0, shape0, N0 = get_polar_zero_form_basis(ns, ps)
basis0 = jit(basis0)

nx = 64
_x1 = jnp.linspace(1e-6, 1, nx)
_x2 = jnp.linspace(1e-6, 1, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)

plt.contourf(_x1, _x2, vmap(basis0, (0, None))(_x, 10).reshape(nx, nx))
plt.colorbar()

# %%
def F(x):
    r, χ, ζ = x
    return jnp.array([R_h(x), Y_h(x), 2 * jnp.pi * ζ])
        
# %%
_y = vmap(F)(_x)
_y1 = _y[:,0].reshape(nx, nx)
_y2 = _y[:,1].reshape(nx, nx)
# %%
plt.contourf(_y1, _y2, vmap(basis0, (0, None))(_x, 0).reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')

# %%
M00 = assemble(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
# %%
β = 1/0.2

def f(x):
    r, χ, ζ = x
    return ( 9*r - 32*r**2 + 25*r**3 - (2 * jnp.pi)**2 * (1-r)**2 * r ) * jnp.sin(χ * 2 * jnp.pi)

def u(x):
    r, χ, ζ = x
    return ( (1-r)**2 * r**3 ) * jnp.sin(χ * 2 * jnp.pi)
# # %%
# plt.contourf(_x1, _x2, vmap(f)(_x).reshape(nx, nx))
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')

# # %%
# plt.contourf(_y1, _y2, vmap(f)(_x).reshape(nx, nx))
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')

# %%
### f is given here in terms of x_hat
def get_0form_projection(basis_fn, x_q, w_q, n, F):
    DF = jacfwd(F)
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_projection(f):
        def f_hat(x):
            return f(x) * jnp.linalg.det(DF(x))
        _k = jnp.arange(n, dtype=jnp.int32)
        return vmap(lambda i: l2_product(f_hat, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

proj = get_0form_projection(basis0, x_q, w_q, N0, F)
# %%
f_hat = jnp.linalg.solve(M00, proj(f))
f_h = get_u_h(f_hat, basis0)

# one_hat = jnp.linalg.solve(M, proj(lambda x: 1.0))
# %%
plt.contourf(_y1, _y2, vmap(f_h)(_x).reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')

# %%
# Stiffness Matrix
########
"""
    Lazy stiffness matrix function for two zero-forms: 
    M_ij = ∫ (DF.-T ∇ϕ_i).T DF.-T ∇ϕ_j det DF dξ
"""
def get_stiffness_matrix_lazy(basis_fn, x_q, w_q, F):
    DF = jacfwd(F)
    def A(k):
        return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k)
    def E(k):
        return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k) * jnp.linalg.det(DF(x)) * R_h(x)**2
    def M_ij(i, j):
        return l2_product(A(i), E(j), x_q, w_q)
    return M_ij

# omit outer ring!
K = assemble((get_stiffness_matrix_lazy(basis0, x_q, w_q, F)), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))

M00_dbc = assemble(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))
# %%
g = lambda x: 4 + 2 * R_h(x)**2
proj_dbc = get_0form_projection(basis0, x_q, w_q, N0-ns[1], F)
rhs = proj_dbc(g)

# %%
lambda_hat = jnp.linalg.solve(K, rhs)
u_h = get_u_h(lambda_hat, basis0)

def psi_h(x):
    return u_h(x) / R_h(x)**2

plt.contourf(_y1, _y2, vmap(psi_h)(_x).reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')
# %%
@jit
def rhs(lambda_hat):
    def psi_h(x):
        return get_u_h(lambda_hat, basis0)(x) / R_h(x)**2
    def p_prime(x):
        return 2 * psi_h(x) / psi_h(jnp.zeros_like(x))**2 * R_h(x)**2
    def ff_prime(x):
        return - psi_h(x)
    return proj_dbc(p_prime) + proj_dbc(ff_prime)

# %%
@jit
def residual(lambda_hat, args):
    return jnp.sum((K @ lambda_hat - rhs(lambda_hat))**2)
    
# %%
lambda_hat = jnp.linalg.solve(K, jnp.ones_like(lambda_hat))
residual(lambda_hat, None)

# %%
residuals = [ residual(lambda_hat, None) ]
theta = 0.1
for i in range(100):
    _rhs = rhs(lambda_hat)
    lambda_hat = (1-theta)*lambda_hat + theta * jnp.linalg.solve(K, _rhs)
    residuals.append(residual(lambda_hat, None))
    print("Iteration: ", i+1, 'Residual: ', residuals[-1])
    if i > 1 and residuals[-1] > residuals[-2]:
        break
# %%
plt.plot(residuals)
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('squared L2 error')

# %%
u_h = get_u_h(lambda_hat, basis0)

def psi_h(x):
    return u_h(x) / R_h(x)**2

plt.contourf(_y1, _y2, vmap(psi_h)(_x).reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')
# %%
residual(lambda_hat, None)
# %%
