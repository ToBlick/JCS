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
        return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k) * jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(A(i), E(j), x_q, w_q)
    return M_ij

def get_0form_projection(basis_fn, x_q, w_q, n, F):
    DF = jacfwd(F)
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_projection(f):
        def f_hat(x):
            return f(x) * jnp.linalg.det(DF(x))
        _k = jnp.arange(n)
        return vmap(lambda i: l2_product(f_hat, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

# f(r, χ, ζ) = ∑ijk [ ∑l f(lk)    ξ(lij)   +   f(ijk) ] φi(r) φj(χ) φk(ζ)
#                       (3, nζ) (3, 2, nχ)   (nr, nχ, nζ)
#
# two parts to the basis:
# - non cartesian part: 3 * nζ
# - cartesian part: (nr - 2) * nχ * nζ
def get_polar_zero_form_basis(ns, ps, ξ):
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
@jit
def J(α):
    # Define the mapping
    nα = α.shape[0]
    map_basis_fn = get_tensor_basis_fn((get_trig_fn(nα, 0, 1), ), (nα,))
    α = α.at[0].set(1.0)
    def ς(θ):
        return get_u_h(α, map_basis_fn)(θ * jnp.ones(1))

    # this is the angle-dependent radius
    R0 = 0
    Y0 = 0
    def R(x):
        r, χ, z = x
        return ς(χ) * r * jnp.cos(2 * jnp.pi * χ)
    def Y(x):
        r, χ, z = x
        return ς(χ) * r * jnp.sin(2 * jnp.pi * χ)
    def F(x):
        r, χ, ζ = x
        return jnp.array([R_h(x), Y_h(x), 2 * jnp.pi * ζ])

    ns = (4, 8, 1)
    ps = (3, 3, 1)

    ### Project this mapping to the zero form basis
    types = ('clamped', 'periodic', 'fourier')
    boundary = ('free', 'periodic', 'periodic')
    map_basis, shape, N = get_zero_form_basis(ns, ps, types, boundary)
    N = ns[0] * ns[1] * ns[2]

    # quadrature grid and projection
    x_q, w_q = quadrature_grid(
        get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
        get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
        get_quadrature_periodic(1)(0,1))
    proj = get_l2_projection(map_basis, x_q, w_q, N)

    M = assemble(get_mass_matrix_lazy_00(map_basis, x_q, w_q, lambda x: x), jnp.arange(N), jnp.arange(N))

    # isogeometric mapping

    R_hat = jnp.linalg.solve(M, proj(R))
    R_h = get_u_h(R_hat, map_basis)
    Y_hat = jnp.linalg.solve(M, proj(Y))
    Y_h = get_u_h(Y_hat, map_basis)

    cR = R_hat.reshape(ns[0], ns[1])
    cY = Y_hat.reshape(ns[0], ns[1])

    ΔR = cR[1,:] - R0
    ΔY = cY[1,:] - Y0

    τ = jnp.max(jnp.array([jnp.max(-2 * ΔR), jnp.max(ΔR - jnp.sqrt(3) * ΔY), jnp.max(ΔR + jnp.sqrt(3) * ΔY)]))

    # plt.scatter(R_hat + R0, Y_hat, s=5)
    # plt.scatter([τ + R0, R0 - τ/2, R0 - τ/2], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2], s=10, c='k')
    # plt.plot([τ + R0, R0 - τ/2, R0 - τ/2, τ + R0], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2, 0], 'k:')

    ξ00 = jnp.ones(ns[1]) / 3
    ξ01 = 1/3 + 2/(3*τ) * ΔR
    ξ10 = jnp.ones(ns[1]) / 3
    ξ11 = 1/3 - 1/(3*τ) * ΔR + jnp.sqrt(3)/(3*τ) * ΔY
    ξ20 = jnp.ones(ns[1]) / 3
    ξ21 = 1/3 - 1/(3*τ) * ΔR - jnp.sqrt(3)/(3*τ) * ΔY
    ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]]) # (3, 2, ns[1]) -> l, i, j

    basis0, shape0, N0 = get_polar_zero_form_basis(ns, ps, ξ)
    N0 = (ns[0] - 2) * ns[1] * ns[2] + 3 * ns[2]

    nx = 64
    _x1 = jnp.linspace(1e-6, 1, nx)
    _x2 = jnp.linspace(1e-6, 1, nx)
    _x3 = jnp.zeros(1)
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
    _x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
            
    _y = vmap(F)(_x)
    _y1 = _y[:,0].reshape(nx, nx)
    _y2 = _y[:,1].reshape(nx, nx)
    # plt.contourf(_y1, _y2, vmap(basis0, (0, None))(_x, 0).reshape(nx, nx))
    # plt.scatter([0], [0], marker='+', c='w')
    # plt.colorbar()
    # plt.xlabel('R')
    # plt.ylabel('Y')

    # omit outer ring!
    K = assemble((get_stiffness_matrix_lazy(basis0, x_q, w_q, F)), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))

    M00_dbc = assemble(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))

    proj = get_0form_projection(basis0, x_q, w_q, N0, F)

    # Analytical test case:
    # -Δu = 4

    g = lambda x: 4
    proj_dbc = get_0form_projection(basis0, x_q, w_q, N0-ns[1], F)
    rhs = proj_dbc(g)

    u_hat = jnp.linalg.solve(K, rhs)
    u_h = get_u_h(u_hat, basis0)
    u_h_vals = vmap(u_h)(_x)

    # plt.contourf(_y1, _y2, vmap(u_h)(_x).reshape(nx, nx))
    # plt.scatter([0], [0], marker='+', c='w')
    # plt.colorbar()
    # plt.xlabel('R')
    # plt.ylabel('Y')

    u_analytic = lambda x: 1 - x[0]**2
    u_hat_an = jnp.linalg.solve(M00_dbc, proj_dbc(u_analytic))
    # print("L2 error: ", jnp.sqrt( (u_hat_an - u_hat) @ M00_dbc @ (u_hat_an - u_hat) / (u_hat_an @ M00_dbc @ u_hat_an) ))
    # print("H1 error: ", jnp.sqrt( (u_hat_an - u_hat) @ K @ (u_hat_an - u_hat) / (u_hat_an @ K @ u_hat_an) ))

    u_h_an = get_u_h(u_hat_an, basis0)
    u_h_an_vals = vmap(u_h_an)(_x)
    # plt.contourf(_y1, _y2, vmap(u_h_an)(_x).reshape(nx, nx))
    # plt.scatter([0], [0], marker='+', c='w')
    # plt.colorbar()
    # plt.xlabel('R')
    # plt.ylabel('Y')
    err = jnp.sqrt( (u_hat_an - u_hat) @ M00_dbc @ (u_hat_an - u_hat) / (u_hat_an @ M00_dbc @ u_hat_an) )
    return err, ( _y1, _y2, u_h_vals, u_h_an_vals )

# %%
import optax

key = jax.random.PRNGKey(42)
α = 0.25 * jax.random.normal(key, (8,)) / jnp.arange(1, 9)
α = α.at[0].set(1.0)
value, _grad = jax.value_and_grad(J, has_aux=True)(α)
value, (_y1, _y2, u_h_vals, u_h_an_vals) = value

trace_α = [ α ]
trace_vals = [ value ]
trace_grads = [ _grad ]
trace_y1 = [ _y1 ]
trace_y2 = [ _y2 ]
trace_u_h_vals = [ u_h_vals ]

it = 0
beta = 0.9
eta = 5e-4
z = 0
while jnp.sum(_grad**2)**0.5 > 1e-2 and it < 200:
    value, _grad = jax.value_and_grad(J, has_aux=True)(α)
    value, (_y1, _y2, u_h_vals, u_h_an_vals) = value
    z = beta * z + _grad
    α = α - eta * z
    trace_α.append(α)
    trace_vals.append(value)
    trace_grads.append(_grad)
    trace_y1.append(_y1)
    trace_y2.append(_y2)
    trace_u_h_vals.append(u_h_vals)
    
    it += 1
    print("Iteration: ", it, "Value: ", value, "Gradient: ", jnp.sum(_grad**2)**0.5)
# %%
plt.plot(trace_vals, label='J(α)')
# plt.plot([ jnp.sqrt(jnp.sum(_grad**2)) for _grad in trace_grads ], label='Gradient norm') 
plt.yscale('log')
plt.xlabel('Iteration')
plt.legend()

# %%
nα = α.shape[0]
map_basis_fn = get_tensor_basis_fn((get_trig_fn(nα, 0, 1), ), (nα,))
α = α.at[0].set(1.0)
def ς(θ):
    return get_u_h(α, map_basis_fn)(θ * jnp.ones(1))

_θ = jnp.linspace(0, 1, 100)
colscale = plt.cm.get_cmap('viridis')
for i,α in enumerate(trace_α[::10]):
    α = α.at[0].set(1.0)
    print(i)
    def ς(θ):
        return get_u_h(α, map_basis_fn)(θ * jnp.ones(1))
    plt.plot(_θ, vmap(ς)(_θ), color=colscale(i/(len(trace_α[::10])-1)))
plt.xlabel('θ')
plt.ylabel('ς(θ)')
# %%
i = 200
nx = 64
plt.contourf(trace_y1[i], trace_y2[i], trace_u_h_vals[i].reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')
# %%
plt.contourf(trace_y1[i], trace_y2[i], u_h_an_vals.reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')
# %%
