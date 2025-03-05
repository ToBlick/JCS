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
R0 = 0
Y0 = 0

def R(x):
    r, χ, z = x
    return a * r * jnp.cos(2 * jnp.pi * χ)
def Y(x):
    r, χ, z = x
    return a * r * jnp.sin(2 * jnp.pi * χ)

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
plt.scatter(R_hat + R0, Y_hat, s=5)
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
M00 = assemble_full_vmap(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
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
        return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k) * jnp.linalg.det(DF(x))
    def M_ij(i, j):
        return l2_product(A(i), E(j), x_q, w_q)
    return M_ij

# omit outer ring!
K = assemble((get_stiffness_matrix_lazy(basis0, x_q, w_q, F)), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))

M00_dbc = assemble_full_vmap(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0-ns[1]), jnp.arange(N0-ns[1]))
# %%
# Analytical test case:
# -Δu = 4
# - 1/r ∂r (r ∂r u) = 4
# -> - ∂r (r ∂r u) = 4r
# -> - r ∂r u = 2r² + c1
# -> - ∂r u = 2r + c1/r
# -> u = - r² - c1 log r - c2
# Dirichlet BC: u(r) = 1 - r²

g = lambda x: 4
proj_dbc = get_0form_projection(basis0, x_q, w_q, N0-ns[1], F)
rhs = proj_dbc(g)

# %%
u_hat = jnp.linalg.solve(K, rhs)
u_h = get_u_h(u_hat, basis0)

plt.contourf(_y1, _y2, vmap(u_h)(_x).reshape(nx, nx))
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')

# %%
u_analytic = lambda x: 1 - x[0]**2
# %%
u_hat_an = jnp.linalg.solve(M00_dbc, proj_dbc(u_analytic))
print("L2 error: ", jnp.sqrt( (u_hat_an - u_hat) @ M00_dbc @ (u_hat_an - u_hat) / (u_hat_an @ M00_dbc @ u_hat_an) ))
print("H1 error: ", jnp.sqrt( (u_hat_an - u_hat) @ K @ (u_hat_an - u_hat) / (u_hat_an @ K @ u_hat_an) ))
# %%

### one forms: two new basis functions
nr, nχ, nζ = ns
pr, pχ, pζ = ps
basis_r = get_spline(nr, pr, 'clamped')
basis_χ = get_spline(nχ, pχ, 'periodic')
basis_ζ = get_trig_fn(nζ, 0, 1)

basis_dr = get_spline(nr - 1, pr - 1, 'clamped')
basis_dχ = get_spline(nχ, pχ - 1, 'periodic')
basis_ζ = get_trig_fn(nζ, 0, 1)

bases = (basis_r, basis_χ, basis_ζ)

# ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]])

grad_basis_0 = jit(grad(lambda x: basis0(x, 0)))
grad_basis_1 = jit(grad(lambda x: basis0(x, 1)))
grad_basis_2 = jit(grad(lambda x: basis0(x, 2)))

# %%
def pushfwd_1form(A, F):
    def pushfwd(x):
        return inv33(jax.jacfwd(F)(x)).T @ A(x)
    return pushfwd

gb0_vals = vmap(pushfwd_1form(grad_basis_0, F))(_x)
gb1_vals = vmap(pushfwd_1form(grad_basis_1, F))(_x)
gb2_vals = vmap(pushfwd_1form(grad_basis_2, F))(_x)

plt.quiver(_y1, _y2, 
           gb0_vals[:,0].reshape(nx, nx), 
           gb0_vals[:,1].reshape(nx, nx))

# %%
plt.quiver(_y1, _y2, 
           gb1_vals[:,0].reshape(nx, nx), 
           gb1_vals[:,1].reshape(nx, nx))

# %%
plt.quiver(_y1, _y2, 
           gb2_vals[:,0].reshape(nx, nx), 
           gb2_vals[:,1].reshape(nx, nx))

# %%
gb_vals = - (gb0_vals + gb1_vals + gb2_vals)
plt.quiver(_y1, _y2,
           gb_vals[:,0].reshape(nx, nx), 
           gb_vals[:,1].reshape(nx, nx))

# %%
def get_polar_one_form_basis(ns, ps):
    
    nr, nχ, nζ = ns
    pr, pχ, pζ = ps

    basis_r = get_spline(nr, pr, 'clamped')
    basis_χ = get_spline(nχ, pχ, 'periodic')
    basis_ζ = get_trig_fn(nζ, 0, 1)
    
    basis_dr = get_spline(nr-1, pr-1, 'clamped')
    basis_dχ = get_spline(nχ, pχ-1, 'periodic')
    basis_dζ = get_trig_fn(nζ, 0, 1)
    bases = (basis_r, basis_χ, basis_ζ)
    nr, nχ, nζ = ns
    
    outer_shapes = jnp.array([ [nr-2, nχ, nζ],
                                [nr-2, nχ, nζ],
                                [nr-2, nχ, nζ] ])
    
    N = jnp.sum(jnp.prod(outer_shapes, axis=1)) + 3 * nζ

    def polar_basis(x, I):        
        def inner_basis(x, I):
            # This part is identical to the 0-form basis and we just take the grad after
            #TODO: This should be changed to an explicit implementation using lower order spline bases at some point
            l, k = jnp.unravel_index(I, (3, nζ))
            φ_r_i = vmap(basis_r, (None, 0))(x[0], jnp.arange(2))
            φ_χ_j = vmap(basis_χ, (None, 0))(x[1], jnp.arange(nχ))
            φ_ζ_k = basis_ζ(x[2], k)
            return ((ξ @ φ_χ_j) @ φ_r_i)[l] * φ_ζ_k
        
        # now, build a standard tensor basis for the outer parts
        def outer_basis_r(r,i):
            return basis_r(r, i + 2)
        def outer_basis_dr(r,i):
            return basis_dr(r, i + 1)
        
        outer_basis_1 = get_tensor_basis_fn((outer_basis_dr, basis_χ, basis_ζ), outer_shapes[0])
        outer_basis_2 = get_tensor_basis_fn((outer_basis_r, basis_dχ, basis_ζ), outer_shapes[1])
        outer_basis_3 = get_tensor_basis_fn((outer_basis_r, basis_χ, basis_dζ), outer_shapes[2])
        
        _outer_basis = get_vector_basis_fn((outer_basis_1, outer_basis_2, outer_basis_3), jnp.prod(outer_shapes, axis=1))
        
        def outer_basis(x, I):
            return _outer_basis(x, I - 3 * nζ)
        
        return jax.lax.cond(I < 3 * nζ, grad(inner_basis), outer_basis, x, I)
    
    return polar_basis, outer_shapes, N

# %%
basis1, _, N1 = get_polar_one_form_basis(ns, ps)
basis1 = jit(basis1)

# %%
plt.quiver(_x1, _x2, 
           vmap(basis1, (0, None))(_x, 50)[:,0].reshape(nx, nx), 
           vmap(basis1, (0, None))(_x, 50)[:,1].reshape(nx, nx))
# # %%
# basis_r = get_spline(nr, pr, 'clamped')
# basis_dr = get_spline(nr-1, pr-1, 'clamped')

# for i in range(nr):
#     plt.plot(_x1, basis_r(_x1, i), c = 'k')
# for i in range(nr-1):
#     plt.plot(_x1, basis_dr(_x1, i), c = 'c')
# %%
def get_assembler(f):
    def assemble(ns, ms):
        def scan_fn(carry, j):
            row = vmap(f, (None, 0))(j, ms)
            return carry, row
        _, M = jax.lax.scan(scan_fn, None, ns)
        return M
    return assemble

# %%
import time



_m11 = get_mass_matrix_lazy_11(jit_basis1, x_q, w_q, F)
jit_m11 = jit(_m11)
jit_m11(0,0), _m11(0,0)

start = time.time()
M00 = assemble_full_vmap(_m11, jnp.arange(N0), jnp.arange(N0))
end = time.time()
print(end - start)

_m11 = get_mass_matrix_lazy_11(jit_basis1, x_q, w_q, F)
jit_m11 = jit(_m11)
jit_m11(0,0), _m11(0,0)

start = time.time()
M00 = assemble_full_vmap(jit_m11, jnp.arange(N0), jnp.arange(N0))
end = time.time()
print(end - start)

_m11 = get_mass_matrix_lazy_11(jit_basis1, x_q, w_q, F)
jit_m11 = jit(_m11)
jit_m11(0,0), _m11(0,0)

start = time.time()
M00 = jnp.array([ vmap(_m11, (0, None))(jnp.arange(N0), i) for i in range(N0)])
end = time.time()
print(end - start)

_m11 = get_mass_matrix_lazy_11(jit_basis1, x_q, w_q, F)
jit_m11 = jit(_m11)
jit_m11(0,0), _m11(0,0)

start = time.time()
M00 = jnp.array([ vmap(jit_m11, (0, None))(jnp.arange(N0), i) for i in range(N0)])
end = time.time()
print(end - start)

# %%
import timeit, statistics
n_rep = 100
durations = timeit.Timer('vmap(vmap(basis0, (0, None)),(None, 0))(x_q, jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('vmap(vmap(basis1, (0, None)),(None, 0))(x_q, jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
n_rep = 8
durations = timeit.Timer('vmap(vmap(_m00, (0, None)),(None, 0))(jnp.arange(N0), jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form vmap assembly: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('vmap(vmap(jit_m00, (0, None)),(None, 0))(jnp.arange(N0), jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form vmap assembly, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('assemble(_m00, jnp.arange(N0), jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form assembly: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx
# %%
durations = timeit.Timer('assemble(jit_m00, jnp.arange(N0), jnp.arange(N0))', globals=globals()).repeat(repeat=n_rep, number=1)
print('0 form assembly, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx
# %%
_m11 = get_mass_matrix_lazy_11(basis1, x_q, w_q, F)
jit_m11 = jit(_m11)
_m11(0,0)

# %%
durations = timeit.Timer('vmap(vmap(_m11, (0, None)),(None, 0))(jnp.arange(N1), jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form vmap assembly: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('vmap(vmap(jit_m11, (0, None)),(None, 0))(jnp.arange(N1), jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form vmap assembly, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('assemble(jit_m11, jnp.arange(N1), jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form vmap assembly: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx

# %%
durations = timeit.Timer('assemble(jit_m11, jnp.arange(N1), jnp.arange(N1))', globals=globals()).repeat(repeat=n_rep, number=1)
print('1 form vmap assembly, jitted: ')
jnp.array([statistics.mean(durations), statistics.stdev(durations), statistics.median(durations)])/nr/nx



# %%
# m11 = l2_product(grad_basis_0, grad_basis_0, x_q, w_q)
# m12 = l2_product(grad_basis_0, grad_basis_1, x_q, w_q)
# m13 = l2_product(grad_basis_0, grad_basis_2, x_q, w_q)
# m22 = l2_product(grad_basis_1, grad_basis_1, x_q, w_q)
# m23 = l2_product(grad_basis_1, grad_basis_2, x_q, w_q)
# m33 = l2_product(grad_basis_2, grad_basis_2, x_q, w_q)

# _m = jnp.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])

# # %%
# plt.plot(jnp.abs(vmap(f_h)(_x) - vmap(f)(_x)))
# # %%
# def get_stiffness_matrix_lazy_00(basis_fn, x_q, w_q, F):
#     DF = jacfwd(F)
#     def A(k):
#         return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k)
#         # return lambda x: grad(basis_fn)(x, k)
#     def E(k):
#         return lambda x: inv33(DF(x)).T @ grad(basis_fn)(x, k) * jnp.linalg.det(DF(x))
#         # return lambda x: grad(basis_fn)(x, k)
#     def M_ij(i, j):
#         return l2_product(A(i), E(j), x_q, w_q)
#     return M_ij

# K = assemble(get_stiffness_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
# # %%
# K_fixed = K.at[-1,:].set(1)
# # %%
# f_hat_fixed = proj(f).at[-1].set(0)
# # f_hat_fixed = get_l2_projection(basis0, x_q, w_q, N0)(f).at[-1].set(0)
# # f_hat_fixed = f_hat.at[-1].set(0)
# # %%
# u_hat = jnp.linalg.solve(-K_fixed, f_hat_fixed)
# # %%
# def u_h(x):
#     r, χ, ζ = x
#     return get_u_h(u_hat, basis0)(x)
# # %%

# u_h_proj = get_u_h(jnp.linalg.solve(M, proj(u)), basis0)

# # %%
# plt.contourf(_x1, _x2, vmap(u)(_x).reshape(nx, nx))
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')

# # %%
# plt.contourf(_x1, _x2, vmap(u_h_proj)(_x).reshape(nx, nx))
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')
# # %%
# plt.contourf(_x1, _x2, vmap(u_h)(_x).reshape(nx, nx))
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')

# # %%
# jnp.sum(jnp.abs(vmap(u_h)(_x) - vmap(u_h_proj)(_x)))
# # %%
# jnp.sum(jnp.abs(vmap(u)(_x) - vmap(u_h)(_x)))
# # %%
# jnp.sum(jnp.abs(vmap(u)(_x) - vmap(u_h_proj)(_x)))
# # %%

# %%
