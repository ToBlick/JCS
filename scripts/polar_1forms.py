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

    nr, nχ, nζ = ns
    def polar_basis(x, I):
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

# %%
### Get polar mapping
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
M = assemble(get_mass_matrix_lazy_00(basis, x_q, w_q, lambda x: x), jnp.arange(N), jnp.arange(N))

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
basis0, shape0, N0 = get_polar_zero_form_basis(ns, ps, ξ)
basis0 = jit(basis0)

def F(x):
    r, χ, ζ = x
    return jnp.array([R_h(x), Y_h(x), 2 * jnp.pi * ζ])

nx = 64
_x1 = jnp.linspace(1e-6, 1, nx)
_x2 = jnp.linspace(1e-6, 1, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)

_nx = 16
__x1 = jnp.linspace(1e-6, 1, _nx)
__x2 = jnp.linspace(1e-6, 1, _nx)
__x3 = jnp.zeros(1)
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = vmap(F)(__x)
__y1 = __y[:,0].reshape(_nx, _nx)
__y2 = __y[:,1].reshape(_nx, _nx)

plt.contourf(_x1, _x2, vmap(basis0, (0, None))(_x, 0).reshape(nx, nx), levels=100)
plt.colorbar()
        
# %%
_y = vmap(F)(_x)
_y1 = _y[:,0].reshape(nx, nx)
_y2 = _y[:,1].reshape(nx, nx)
# %%
plt.contourf(_y1, _y2, vmap(basis0, (0, None))(_x, 0).reshape(nx, nx), levels=100)
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
# plt.contourf(_x1, _x2, vmap(f)(_x).reshape(nx, nx), levels=100)
# plt.scatter([0], [0], marker='+', c='w')
# plt.colorbar()
# plt.xlabel('R')
# plt.ylabel('Y')

# # %%
# plt.contourf(_y1, _y2, vmap(f)(_x).reshape(nx, nx), levels=100)
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
_f_hat = jnp.linalg.solve(M00, proj(f))
_f_hhat = proj(f)
f_h = get_u_h(f_hat, basis0)

# %%
jnp.max(vmap(f)(_x) - vmap(f_h)(_x))
# %%
plt.contourf(_y1, _y2, vmap(f_h)(_x).reshape(nx, nx), levels=100)
plt.scatter([0], [0], marker='+', c='w')
plt.colorbar()
plt.xlabel('R')
plt.ylabel('Y')

### One-forms
def get_polar_form_bases(ns, ps, ξ):
    basis_r = get_spline(ns[0], ps[0], 'clamped')
    basis_dr = get_deriv_spline(ns[0], ps[0], 'clamped')
    basis_χ = get_spline(ns[1], ps[1], 'periodic')
    basis_dχ = get_deriv_spline(ns[1], ps[1], 'periodic')
    basis_ζ = get_trig_fn(ns[2], 0, 1)
    basis_dζ = basis_ζ

    nr, nχ, nζ = ns
    ndr, ndχ, ndζ = nr - 1, nχ, nζ
    
    N0 = (3 + (nr - 2) * nχ) * nζ
    
    N1 = ((ndr - 1) * nχ * nζ,
          (2 + (nr - 2) * ndχ) * nζ,
          (3 + (nr - 2) * nχ) * ndζ)
    
    N2 = ((2 + (nr - 2)) * ndχ * ndζ,
          (ndr - 1) * nχ * ndζ,
          (ndr - 1) * ndχ * nζ)
    
    N3 = (ndr - 1) * ndχ * ndζ
    
    def zeroform_inner_basis(x, I):
        # we are in the f(lk) part of the vector
        l, k = jnp.unravel_index(I, (3, nζ))
        # sum over all j and the first 2 i
        φ_r_i = vmap(basis_r, (None, 0))(x[0], jnp.arange(2))
        φ_χ_j = vmap(basis_χ, (None, 0))(x[1], jnp.arange(nχ))
        φ_ζ_k = basis_ζ(x[2], k)
        return ((ξ @ φ_χ_j) @ φ_r_i)[l] * φ_ζ_k
    
    def zeroform_outer_basis(x, I):
        I -= 3 * nζ
        i, j, k = jnp.unravel_index(I, (nr - 2, nχ, nζ))
        φ_r_i = basis_r(x[0], i + 2)
        φ_χ_j = basis_χ(x[1], j)
        φ_ζ_k = basis_ζ(x[2], k)
        return φ_r_i * φ_χ_j * φ_ζ_k
    
    def oneform_inner_basis(x, I):
        l, k = jnp.unravel_index(I, (2, nζ))
        φ_r_1 = basis_r(x[0], 1)
        φ_dr_0 = basis_dr(x[0], 0)
        φ_χ_j = vmap(basis_χ, (None, 0))(x[1], jnp.arange(nχ))
        φ_dχ_j = vmap(basis_dχ, (None, 0))(x[1], jnp.arange(ndχ))
        φ_ζ_k = basis_ζ(x[2], k)
        val0 = ((ξ @ φ_χ_j)[l,1] - (ξ @ φ_χ_j)[l,0]) * φ_dr_0 * φ_ζ_k
        _v = ξ[l,1,:] # this is a ndχ vector
        _v_offset = jnp.concatenate([_v[1:], _v[:1]])
        val1 = (_v_offset - _v) @ φ_dχ_j * φ_r_1 * φ_ζ_k
        return jnp.array([val0, val1, 0.0])
    
    def oneform_outer_basis(x, I):
        I -= 2 * nζ
        i, j, k = jnp.unravel_index(I, (nr - 2, ndχ, nζ))
        φ_r_i = basis_r(x[0], i + 2)
        φ_dχ_j = basis_dχ(x[1], j)
        φ_ζ_k = basis_ζ(x[2], k)
        return jnp.zeros(3).at[1].set(φ_r_i * φ_dχ_j * φ_ζ_k)
    
    def zeroform_basis(x, I):
        return jax.lax.cond(I < 3 * nζ, 
                            zeroform_inner_basis, 
                            zeroform_outer_basis, x, I)
    
    # First component is a tensor product basis excluding the inner ring
    def oneform_component_1(x, I):
        i, j, k = jnp.unravel_index(I, (ndr - 1, nχ, nζ))
        φ_dr_i = basis_dr(x[0], i + 1)
        φ_χ_j = basis_χ(x[1], j)
        φ_ζ_k = basis_ζ(x[2], k)
        return jnp.zeros(3).at[0].set(φ_dr_i * φ_χ_j * φ_ζ_k)
    
    def oneform_component_2(x, I):
        return jax.lax.cond(I < 2 * nζ, 
                            oneform_inner_basis, 
                            oneform_outer_basis, x, I)
        
    def oneform_component_3(x, I):
        return jnp.zeros(3).at[2].set(zeroform_basis(x, I))
    
    def twoform_inner_basis(x, I):
        return jnp.linalg.cross(oneform_inner_basis(x, I), jnp.array([0, 0, 1]))
    
    def twoform_outer_basis(x, I):
        I -= 2 * nζ
        i, j, k = jnp.unravel_index(I, (nr - 2, ndχ, ndζ))
        φ_r_i = basis_r(x[0], i + 2)
        φ_dχ_j = basis_dχ(x[1], j)
        φ_dζ_k = basis_dζ(x[2], k)
        return jnp.zeros(3).at[0].set(φ_r_i * φ_dχ_j * φ_dζ_k)
    
    def twoform_component_1(x, I):
        return jax.lax.cond(I < 2 * nζ, 
                            twoform_inner_basis, 
                            twoform_outer_basis, x, I)
    
    def twoform_component_2(x, I):
        i, j, k = jnp.unravel_index(I, (ndr - 1, nχ, ndζ))
        φ_dr_i = basis_dr(x[0], i + 1)
        φ_χ_j = basis_χ(x[1], j)
        φ_dζ_k = basis_dζ(x[2], k)
        return jnp.zeros(3).at[1].set(φ_dr_i * φ_χ_j * φ_dζ_k)
    
    def twoform_component_3(x, I):
        # tensor product basis excluding the inner ring
        i, j, k = jnp.unravel_index(I, (ndr - 1, ndχ, nζ))
        φ_dr_i = basis_dr(x[0], i + 1)
        φ_dχ_j = basis_dχ(x[1], j)
        φ_ζ_k = basis_ζ(x[2], k)
        return jnp.zeros(3).at[2].set(φ_dr_i * φ_dχ_j * φ_ζ_k)
    
    def threeform_basis(x, I):
        i, j, k = jnp.unravel_index(I, (ndr - 1, ndχ, ndζ))
        φ_dr_i = basis_dr(x[0], i + 1)
        φ_dχ_j = basis_dχ(x[1], j)
        φ_dζ_k = basis_dζ(x[2], k)
        return φ_dr_i * φ_dχ_j * φ_dζ_k
    
    oneform_basis = get_vector_basis_fn( (oneform_component_1, oneform_component_2, oneform_component_3), N1 )
    twoform_basis = get_vector_basis_fn( (twoform_component_1, twoform_component_2, twoform_component_3), N2 )
    
    return zeroform_basis, oneform_basis, twoform_basis, threeform_basis

# %%
basis_0, basis_1, basis_2, basis_3 = get_polar_form_bases(ns, ps, ξ)
basis_0 = jit(basis_0)
basis_1 = jit(basis_1)
basis_2 = jit(basis_2)
basis_3 = jit(basis_3)

def _test_basis(x, i):
    return jnp.sum(basis_1(x, i)**2)**0.5

# %%
plt.contourf(_x1, _x2, vmap(basis0, (0, None))(_x, 2).reshape(nx, nx), levels=100)
plt.colorbar()

# %%
nr, nχ, nζ = ns
ndr, ndχ, ndζ = nr - 1, nχ, nζ

N0 = (3 + (nr - 2) * nχ) * nζ
    
N1 = ((ndr - 1) * nχ * nζ,
      (2 + (nr - 2) * ndχ) * nζ,
      (3 + (nr - 2) * nχ) * ndζ)

N2 = ((2 + (nr - 2)) * ndχ * ndζ,
      (ndr - 1) * nχ * ndζ,
      (ndr - 1) * ndχ * nζ)

N3 = (ndr - 1) * ndχ * ndζ

# %%
i_test = N1[0] + 1
plt.contourf(_x1, _x2, vmap(_test_basis, (0, None))(_x, i_test).reshape(nx, nx), levels=100)
plt.colorbar()

# %%
plt.contourf(_x1, _x2, vmap(_test_basis, (0, None))(_x, i_test).reshape(nx, nx), levels=100)
plt.quiver(__x1, __x2, 
           vmap(basis_1, (0, None))(__x, i_test).reshape(_nx, _nx, 3)[:,:,0], 
           vmap(basis_1, (0, None))(__x, i_test).reshape(_nx, _nx, 3)[:,:,1],
           color='w')
plt.colorbar()

# %%
def pushfwd_0form(p, F):
    def pushfwd(x):
        return p((x))
    return pushfwd

def pushfwd_1form(A, F):
    def pushfwd(x):
        return inv33(jax.jacfwd(F)(x)).T @ A((x))
    return pushfwd

def pushfwd_2form(B, F):
    def pushfwd(x):
        J = jnp.linalg.det(jax.jacfwd(F)(x))
        return (jax.jacfwd(F)(x)) @ B((x)) * 1/J
    return pushfwd

def pushfwd_3form(f, F):
    def pushfwd(x):
        J = jnp.linalg.det(jax.jacfwd(F)(x))
        return 1/J * f((x))
    return pushfwd
# %%
__test_basis = pushfwd_1form(lambda x: basis_1(x, N1[0]+1), F)
# %%
vals = vmap(__test_basis)(__x).reshape(_nx, _nx, 3)
_norm = jnp.sum(vals**2, axis=2)**0.5

plt.contourf(_y1, _y2, jnp.sum(vmap(__test_basis)(_x).reshape(nx, nx, 3)**2, axis=2)**0.5, levels=100)
plt.colorbar()
plt.quiver(__y1, __y2,
            vals[:,:,0] / jnp.sum(vals**2, axis=2)**0.5,
            vals[:,:,1] / jnp.sum(vals**2, axis=2)**0.5,
            color='w')



# %%
def _test_basis(x, i):
    return jnp.sum(basis_2(x, i)**2)**0.5
i_test = 0
plt.contourf(_x1, _x2, vmap(_test_basis, (0, None))(_x, i_test).reshape(nx, nx), levels=100)
plt.colorbar()

# %%
plt.contourf(_x1, _x2, vmap(_test_basis, (0, None))(_x, i_test).reshape(nx, nx), levels=100)
plt.quiver(__x1, __x2, 
           vmap(basis_2, (0, None))(__x, i_test).reshape(_nx, _nx, 3)[:,:,0], 
           vmap(basis_2, (0, None))(__x, i_test).reshape(_nx, _nx, 3)[:,:,1],
           color='w')
plt.colorbar()
# %%
__test_basis = pushfwd_2form(lambda x: basis_2(x, 1), F)
vals = vmap(__test_basis)(__x).reshape(_nx, _nx, 3)
_norm = jnp.sum(vals**2, axis=2)**0.5

plt.contourf(_y1, _y2, jnp.sum(vmap(__test_basis)(_x).reshape(nx, nx, 3)**2, axis=2)**0.5, levels=100)
plt.colorbar()
plt.quiver(__y1, __y2,
            vals[:,:,0] / jnp.sum(vals**2, axis=2)**0.5,
            vals[:,:,1] / jnp.sum(vals**2, axis=2)**0.5,
            color='w')
# %%
