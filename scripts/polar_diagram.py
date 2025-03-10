# %%
from jax import jit, config, vmap, grad, jacfwd, jacrev
import jax.numpy as jnp
from jax.lib import xla_bridge

import matplotlib.pyplot as plt 

from mhd_equilibria.vector_bases import get_zero_form_basis
from mhd_equilibria.quadratures import quadrature_grid, get_quadrature_composite, get_quadrature_periodic
from mhd_equilibria.projections import get_l2_projection
from mhd_equilibria.forms import assemble, inv33, get_mass_matrix_lazy_00, get_mass_matrix_lazy_11, get_mass_matrix_lazy_22, get_mass_matrix_lazy_33
from mhd_equilibria.bases import get_u_h, get_u_h_vec
from mhd_equilibria.polar_splines import *
from mhd_equilibria.projections import get_0form_projection, l2_product
from mhd_equilibria.operators import curl, div

### Enable double precision
config.update("jax_enable_x64", True)

### This will print out the current backend (cpu/gpu)
print(xla_bridge.get_backend().platform)

# %%

###
# Mapping definition
###

a = 1
R0 = 0.0
Y0 = 0.0

def θ(x):
    r, χ, z = x
    return 2 * jnp.atan( jnp.sqrt( (1 + a*r/R0)/(1 - a*r/R0) ) * jnp.tan(jnp.pi * χ) )

def _R(r, χ):
    return R0 + a * r * jnp.cos(2 * jnp.pi * χ)
def _Y(r, χ):
    return a * r * jnp.sin(2 * jnp.pi * χ)

def F(x):
    r, χ, z = x
    return jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z), 
                      _Y(r, χ),
                      _R(r, χ) * jnp.sin(2 * jnp.pi * z)])

# %%
###
# Isogeometric mapping: Get a zero-form basis function to represent the mapping
###

ns = (4, 8, 3)
ps = (2, 2, 1)

# %%
ξ, R_hat, Y_hat, basis_map, τ = get_xi(_R, _Y, ns, ps, R0, Y0)
R_h = get_u_h(R_hat, basis_map)
Y_h = get_u_h(Y_hat, basis_map)

def F(x):
    r, χ, ζ = x
    return jnp.array([R_h(x), Y_h(x), 2 * jnp.pi * ζ])
# %%
basis_0, basis_1, basis_2, basis_3 = get_polar_form_bases(ns, ps, ξ)
basis_0 = jit(basis_0)
basis_1 = jit(basis_1)
basis_2 = jit(basis_2)
basis_3 = jit(basis_3)

nr, nχ, nζ = ns
ndr, ndχ, ndζ = nr - 1, nχ, nζ
pr, pχ, pζ = ps

N0 = (3 + (nr - 2) * nχ) * nζ
N1 = ((ndr - 1) * nχ * nζ,
      (2 + (nr - 2) * ndχ) * nζ,
      (3 + (nr - 2) * nχ) * ndζ)
N2 = ((2 + (nr - 2) * ndχ) * ndζ,
      (ndr - 1) * nχ * ndζ,
      (ndr - 1) * ndχ * nζ)
N3 = (ndr - 1) * ndχ * ndζ

x_q, w_q = quadrature_grid(
    get_quadrature_composite(jnp.linspace(0, 1, nr - pr + 1), 15),
    get_quadrature_composite(jnp.linspace(0, 1, nχ - pχ + 1), 15),
    get_quadrature_periodic(8)(0,1))

# %%
###
# Test 1: Zero forms
### 
def f(x):
    r, χ, ζ = x
    return ( 9*r - 32*r**2 + 25*r**3 - (2 * jnp.pi)**2 * (1-r)**2 * r ) * jnp.sin(χ * 2 * jnp.pi) * jnp.cos(ζ * 2 * jnp.pi)

proj_0 = get_0form_projection(basis_0, x_q, w_q, N0, F)

# %%
M00 = assemble(get_mass_matrix_lazy_00(basis_0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
# %%
jnp.linalg.cond(M00)
# %%
f_hat = jnp.linalg.solve(M00, proj_0(f))
f_h = get_u_h(f_hat, basis_0)

# Check mismatch
nx = 8
nz = 8
_x1 = jnp.linspace(1e-6, 1, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.linspace(0, 1, nz)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*nz, 3)

print(jnp.linalg.norm(vmap(f)(_x) - vmap(f_h)(_x)) / nx**3)

# %%
###
# Test 2: exact gradients
###


def get_1form_projection(basis_fn, x_q, w_q, n, F):
    DF = jacfwd(F)
    def get_basis(k):
        return lambda x: inv33(DF(x)).T @ basis_fn(x, k)
    def l2_projection(f):
        def f_hat(x):
            return inv33(DF(x)).T @ f(x) * jnp.linalg.det(DF(x))
        _k = jnp.arange(n)
        return vmap(lambda i: l2_product(f_hat, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

M11 = assemble(get_mass_matrix_lazy_11(basis_1, x_q, w_q, F), jnp.arange(sum(N1)), jnp.arange(sum(N1)))


# %%
proj_1 = get_1form_projection(basis_1, x_q, w_q, sum(N1), F)

grad_f_hat = jnp.linalg.solve(M11, proj_1(grad(f_h)))
grad_f_h = get_u_h_vec(grad_f_hat, basis_1)


# %%
print(jnp.linalg.norm(vmap(grad_f_h)(_x) - vmap(grad(f))(_x)) / nx**3)
print(jnp.max(vmap(grad_f_h)(_x) - vmap(grad(f_h))(_x)))
# this should be machine precision

# %%
def get_2form_projection(basis_fn, x_q, w_q, n, F):
    DF = jacfwd(F)
    def get_basis(k):
        return lambda x: DF(x) @ basis_fn(x, k)
    def l2_projection(f):
        def f_hat(x):
            return DF(x) @ f(x) / jnp.linalg.det(DF(x))
        _k = jnp.arange(n)
        return vmap(lambda i: l2_product(f_hat, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

M22 = assemble(get_mass_matrix_lazy_22(basis_2, x_q, w_q, F), jnp.arange(sum(N2)), jnp.arange(sum(N2)))

M33 = assemble(get_mass_matrix_lazy_33(basis_3, x_q, w_q, F), jnp.arange((N3)), jnp.arange(N3))

# %%
print(jnp.linalg.cond(M00),
        jnp.linalg.cond(M11),
        jnp.linalg.cond(M22),
        jnp.linalg.cond(M33))
# %%
###
# Test 3: exact curls
###

def A(x):
    r, χ, ζ = x
    A1 = r**2 * jnp.cos(2*jnp.pi*χ) * jnp.sin(2*jnp.pi*ζ)
    A2 = r**3 * jnp.cos(2*jnp.pi*χ) * jnp.sin(2*jnp.pi*ζ)
    A3 = r**2 * (1-r) * jnp.sin(2*jnp.pi*χ) * jnp.cos(2*jnp.pi*ζ)
    return jnp.array([A1, A2, A3])

proj_2 = get_2form_projection(basis_2, x_q, w_q, sum(N2), F)

A_hat = jnp.linalg.solve(M11, proj_1(A))
A_h = get_u_h_vec(A_hat, basis_1)
# %%
print(jnp.linalg.norm(vmap(A)(_x) - vmap(A_h)(_x)) / nx**3)


# %%
curl_A_hat = jnp.linalg.solve(M22, proj_2(curl(A_h)))
curl_A_h = get_u_h_vec(curl_A_hat, basis_2)

print(jnp.max(vmap(curl(A_h))(_x) - vmap(curl_A_h)(_x)))
print(jnp.linalg.norm(vmap(curl(A_h))(_x) - vmap(curl(A))(_x))/ nx**3)
# %%
###
# Test 4: exact divergences
###

def get_3form_projection(basis_fn, x_q, w_q, n, F):
    DF = jacfwd(F)
    def get_basis(k):
        return lambda x: basis_fn(x, k)
    def l2_projection(f):
        def f_hat(x):
            return f(x) / jnp.linalg.det(DF(x))
        _k = jnp.arange(n)
        return vmap(lambda i: l2_product(f_hat, get_basis(i), x_q, w_q))(_k) 
    return l2_projection

proj_3 = get_3form_projection(basis_3, x_q, w_q, N3, F)

A_hat = jnp.linalg.solve(M22, proj_2(A))
A_h = get_u_h_vec(A_hat, basis_2)

div_A_hat = jnp.linalg.solve(M33, proj_3(div(A_h)))
div_A_h = get_u_h(div_A_hat, basis_3)

print(jnp.max(vmap(div(A_h))(_x) - vmap(div_A_h)(_x)))
print(jnp.linalg.norm(vmap(div(A_h))(_x) - vmap(div(A))(_x))/ nx**3)
# %%
