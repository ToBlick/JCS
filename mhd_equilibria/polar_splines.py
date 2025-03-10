import jax
from jax import numpy as jnp
from jax import vmap

from functools import partial
from mhd_equilibria.splines import *
from mhd_equilibria.vector_bases import get_vector_basis_fn, get_zero_form_basis
from mhd_equilibria.bases import get_trig_fn, get_u_h
from mhd_equilibria.quadratures import quadrature_grid, get_quadrature_composite, get_quadrature_periodic
from mhd_equilibria.projections import get_l2_projection
from mhd_equilibria.forms import assemble, get_mass_matrix_lazy_00


__all__ = [
    "get_polar_form_bases",
    "get_xi"
]

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
    
    N2 = ((2 + (nr - 2) * ndχ) * ndζ,
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

def get_xi(_R, _Y, ns, ps, R0, Y0):
    
    nr, nχ, nζ = ns
    pr, pχ, pζ = ps
    
    basis, _, _ = get_zero_form_basis((nr, nχ, 1), ps, ('clamped', 'periodic', 'fourier'), ('free', 'periodic', 'periodic'))
    n_basis_map = nr * nχ
    basis = jax.jit(basis)

    # quadrature grid and projection
    x_q, w_q = quadrature_grid(
        get_quadrature_composite(jnp.linspace(0, 1, nr - pr + 1), 15),
        get_quadrature_composite(jnp.linspace(0, 1, nχ - pχ + 1), 15),
        get_quadrature_periodic(16)(0,1))
    proj_basis_map = get_l2_projection(basis, x_q, w_q, n_basis_map)

    # This is the mass matrix for the mapping itself, hence F is the identity here
    M_map = assemble(get_mass_matrix_lazy_00(basis, x_q, w_q, lambda x: x), jnp.arange(n_basis_map), jnp.arange(n_basis_map))

    def R(x):
        return _R(x[0], x[1])
    def Y(x):
        return _Y(x[0], x[1])

    R_hat = jnp.linalg.solve(M_map, proj_basis_map(R))
    Y_hat = jnp.linalg.solve(M_map, proj_basis_map(Y))

    cR = R_hat.reshape(nr, nχ)
    cY = Y_hat.reshape(nr, nχ)
    ΔR = cR[1,:] - R0
    ΔY = cY[1,:] - Y0
    τ = max([jnp.max(-2 * ΔR), jnp.max(ΔR - jnp.sqrt(3) * ΔY), jnp.max(ΔR + jnp.sqrt(3) * ΔY)])
    ξ00 = jnp.ones(ns[1]) / 3
    ξ01 = 1/3 + 2/(3*τ) * ΔR
    ξ10 = jnp.ones(ns[1]) / 3
    ξ11 = 1/3 - 1/(3*τ) * ΔR + jnp.sqrt(3)/(3*τ) * ΔY
    ξ20 = jnp.ones(ns[1]) / 3
    ξ21 = 1/3 - 1/(3*τ) * ΔR - jnp.sqrt(3)/(3*τ) * ΔY
    ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]]) # (3, 2, ns[1]) -> l, i, j
    return ξ, R_hat, Y_hat, basis, τ