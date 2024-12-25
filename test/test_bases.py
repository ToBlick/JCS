import unittest
from mhd_equilibria.bases import *
from mhd_equilibria.bases import _binom, _get_legendre_coeffs, _get_radial_zernike_coeff, _get_radial_zernike_coeffs
import numpy.testing as npt
import numpy as np
from jax import numpy as jnp
from jax import vmap, jit, grad
import jax
jax.config.update("jax_enable_x64", True)
import time

#TODO: Add tests for chebychev

class BasisTests(unittest.TestCase):
    
    def test_binom(self):
        npt.assert_allclose(_binom(5, 2) , 10,   rtol=1e-9)
        npt.assert_allclose(_binom(10, 5), 252,  rtol=1e-9)
        npt.assert_allclose(_binom(7, 7) , 1,    rtol=1e-9)
        npt.assert_allclose(_binom(7, 0) , 1,    rtol=1e-9)
        npt.assert_allclose(_binom(0, 0) , 1,    rtol=1e-9)
        npt.assert_allclose(_binom(10, 1), 10,   rtol=1e-9) 
    
    def test_legendre(self):
        nx = 1000
        L = 1.32
        K = 5
        computed_coeffs = (_get_legendre_coeffs(K))
        npt.assert_allclose([1, 12, 30, 20, 0, 0], computed_coeffs[3],       rtol=1e-9)
        npt.assert_allclose([1, 20, 90, 140, 70, 0], computed_coeffs[4],     rtol=1e-9)
        npt.assert_allclose([1, 30, 210, 560, 630, 252], computed_coeffs[5], rtol=1e-9)
        
        # closure
        legendre_fn_x = get_legendre_fn_x(5, 0, L)
        
        _x = np.linspace(0, L, nx)
        for x in _x[::10]:
            ψ3 = (20*(x/L)**3 - 30*(x/L)**2 + 12*(x/L) - 1) * np.sqrt((2*3 + 1)/L)
            ψ4 = (70*(x/L)**4 - 140*(x/L)**3 + 90*(x/L)**2 - 20*(x/L) + 1) * np.sqrt((2*4 + 1)/L)
            npt.assert_allclose(legendre_fn_x(x, 3), ψ3, rtol=1e-6)
            npt.assert_allclose(legendre_fn_x(x, 4), ψ4, rtol=1e-6)
            
        weights = L * np.ones(nx) / (nx-1)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        t0 = time.time_ns()
        ψ3 = vmap(legendre_fn_x, (0, None))(_x, 3)
        ψ4 = vmap(legendre_fn_x, (0, None))(_x, 4)
        ψ5 = vmap(legendre_fn_x, (0, None))(_x, 5)
        t1 = time.time_ns()
        print("legendre time = ", (t1-t0)/1e9, ' s')
        npt.assert_allclose(np.sum(weights * ψ3**2), 1, rtol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ4**2), 1, rtol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ3*ψ4), 0, atol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ5*ψ4), 0, atol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ3*ψ5), 0, atol=1e-3)
        
    def test_fourier(self):
        nx = 1000
        L = 2.71
        
        trig_fn_x = jit(get_trig_fn(5, 0, L))
        
        _x = np.linspace(0, L, nx)
        for k in range(1, 10):
            for x in _x[::10]:
                k_half = (k+1)//2
                if k == 0:
                    ψ = 1/jnp.sqrt(L)
                elif k % 2 == 0:
                    ψ = np.cos(2*k_half*np.pi*x/L) * np.sqrt(2/L)
                else:
                    ψ = np.sin(2*k_half*np.pi*x/L) * np.sqrt(2/L)
                npt.assert_allclose(trig_fn_x(x, k), ψ, rtol=1e-12)
        
        weights = L * np.ones(nx) / (nx-1)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        ψ3 = vmap(trig_fn_x, (0, None))(_x, 3)
        ψ4 = vmap(trig_fn_x, (0, None))(_x, 4)
        ψ5 = vmap(trig_fn_x, (0, None))(_x, 5)
        
        npt.assert_allclose(np.sum(weights * ψ3**2), 1, rtol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ4**2), 1, rtol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ3*ψ4), 0, atol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ5*ψ4), 0, atol=1e-3)
        npt.assert_allclose(np.sum(weights * ψ3*ψ5), 0, atol=1e-3)
        
    def test_tensor(self):
        n_r = 7
        n_θ = 2*5 + 1
        n_x = 100
        
        Omega = ((0, 1), (0, 2*jnp.pi))
        bases = (get_legendre_fn_x(n_r, *Omega[0]), get_trig_fn(n_θ, *Omega[1]))
        shape = (n_r, n_θ)
        
        basis_fn = get_tensor_basis_fn(bases, shape)
        
        r = jnp.linspace(*Omega[0], n_x)
        θ = jnp.linspace(*Omega[1], n_x)
        
        x = jnp.array(jnp.meshgrid(r, θ)) # shape 2, n_x, n_x
        x = x.transpose(1, 2, 0).reshape(n_x**2, 2)
        
        # first basis fct. is a constant
        ψ = vmap(basis_fn, (0, None))(x, 0)
        ψ_ref = 1/jnp.sqrt(2*jnp.pi)
        npt.assert_allclose(ψ, ψ_ref, rtol=1e-4)
        
        ψ = vmap(grad(basis_fn, 0), (0, None))(x, 0)
        ψ_ref = 0
        npt.assert_allclose(ψ, ψ_ref, atol=1e-9)
        
        # basis fct. 36 the product of the Legendre poly. of order 3 
        # and the 4th trig. basis which is sin(4*pi*x/L)
        def _ψ_ref(x):
            a = ( 20*x[0]**3 - 30*x[0]**2 + 12*x[0] - 1 ) * np.sqrt((2*3 + 1))
            b = jnp.sin(2*x[1]) * jnp.sqrt(1/jnp.pi)
            return a*b
        ψ_ref = vmap(_ψ_ref)(x)
        ψ = vmap(basis_fn, (0, None))(x, 36)
        npt.assert_allclose(ψ, ψ_ref, atol=1e-6)
        
        def _ψ_ref_grad(x):
            a = ( 20*x[0]**3 - 30*x[0]**2 + 12*x[0] - 1 ) * np.sqrt((2*3 + 1))
            b = jnp.sin(2*x[1]) * jnp.sqrt(1/jnp.pi)
            da = ( 60*x[0]**2 - 60*x[0]**1 + 12) * np.sqrt((2*3 + 1))
            db = 2 * jnp.cos(2*x[1]) * jnp.sqrt(1/jnp.pi)
            return jnp.array([ da*b, a*db ])
        ψ_ref = vmap(_ψ_ref_grad)(x)
        ψ = vmap(grad(basis_fn, 0), (0, None))(x, 36)
        npt.assert_allclose(ψ, ψ_ref, atol=1e-6)
        
    def test_zernike(self):

        npt.assert_allclose(_get_radial_zernike_coeff(3, 6, 5), 0   ,atol=1e-16)  # odd
        npt.assert_allclose(_get_radial_zernike_coeff(0, 10, 5), 0   ,atol=1e-16)  # odd
        npt.assert_allclose(_get_radial_zernike_coeff(1, 12, 5), 0   ,atol=1e-16)  # odd
        # R26(r) = 15*r^6 - 20*r^4 + 6*r^2
        npt.assert_allclose(_get_radial_zernike_coeff(0, 6, 2), 15  ,atol=1e-16)
        npt.assert_allclose(_get_radial_zernike_coeff(1, 6, 2), -20 ,atol=1e-16)
        npt.assert_allclose(_get_radial_zernike_coeff(2, 6, 2), 6   ,atol=1e-16)
        npt.assert_allclose(_get_radial_zernike_coeff(3, 6, 2), 0   ,atol=1e-16)
        # R15(r) = 10r^5 - 12r^3 + 3r
        npt.assert_allclose(_get_radial_zernike_coeff(0, 5, 1), 10  ,atol=1e-16)
        npt.assert_allclose(_get_radial_zernike_coeff(1, 5, 1), -12 ,atol=1e-16)
        npt.assert_allclose(_get_radial_zernike_coeff(2, 5, 1), 3   ,atol=1e-16)
        
        # check orthogonality
        