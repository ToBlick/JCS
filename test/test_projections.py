import unittest
from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.operators import curl, div
from mhd_equilibria.quadratures import *
from mhd_equilibria.vector_bases import *
import numpy.testing as npt
from jax import numpy as jnp
import jax
import quadax as quad
jax.config.update("jax_enable_x64", True)

class ProjectionTests(unittest.TestCase):
    
    def test_inprod(self):
        def f(x):
            return jnp.dot(x,x)
        def g(x):
            return x[1]
    
        L = 1.0
        
        q = quad.GaussKronrodRule(31, 2)
        wh_scaled = q._wh * (1 - 0) / 2
        # wl_scaled = q._wl * (1 - 0) / 2
        xh_scaled = (q._xh + 1) / 2 * (1 - 0) + 0
        
        nx = xh_scaled.size
        wq_x = wh_scaled
        _x = xh_scaled
        x = jnp.array(jnp.meshgrid(_x, _x, _x)) # shape 3, nx, nx, nx
        x = x.transpose(1, 2, 3, 0).reshape(nx**3, 3)
        w = jnp.array(jnp.meshgrid(wq_x, wq_x, wq_x)).transpose(1, 2, 3, 0).reshape(nx**3, 3)
        w = jnp.prod(w, 1)
             
        npt.assert_allclose(l2_product(f, g, x, w), 7/12, rtol=1e-6)
        
        def F(x):
            return x
        def G(x):
            return x
        npt.assert_allclose(l2_product(F, G, x, w), 1.0, rtol=1e-6)
        
        def f(x):
            return x[0]**3 * jnp.sin(2*jnp.pi*x[1]/L) + jnp.sin(4*jnp.pi*x[2]/L) + 4 * x[0]**2 * jnp.sin(8*jnp.pi*x[2]/L)
        
        n1, n2, n3 = 4, 6, 8
        _bases = (get_legendre_fn_x(n1, 0, 1), get_trig_fn(n2, 0, 1), get_trig_fn(n3, 0, 1))
        shape = (n1, n2, n3)
        basis_fn = get_tensor_basis_fn(_bases, shape) # basis_fn(x, k)
        
        l2_proj = get_l2_projection(basis_fn, x, w, n1*n2*n3)
        f_hat = l2_proj(f)
        # f_hat = vmap(lambda i: l2_product(f, lambda x: basis_fn(x, i), x, w), 0)(jnp.arange(n1*n2*n3))
        f_h = get_u_h(f_hat, basis_fn)
        
        # plt.plot(f_hat)
        # plt.show()
        
        print(jax.jacfwd(f_h)(jnp.array([0.5, 0.5, 0.5])))
        print(jax.jacfwd(f)(jnp.array([0.5, 0.5, 0.5])))
        
        @jit
        def l2_err(x):
            return (f(x) - f_h(x))**2
        @jit
        def h1_err(x):
            return jnp.dot(jax.jacfwd(f)(x) - jax.jacfwd(f_h)(x), jax.jacfwd(f)(x) - jax.jacfwd(f_h)(x))
        
        # f_slice = lambda x: f(jnp.array([x, x, x]))
        # f_h_slice = lambda x: f_h(jnp.array([x, x, x]))   
        # plt.plot(_x, vmap(f_slice)(_x), label='f')
        # plt.plot(_x, vmap(f_h_slice)(_x), label='f_h')
        # plt.legend()
        # plt.show()
        
        npt.assert_allclose(jnp.sqrt(integral(l2_err, x, w)), 0, atol=1e-6)
        npt.assert_allclose(jnp.sqrt(integral(h1_err, x, w)), 0, atol=1e-6)
        
    def test_zernike(self):
        # project Gaussian onto Zernike polynomials
        def f(x):
            r, θ = x 
            return jnp.exp(-(r)**2/(2 * 0.5**2))
        def f_3d(x):
            r, θ, z = x
            return f(jnp.array([r, θ]))
        
        Omega = ((0, 1), (0, 2*jnp.pi), (0, 2*jnp.pi))
        x_q, w_q = quadrature_grid(get_quadrature_spectral(31)(*Omega[0]),
                                   get_quadrature_periodic(64)(*Omega[1]),
                                   get_quadrature_periodic(1)(*Omega[2]))
        
        _n = 8
        n_r = _n
        n_θ = _n
        n_φ = 1
        bases = (get_zernike_fn_x(n_r*n_θ, *Omega[0], *Omega[1]), get_trig_fn(n_φ, *Omega[2]))
        shape = (n_r*n_θ, n_φ)
        basis_fn = jit(get_zernike_tensor_basis_fn(bases, shape))
        N = n_r*n_θ*n_φ

        l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
        f_hat = l2_proj(lambda x: f_3d(x) * x[0])
        basis_fns = (basis_fn, basis_fn, basis_fn)
        f_h = get_u_h(f_hat, basis_fn)

        def err(x):
            return (f_h(x) - f_3d(x))**2 * x[0]

        def normalization(x):
            return (f_3d(x))**2 * x[0]
    
        error = ( jnp.sqrt( integral(err, x_q, w_q)) 
                  / jnp.sqrt( integral(normalization, x_q, w_q)) )

        npt.assert_allclose(error, 0, atol=5e-3)