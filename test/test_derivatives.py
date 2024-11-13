import unittest
from mhd_equilibria.coordinate_transforms import *
from mhd_equilibria.operators import *
from mhd_equilibria.pullbacks import *
import numpy.testing as npt
from jax import numpy as jnp
from jax import jit, grad, vmap
import jax
jax.config.update("jax_enable_x64", True)

class DerivativeTests(unittest.TestCase):
    
    def test(self):
        
        F = cyl_to_cart
        G = cart_to_cyl
        
        n = 32
        key = jax.random.PRNGKey(0)
        _x_hat = jax.random.uniform(key, (n**3, 3), 
                    minval=jnp.array((0.0, 0.0, 0.0)),
                    maxval=jnp.array((1.0, 2*jnp.pi, 1.0)))
        x = vmap(F)(_x_hat)
        
        def p_hat(x):
            r, phi, z = x
            return jnp.sin(phi)*r + z**2/2 + r*r
        p = pullback_0form(p_hat, G)
        grad_hat_p_hat = pullback_1form(grad(p), F)
        grad_p = pullback_1form(grad(p_hat), G)
        
        npt.assert_allclose(vmap(grad_hat_p_hat)(_x_hat), vmap(grad(p_hat))(_x_hat), atol=1e-12)
        npt.assert_allclose(vmap(grad_p)(x), vmap(grad(p))(x), atol=1e-12)
        
        def A_hat(x):
            r, phi, z = x
            return jnp.array([  jnp.cos(phi)*r + r*r,
                                jnp.cos(phi) + z*r,
                                r**2 + z])
        A = pullback_1form(A_hat, G)
        curl_hat_A_hat = pullback_2form(curl(A), F)
        curl_A = pullback_2form(curl(A_hat), G)
        
        npt.assert_allclose(vmap(curl_hat_A_hat)(_x_hat), vmap(curl(A_hat))(_x_hat), atol=1e-12)
        npt.assert_allclose(vmap(curl_A)(x), vmap(curl(A))(x), atol=1e-12)
        
        def B_hat(x):
            r, phi, z = x
            return jnp.array([  jnp.sin(phi)*r + z**2,
                                jnp.sin(phi) + r**2,
                                r + z * jnp.cos(phi)])
        B = pullback_2form(B_hat, G)
        div_hat_B_hat = pullback_3form(div(B), F)
        div_B = pullback_3form(div(B_hat), G)
        
        npt.assert_allclose(vmap(div_hat_B_hat)(_x_hat), vmap(div(B_hat))(_x_hat), atol=1e-12)
        npt.assert_allclose(vmap(div_B)(x), vmap(div(B))(x), atol=1e-12)


        
        