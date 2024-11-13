import jax.numpy as jnp
import quadax as quad
import jax

def quadrature_grid(x, y, z):
    x_x, w_x = x
    x_y, w_y = y
    x_z, w_z = z
    
    x_s = [x_x, x_y, x_z]
    w_s = [w_x, w_y, w_z]
    d = 3
    N = w_x.size * w_y.size * w_z.size
    
    #TODO: could rewrite this for arbitrary dimensions some day
    x_hat = jnp.array(jnp.meshgrid(*x_s)) # shape d, n1, n2, n3, ...
    x_hat = x_hat.transpose(*range(1, d+1), 0).reshape(N, d)
    w_q = jnp.array(jnp.meshgrid(*w_s)).transpose(*range(1, d+1), 0).reshape(N, d)
    w_q = jnp.prod(w_q, 1)
        
    return x_hat, w_q
    
def get_quadrature_periodic(n):
    def _get_quadrature(a, b):
        h = (b - a) / n
        _x = jnp.linspace(a, b - h, n)
        w_q = h * jnp.ones(n)
        return _x, w_q
    return _get_quadrature

def get_quadrature(n):
    def _get_quadrature(a, b):
        q = quad.GaussKronrodRule(n, 2)
        w_q = q._wh * (b - a) / 2
        _x = (q._xh + 1) / 2 * (b - a) + a
        return _x, w_q
    return _get_quadrature