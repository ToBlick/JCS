from jax import grad, jacrev, jacfwd, hessian
import jax.numpy as jnp

__all__ = [
    # Cartesian coordinates
    "div",
    "curl",
    "laplacian",
    # Polar coordinates
    "polar_grad",
    "polar_div",
    "polar_rot",
    # Cylindrical coordinates
    "cyl_grad",
    "cyl_div",
    "cyl_curl",
    # Tokamak coordinates
    "get_tok_grad",
    "get_tok_div",
    "get_tok_curl",
]
### Cartesian coordinates

def div(F):
    def div_F(x):
        return jnp.trace(jacfwd(F)(x))
    return div_F

def curl(F):
    def curl_F(x):
        DF = jacfwd(F)(x)
        return jnp.array([  DF[2, 1] - DF[1, 2], 
                            DF[0, 2] - DF[2, 0], 
                            DF[1, 0] - DF[0, 1] ])
    return curl_F

def laplacian(f):
    def lap_f(x):
        return jnp.trace(hessian(f)(x))
    return lap_f

### Polar coordinates

def polar_grad(f):
    def grad_f(x):
        r, θ = x
        grad_r, grad_θ = grad(f)(x)
        return jnp.array([grad_r, grad_θ/r])
    return grad_f

def polar_div(F):
    def div_F(x):
        r, θ = x
        def _F(x):
            r, θ = x
            return F(x) * jnp.array([r, 1])
        DF = jacfwd(_F)(x)
        return jnp.sum( jnp.diagonal(DF) * jnp.array([1/r, 1/r]) )
    return div_F

def polar_rot(F):
    def rot_F(x):
        r, θ = x
        def _F(x):
            r, θ = x
            return F(x) * jnp.array([ 1, r])
        DF = jacfwd(_F)(x)
        return DF[1, 0]/r - DF[0, 1]/r
    return rot_F

### Cylindrical coordinates

def cyl_grad(f):
    def grad_f(x):
        r, θ, z = x
        grad_r, grad_θ, grad_z = grad(f)(x)
        return jnp.array([grad_r, grad_θ/r, grad_z])
    return grad_f

def cyl_div(F):
    def div_F(x):
        r, θ, z = x
        def _F(x):
            r, θ, z = x
            return F(x) * jnp.array([ r, 1, r ])
        DF = jacfwd(_F)(x)
        return jnp.trace(DF / r )
    return div_F

def cyl_curl(F):
    def curl_F(x):
        r, θ, z = x
        def _F(x):
            r, θ, z = x
            return F(x) * jnp.array([ 1, r, 1])
        DF = jacfwd(_F)(x)
        return jnp.array([  DF[2, 1]/r - DF[1, 2]/r, 
                            DF[0, 2]   - DF[2, 0], 
                            DF[1, 0]/r - DF[0, 1]/r ])
    return curl_F

### Tokamak coordinates

def get_tok_grad(R0):
    def grad_f(f):
        def _grad_f(x):
            r, theta, z = x
            R = R0 + r * jnp.cos(theta)
            return grad(f)(x) * jnp.array([1.0, 1/r, R0/R])
        return _grad_f
    return grad_f

def get_tok_div(R0):
    def div(F):
        def _div_F(x):
            def _F(x):
                r, theta, z = x
                # round-off danger here!
                R = R0 + r * jnp.cos(theta)
                return F(x) * jnp.array([r*R, R, r*R0])
            DF = jacfwd(_F)(x)
            r, theta, z = x
            R = R0 + r * jnp.cos(theta)
            return jnp.trace(DF / (r * R) )
        return _div_F
    return div

def get_tok_curl(R0):
    def curl(F):
        def _curl_F(x):
            def _F(x):
                r, theta, z = x
                R = R0 + r * jnp.cos(theta)
                return F(x) * jnp.array([1, r, R]) # Br, r Bθ, R Bz
            DF = jacfwd(_F)(x)
            r, theta, z = x
            R = R0 + r * jnp.cos(theta)
            return jnp.array([  DF[2, 1]/(r*R) - DF[1, 2]*R0/(r*R), 
                                DF[0, 2]*R0/R - DF[2, 0]/R, 
                                DF[1, 0]/r - DF[0, 1]/r ])
        return _curl_F
    return curl