import jax.numpy as jnp

__all__ = [
    "get_cuboid_map",
    "get_cuboid_map_inverse",
    "get_cylinder_map",
    "get_cylinder_map_inverse",
    "get_torus_map"
]

# See Holderied, pg 56ff

### Cuboid

def get_cuboid_map(a, R0):
    def F(x):
        r, χ, ζ = x
        return jnp.array([a * r, 
                          2 * jnp.pi * a * χ, 
                          2 * jnp.pi * R0 * ζ])
    return F

def get_cuboid_map_inverse(a, R0):
    def F_inv(x):
        x1, x2, x3 = x
        return jnp.array([x1 / a, 
                          x2 / (2 * jnp.pi * a), 
                          x3 / (2 * jnp.pi * R0)])
    return F_inv

### Cylindrical coordinates
def get_cylinder_map(a, R0):
    def F(x):
        r, χ, ζ = x
        return jnp.array([R0 + a * r * jnp.cos(2 * jnp.pi * χ), 
                          a * r * jnp.sin(2 * jnp.pi * χ), 
                          2 * jnp.pi * R0 * ζ])
    return F

def get_cylinder_map_inverse(a, R0):
    def F_inv(x):
        x1, x2, x3 = x
        r = jnp.sqrt((x1 - R0)**2 + x2**2) / a
        χ = jnp.arctan2(x2, x1 - R0) / (2 * jnp.pi)
        ζ = x3 / (2 * jnp.pi * R0)
        return jnp.array([r, χ, ζ])
    return F_inv

### Tokamak coordinates

def get_torus_map(a, R0):
    def F(x):
        r, χ, ζ = x
        
        s = a * r / R0
        θ = 2 * jnp.arctan2( jnp.sqrt((1 + s)/(1 - s)) * jnp.tan(jnp.pi * χ) )
        
        R = R0 + a * r * jnp.cos(θ)
        Y = a * r * jnp.sin(θ)
        
        return jnp.array([R * jnp.cos(2 * jnp.pi * ζ),
                          Y,
                          R * jnp.sin(2 * jnp.pi * ζ)])

#TODO: Inverse?