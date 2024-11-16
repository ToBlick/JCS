import jax
from jax import numpy as jnp

def pullback_0form(p, F):
    def pullback(x):
        return p(F(x))
    return pullback

def pullback_1form(A, F):
    def pullback(x):
        return jax.jacfwd(F)(x).T @ A(F(x))
    return pullback

def pullback_2form(B, F):
    def pullback(x):
        J = jnp.linalg.det(jax.jacfwd(F)(x))
        return inv33(jax.jacfwd(F)(x)) @ B(F(x)) * J
    return pullback

def pullback_3form(f, F):
    def pullback(x):
        J = jnp.linalg.det(jax.jacfwd(F)(x))
        return J * f(F(x))
    return pullback

def inner_product_1form(u, v, F):
    # u and v are in logical domain
    def integrand(x):
        DF = jax.jacfwd(F)(x)
        J = jnp.linalg.det(DF)
        return u(x) @ inv33(DF.T @ DF).T @ v(x) * J
    return integrand
    
    
    

# jax does not have these hardcoded, so this is a speedup of ~30x

# TODO: who knows what is going on here
# def __inv(mat):
#     return jax.lax.cond((mat.size == 9) | ( mat.size == 4), _inv, jnp.linalg.inv, mat)
# def _inv(mat):
#     return jax.lax.cond((mat.size == 9), inv33, inv22, mat)

def inv22(mat):
    print(mat)
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    det = m1 * m4 - m2 * m3
    return jnp.array([[m4, -m2], [-m3, m1]]) / det

def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    return (
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )