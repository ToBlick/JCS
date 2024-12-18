#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from functools import partial
import numpy.testing as npt

from mhd_equilibria.bases import *
from mhd_equilibria.splines import *

def indicator(x, i, T, p, n, m):
    return jnp.where(jnp.logical_and(T[i] <= x, x < T[i+1]), 
        1.0, 
        0.0
        )
    
def indicator_clamped(x, i, T, p, n, m):
    return jnp.where(jnp.logical_and(T[i] <= x, x < T[i+1]), 
        1.0, 
        jnp.where(jnp.logical_and(x == T[i+1], i == n-1),
            1.0,
            0.0)
        )

def safe_divide(numerator, denominator):
        return jnp.where(denominator == 0, 
            jnp.where(numerator == 0, 
                1.0, 
                0.0),
            numerator / denominator)

@partial(jax.jit, static_argnums=(3,6))
def spline(x, i, T, p, n, m, type):
    eps = 1e-16
    if p == 0:
        if type == 'clamped':
            return indicator_clamped(x, i, T, p, n, m)
        else:
            return indicator(x, i, T, p, n, m)
    else:
        w_1 = safe_divide(x - T[i] + eps,T[i + p] - T[i] + eps)
        w_2 = safe_divide(T[i + p + 1] - x + eps, T[i + p + 1] - T[i + 1] + eps)
        return w_1 * spline(x, i, T, p - 1, n, m, type) \
            + w_2 * spline(x, i + 1, T, p - 1, n, m, type)
    
# %% 
import jax.random
key = jax.random.PRNGKey(0)
n = 5
p = 4
_x = jnp.linspace(0, 1, 1000)
T = jnp.concatenate([jnp.zeros(p),
                     jnp.linspace(0, 1, n-p+1),
                     jnp.ones(p)])
m = n-p+1

# %%
### Periodic spline
T_p = jnp.concatenate([jnp.linspace(0, 1, n-p+1)[-(p+1):-1] - 1, 
                     jnp.linspace(0, 1, n-p+1), 
                     jnp.linspace(0, 1, n-p+1)[1:(p+2)] + 1])

# %%
def p_spline(i, x):
    i = i + p
    return jax.lax.cond(i > n - 2*p,
        lambda _: spline(x, i, T_p, p, n, m, 'periodic') \
                + spline(x, i - n + p, T_p, p, n, m, 'periodic'),
        lambda _: spline(x, i, T_p, p, n, m, 'periodic'),
        operand=None)
def c_spline(i, x):
    return spline(x, i, T, p, n, m, 'clamped')

for i in range(n):
    plt.plot(_x, vmap(lambda x: c_spline(i, x))(_x))

#%%
for i in range(n):
    plt.plot(_x, vmap(lambda x: p_spline(i, x))(_x))

# %%
###
# Spline regression
###

key, _ = jax.random.split(key)
m = 16
xs = jax.random.uniform(key, (m,), minval=0, maxval=0.3)
xs = jnp.sort(xs)
ys = jnp.sin(2 * jnp.pi * xs) + jax.random.normal(key, (m,), dtype=jnp.float64) * 0.1

# %%
def fit_spline(xs, ys, p, alpha):
    _xs = (xs - xs[0]) / (xs[-1] - xs[0])
    m = len(_xs)
    n = m//2
    nx = 64
    _x = jnp.linspace(0, 1, nx)
    # uniform knot vector
    T = jnp.concatenate([jnp.zeros(p),
                    jnp.linspace(0, 1, n-p+1),
                    jnp.ones(p)])
    def c_spline(x, i):
        return spline(x, i, T, p, n, n-p+1, 'clamped')
    def a(i, j):
        return c_spline(_xs[i], j)
    def b(i, j):
        return grad(grad(c_spline))(_xs[i], j)
    A = jax.vmap(lambda i: jax.vmap(lambda j: a(i, j))(jnp.arange(n)))(jnp.arange(m))
    B = jax.vmap(lambda i: jax.vmap(lambda j: b(i, j))(jnp.arange(n)))(jnp.arange(m))
    
    q = jnp.linalg.solve(A.T @ A + alpha * B.T @ B, A.T @ ys)
    def f(x):
        return jnp.sum(vmap(lambda i: q[i] * c_spline((x - xs[0])/(xs[-1] - xs[0]), i))(jnp.arange(n)))
    return f

# %%
__x = jnp.linspace(0, 0.3, 100)
cm = plt.get_cmap('viridis')
plt.scatter(xs,ys, color='red')
for i, alpha in enumerate([0.0, 1e-2, 1e-1, 1, 10]):
    plt.plot(__x, vmap(fit_spline(xs, ys, 3, alpha))(__x), color=cm(i/5))
    plt.plot(__x, vmap(grad(fit_spline(xs, ys, 3, alpha)))(__x), linestyle=':', color=cm(i/5))
plt.plot() 

# %%

f = fit_spline(xs, ys, 3, 1e-3)
g = jnp.interp(__x, xs, ys)
plt.scatter(xs,ys, color='red')
plt.plot(__x, vmap(f)(__x))
plt.plot(__x, vmap(grad(f))(__x))
plt.plot(__x, 2 * jnp.pi * jnp.cos(2 * jnp.pi * __x))
plt.plot(__x, g)
plt.plot(__x, jnp.gradient(g, __x))

# %%
### quadrature points
from mhd_equilibria.quadratures import *

x_q_1d, w_q_1d = get_quadrature_spectral(61)(0, 1)

def M_p_lazy(i, j):
    return vmap(lambda x: p_spline(i, x) * p_spline(j, x))(x_q_1d) @ w_q_1d

# %%
M_p = jax.vmap(lambda i: jax.vmap(lambda j: M_p_lazy(i, j))(jnp.arange(n-p)))(jnp.arange(n-p))

# %%
plt.imshow(M_p)
# %%
M_c_lazy = vmap(lambda i: vmap(lambda j: vmap(lambda x: c_spline(i, x) * c_spline(j, x))(x_q_1d) @ w_q_1d)(jnp.arange(n)))(jnp.arange(n))
M_c = jax.vmap(lambda i: jax.vmap(lambda j: M_c_lazy[i, j])(jnp.arange(n)))(jnp.arange(n))
plt.imshow(M_c)

# %%
f = lambda x: jnp.cos(4 * jnp.pi * x)
# f = lambda x: jnp.exp( -1/2 * (x - 0.3)**2 / 0.2**2)

type = 'clamped'

if type == 'periodic':
    basis = lambda i, x: p_spline(i, x)
    n_b = n-p
    M = M_p
else:
    basis = lambda i, x: c_spline(i, x)
    n_b = n
    M = M_c

rhs = vmap(lambda i: vmap(lambda x: f(x) * basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_b))
f_dofs = jnp.linalg.solve(M, rhs)
def f_h(x):
    return jnp.sum(vmap(lambda i: f_dofs[i] * basis(i, x))(jnp.arange(n_b)))
plt.plot(_x, f(_x))
plt.plot(_x, vmap(f_h)(_x))
print( jnp.sqrt((vmap(f_h)(x_q_1d) - f(x_q_1d))**2 @ w_q_1d) )

# %%
### Flat torus
n_r = 8
p_r = 2
n_θ = 8
p_θ = 2
n_z = 1
p_z = 0

T_r = jnp.concatenate([jnp.zeros(p_r), 
                     jnp.linspace(0, 1, n_r-p_r+1),
                     jnp.ones(p_r)])
m_r = n_r-p_r+1
T_θ = jnp.concatenate([jnp.linspace(0, 1, n_θ-p_θ+1)[-(p_θ+1):-1] - 1, 
                     jnp.linspace(0, 1, n_θ-p_θ+1),
                     jnp.linspace(0, 1, n_θ-p_θ+1)[1:(p_θ+2)] + 1])
m_θ = n_θ-p_θ+1

Omega = ((0,1), (0,1), (0,1))

# %%
@jit
def c_spline(x, i):
    return spline(x, i, T_r, p_r, n_r, m_r, 'clamped')
@jit
def dx_c_spline(x, i):
    # h = 1 #/(T_r[i + p_r + 1] - T_r[i + 1])
    # return h * spline(x, i + 1, T_r, p_r - 1, n_r, m_r, 'clamped') * p_r
    return jax.grad(c_spline)(x, i)
@jit
def p_spline(x, i):
    i = i + p_θ
    return jax.lax.cond(i > n_θ - 2*p_θ,
        lambda _: spline(x, i, T_θ, p_θ, n_θ, m_θ, 'periodic') \
                + spline(x, i - n_θ + p_θ, T_θ, p_θ, n_θ, m_θ, 'periodic'),
        lambda _: spline(x, i, T_θ, p_θ, n_θ, m_θ, 'periodic'),
        operand=None)
@jit
def dx_p_spline(x, i):
    return jax.grad(p_spline)(x, i)
    # h = 1 #/(T_θ[p_θ + i + 1] - T_θ[i + 1])
    # i = i + p_θ + 1
    # return jax.lax.cond(i > n_θ - 2*p_θ,
    #     lambda _: h * p_θ * spline(x, i, T_θ, p_θ - 1, n_θ, m_θ, 'periodic') \
    #             + h * p_θ * spline(x, i - n_θ + p_θ, T_θ, p_θ - 1, n_θ, m_θ, 'periodic'),
    #     lambda _: h * p_θ * spline(x, i, T_θ, p_θ - 1, n_θ, m_θ, 'periodic'),
    #     operand=None)
@jit
def c_basis(x, i):
    return 1.0
# %%
plt.plot(_x, vmap(lambda x: c_spline(x, 0))(_x))
plt.plot(_x, vmap(lambda x: dx_c_spline(x, 0))(_x))
# %%
x_q, w_q = quadrature_grid(get_quadrature_spectral(61)(*Omega[0]),
            get_quadrature_periodic(64)(*Omega[1]),
            get_quadrature_periodic(1)(*Omega[2]))

S_000 = get_tensor_basis_fn((c_spline, p_spline, c_basis), (n_r, n_θ - p_θ, n_z))
N_000 = n_r * (n_θ - p_θ) * n_z
S_100 = get_tensor_basis_fn((dx_c_spline, p_spline, c_basis), (n_r - 1, n_θ - p_θ, n_z))
N_100 = (n_r - 1) * (n_θ - p_θ) * n_z
S_010 = get_tensor_basis_fn((c_spline, dx_p_spline, c_basis), (n_r, n_θ - p_θ - 1, n_z))
N_010 = n_r * (n_θ - p_θ - 1) * n_z

# %%
nx = 512
_r = jnp.linspace(0, 1, nx)
_θ = jnp.linspace(0, 1, nx)
_z = jnp.array([0.0])

x = jnp.array(jnp.meshgrid(_r, _θ, _z)) # shape 3, n_x, n_x, 1
x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)
# %%
plt.contourf(_r, _θ, vmap(S_100, (0, None))(x, 32).reshape(nx, nx))
plt.xlabel('r')
plt.ylabel('θ')
plt.colorbar()
plt.title('S_100')
plt.show()

# %%
plt.contourf(_r, _θ, vmap(S_000, (0, None))(x, 5).reshape(nx, nx))
plt.xlabel('r')
plt.ylabel('θ')
plt.colorbar()
plt.title('S_000')
plt.show()

# %%
plt.contourf(_r, _θ, vmap(S_010, (0, None))(x, 4).reshape(nx, nx))
plt.xlabel('r')
plt.ylabel('θ')
plt.colorbar()
plt.title('S_010')
plt.show()

# %%
def M_000_lazy(i, j):
    return vmap(lambda x: S_000(x, i) * S_000(x, j))(x_q) @ w_q

M_000 = jax.vmap(lambda i: jax.vmap(lambda j: M_000_lazy(i, j))(jnp.arange(N_000)))(jnp.arange(N_000))

def M_100_lazy(i, j):
    return vmap(lambda x: S_100(x, i) * S_100(x, j))(x_q) @ w_q
M_100 = jax.vmap(lambda i: jax.vmap(lambda j: M_100_lazy(i, j))(jnp.arange(N_100)))(jnp.arange(N_100))

def M_010_lazy(i, j):
    return vmap(lambda x: S_010(x, i) * S_010(x, j))(x_q) @ w_q
M_010 = jax.vmap(lambda i: jax.vmap(lambda j: M_010_lazy(i, j))(jnp.arange(N_010)))(jnp.arange(N_010))
# %%
plt.imshow(M_000)
plt.colorbar()
plt.title('M_000')
plt.show()
print(jnp.linalg.cond(M_000))
# %%
plt.imshow(M_100)
plt.colorbar()
plt.title('M_100')
plt.show()
print(jnp.linalg.cond(M_100))
# %%
plt.imshow(M_010)
plt.colorbar()
plt.title('M_010')
plt.show()
print(jnp.linalg.cond(M_010))

# %%
for i in range(n_θ-p_θ):
    plt.plot(_θ, vmap(lambda x: p_spline(x, i))(_θ), linestyle='--')
for i in range(n_θ-p_θ-1):
    plt.plot(_θ, 0.1 * vmap(lambda x: dx_p_spline(x, i))(_θ))
    
# %%
plt.plot(_θ, vmap(lambda x: c_spline(x, n_r-1))(_θ), linestyle='--')
plt.plot(_θ, 0.1 * vmap(lambda x: dx_c_spline(x, n_r-1))(_θ))
# %%
for i in range(n_r):
    plt.plot(_r, vmap(lambda x: c_spline(x, i))(_r), linestyle='--')
for i in range(n_r-1):
    plt.plot(_r, 0.1 * vmap(lambda x: dx_c_spline(x, i))(_r))
# %%
def M_r_lazy(i, j):
    return vmap(lambda x: c_spline(x, i) * c_spline(x, j))(x_q_1d) @ w_q_1d
def M_θ_lazy(i, j):
    return vmap(lambda x: p_spline(x, i) * p_spline(x, j))(x_q_1d) @ w_q_1d
M_r = jax.vmap(lambda i: jax.vmap(lambda j: M_r_lazy(i, j))(jnp.arange(n_r)))(jnp.arange(n_r))
M_θ = jax.vmap(lambda i: jax.vmap(lambda j: M_θ_lazy(i, j))(jnp.arange(n_θ-p_θ)))(jnp.arange(n_θ-p_θ))
# %%
plt.imshow(M_r)
plt.colorbar()
jnp.linalg.cond(M_r)

# %%
plt.imshow(M_θ)
plt.colorbar()
jnp.linalg.cond(M_θ)

# %%
# f = lambda x: jnp.cos(4 * jnp.pi * x)
f = lambda x: jnp.exp( -1/2 * (x - 0.3)**2 / 0.2**2)

type = 'clamped'

if type == 'periodic':
    basis = lambda i, x: dx_p_spline(x, i)
    n_b = n_θ - p_θ - 1
else:
    basis = lambda i, x: dx_c_spline(x, i)
    n_b = n_r - 1

def M_lazy(i, j):
    return vmap(lambda x: basis(i, x) * basis(j, x))(x_q_1d) @ w_q_1d
M = jax.vmap(lambda i: jax.vmap(lambda j: M_lazy(i, j))(jnp.arange(n_b)))(jnp.arange(n_b))
print(jnp.linalg.cond(M))

rhs = vmap(lambda i: vmap(lambda x: f(x) * basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_b))
f_dofs = jnp.linalg.solve(M, rhs)
def f_h(x):
    return jnp.sum(vmap(lambda i: f_dofs[i] * basis(i, x))(jnp.arange(n_b)))
plt.plot(_x, vmap(f)(_x))
plt.plot(_x, vmap(grad(f))(_x))
plt.plot(_x, vmap(f_h)(_x))
plt.plot(_x, vmap(grad(f_h))(_x))
print( jnp.sqrt((vmap(f_h)(x_q_1d) - vmap(f)(x_q_1d))**2 @ w_q_1d) /
      jnp.sqrt((vmap(f)(x_q_1d))**2 @ w_q_1d))
print( jnp.sqrt((vmap(grad(f_h))(x_q_1d) - vmap(grad(f))(x_q_1d))**2 @ w_q_1d) / jnp.sqrt(vmap(grad(f))(x_q_1d)**2 @ w_q_1d))
# %%
### Project F on basis and then project the derivative of the basis:
# %%
#f = lambda x: jnp.cos(4 * jnp.pi * x)
f = lambda x: jnp.exp( -1/2 * (x - 0.7)**2 / 0.2**2)

type = 'clamped'

if type == 'periodic':
    basis = lambda i, x: p_spline(x, i)
    n_b = n_θ - p_θ
    dx_basis = lambda i, x: dx_p_spline(x, i)
    n_dx_b = n_θ - p_θ - 1
else:
    basis = lambda i, x: c_spline(x, i)
    n_b = n_r
    dx_basis = lambda i, x: dx_c_spline(x, i)
    n_dx_b = n_r - 1

def M_lazy(i, j):
    return vmap(lambda x: basis(i, x) * basis(j, x))(x_q_1d) @ w_q_1d
M = jax.vmap(lambda i: jax.vmap(lambda j: M_lazy(i, j))(jnp.arange(n_b)))(jnp.arange(n_b))
print(jnp.linalg.cond(M))
def M_dx_lazy(i, j):
    return vmap(lambda x: dx_basis(i, x) * dx_basis(j, x))(x_q_1d) @ w_q_1d
M_dx = jax.vmap(lambda i: jax.vmap(lambda j: M_dx_lazy(i, j))(jnp.arange(n_dx_b)))(jnp.arange(n_dx_b))
print(jnp.linalg.cond(M_dx))
# %%

### project first
rhs = vmap(lambda i: vmap(lambda x: f(x) * basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_b))
f_dofs = jnp.linalg.solve(M, rhs)
def f_h(x):
    return jnp.sum(vmap(lambda i: f_dofs[i] * basis(i, x))(jnp.arange(n_b)))

### differentiate first
rhs = vmap(lambda i: vmap(lambda x: grad(f_h)(x) * dx_basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_dx_b))
f_dx_dofs = jnp.linalg.solve(M_dx, rhs)
def f_dx_h(x):
    return jnp.sum(vmap(lambda i: f_dx_dofs[i] * dx_basis(i, x))(jnp.arange(n_dx_b)))

rhs = vmap(lambda i: vmap(lambda x: grad(f_h)(x) * basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_b))
f_dx_dofs_0 = jnp.linalg.solve(M, rhs)
def f_dx_h_0(x):
    return jnp.sum(vmap(lambda i: f_dx_dofs_0[i] * basis(i, x))(jnp.arange(n_b)))

plt.plot(_x, vmap(grad(f))(_x), label='grad(f)', linestyle='-')
plt.plot(_x, vmap(grad(f_h))(_x), label='grad(f_h)', linestyle='--')
plt.plot(_x, vmap(f_dx_h)(_x), label='grad(f_h) in dx_basis', linestyle='-.')
plt.plot(_x, vmap(f_dx_h_0)(_x), label='grad(f_h) in basis', linestyle=':')
plt.legend()

def error(f_h, f):
    return jnp.sqrt((vmap(f_h)(x_q_1d) - vmap(f)(x_q_1d))**2 @ w_q_1d) / jnp.sqrt((vmap(f)(x_q_1d))**2 @ w_q_1d)

print('|f - f_h|: ', error(f_h, f) )
print('|grad(f) - grad(f_h)|: ', error(grad(f_h), grad(f)) )
print('|grad(f) - grad(f_h) in dx_basis|: ', error(f_dx_h, grad(f)) )
print('|grad(f) - grad(f_h) in basis|: ', error(f_dx_h_0, grad(f)) )
print('|grad(f_h) - grad(f_h) in dx_basis|: ', error(f_dx_h, grad(f_h)))
print('|grad(f_h) - grad(f_h) in basis|: ', error(f_dx_h_0, grad(f_h)))

# %%
plt.plot(f_dx_dofs, linestyle='', marker='o')
plt.plot(f_dofs, linestyle='', marker='o')
# %%
if type == 'periodic':
    B = jnp.eye(n_θ - p_θ - 2, n_θ - p_θ, k=1)
    I = jnp.eye(n_θ - p_θ)
    basis_0 = lambda i, x: p_spline(x, i+1)
    n_b_0 = n_θ - p_θ - 2
else:
    B = jnp.eye(n_r - 2, n_r, k=1)
    I = jnp.eye(n_r)
    basis_0 = lambda i, x: c_spline(x, i+1)
    n_b_0 = n_r - 2

rhs = vmap(lambda i: vmap(lambda x: f(x) * basis(i, x))(x_q_1d) @ w_q_1d)(jnp.arange(n_b))
f_dofs = jnp.linalg.solve(B @ M @ B.T, rhs[1:-1])
# %%
def f_h(x):
    return jnp.sum(vmap(lambda i: f_dofs[i] * basis_0(i, x))(jnp.arange(n_b_0)))

plt.plot(_x, vmap(f)(_x))
plt.plot(_x, vmap(f_h)(_x))
# %%
