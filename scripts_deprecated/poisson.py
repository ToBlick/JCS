# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, grad, jit

domain = ((0, 1), (0, 1))
d = 2 # dimension
n = 16 # number of basis functions per dimension
N = n**d # total number of basis functions
n_s = jnp.arange(N) # number of basis functions (linear indexing)
nx = 128 # number of points for plotting

# source term
def p(x):
    return jnp.cos(3 * x[0] * jnp.pi) * jnp.cos(2 * x[1] * jnp.pi)

# 1D basis function ψ number i at point x
# Neumann boundary conditions: ∇u(x)·n(x)  = 0 for x on boundary 
# -> only cosines, set the undefined constant to zero.
# normed such that ∫ ψ_i * ψ_i = 1 for all i=j and 0 otherwise
def basis_fn_x(x, i):
    return jnp.sqrt(2) * jnp.cos(jnp.pi * i * x)
    # sreturn 1/(i+1) * x**i
    
# converts a linear index i to a cartesian index (i,j)
def lin_to_cart(i):
    ci = jnp.zeros(d, dtype=int)
    for dim in range(d-1, -1, -1):
        ci = ci.at[dim].set(i % n)
        i = i // n
    return ci

# computes the ith basis function as a cartesian product of 1D basis functions
# for example, for d=2 and n=2, the basis functions are
# ϕ_0(x) = ψ_0(x_1) * ψ_0(x_2)
# ϕ_1(x) = ψ_0(x_1) * ψ_1(x_2)
# ϕ_2(x) = ψ_1(x_1) * ψ_0(x_2)
# ϕ_3(x) = ψ_1(x_1) * ψ_1(x_2)
def basis_fn(x, i):
    ci = lin_to_cart(i)
    # x is a vector of length d
    # i a vector of length d, all entries between 0 and n
    # get the vector (ψ_i1(x_1), ψ_i2(x_2), ..., ψ_id(x_d))
    basis_fns = vmap(basis_fn_x)(x, ci)
    # ϕ(x) = ψ_i1(x_1) * ψ_i2(x_2) * ... * ψ_id(x_d)
    return jnp.prod(basis_fns)
basis_fn_v = jit(vmap(basis_fn, (0, None)))
# vmap -> vectorize, jit -> compile

# function to evaluate the basis functions at a point x 
# with given degrees of freedom phi_hat
def phi(x, phi_hat):
    basis_at_x = vmap(basis_fn, (None, 0))(x, n_s)
    return jnp.dot(phi_hat, basis_at_x)
phi_v = jit(vmap(phi, (0, None)))


#%%
# plot a few of them
x_1 = jnp.linspace(domain[0][0], domain[0][1], nx)
x_2 = jnp.linspace(domain[1][0], domain[1][1], nx)
x = jnp.array(jnp.meshgrid(x_1, x_2)).reshape(d, nx**2).T

# quadrature points
h1 = (domain[0][1] - domain[0][0]) / (nx - 1)
h2 = (domain[1][1] - domain[1][0]) / (nx - 1)
x_q_1 = jnp.linspace(domain[0][0] + h1/2, domain[0][1] - h1/2, (nx-1))
x_q_2 = jnp.linspace(domain[1][0] + h2/2, domain[1][1] - h2/2, (nx-1))
x_q = jnp.array(jnp.meshgrid(x_q_1, x_q_2)).reshape(d, (nx-1)**2).T
# x_q.shape is now ((nx-1)^2, d)

# ϕ_(n-1)(x) = cos((n-1) * pi * x_1) * cos(0 * pi * x_2) / 2
phi_hat = jnp.zeros(N).at[n-1].set(1)
plt.contourf(x_1, x_2, phi_v(x, phi_hat).reshape(nx, nx), levels=50)
plt.colorbar()
# %%
# ϕ_(2*n+3)(x) = cos(3 * pi * x_1) * cos(3 * pi * x_2) / 2
phi_hat = jnp.zeros(N).at[2*n+3].set(1)
plt.contourf(x_1, x_2, phi_v(x, phi_hat).reshape(nx, nx), levels=50)
plt.colorbar()
# %%

### Jax magic
@jit
def laplace_phi(x, phi_hat):
    # take derivative only with respect to x
    H = jax.hessian(phi, argnums=0)(x, phi_hat)
    return jnp.trace(H)
laplace_phi_v = jit(vmap(laplace_phi, in_axes=(0, None)))

def grad_phi(x, phi_hat):
    # take derivative only with respect to x
    return jax.grad(phi, argnums=0)(x, phi_hat)
grad_phi_v = jit(vmap(grad_phi, in_axes=(0, None)))

# %%
# First check: take the Laplacian of a basis function
# Δϕ_(2*n+3)(x) = Δ (cos(3 * pi * x_1) * cos(4 * pi * x_2)) / 2
#               = - (9 + 16) * pi^2 * cos(3 * pi * x_1) * cos(4 * pi * x_2) / 2
phi_hat = jnp.zeros(N).at[2*n+3].set(1)
plt.contourf(x_1, x_2, laplace_phi_v(x, phi_hat).reshape(nx, nx), levels=50)
plt.colorbar()
# print(
#     jnp.sum((laplace_phi_v(x_q, phi_hat) -
#              - 25 * jnp.pi**2 * phi_v(x_q, phi_hat))**2) / (nx-1)**2
# )
# %%

# L2 projection onto the space spanned by the basis functions
# min_w_i ∫ (f(x) - ∑ w_i ϕ_i(x) dx)²
# => ∫ (f(x) - ∑ w_i ϕ_i(x)) * ϕ_j(x) dx = 0 for all j
# since ∫ ϕ_i(x) * ϕ_j(x) dx = 0 for i != j, we get
# ∫ f(x) * ϕ_j(x) dx = w_j

def l2_product(f, g, x):
    # inner product with the ith basis function
    # ∫ f(x) * ϕ_i(x) dx
    return jnp.sum(vmap(f)(x) * vmap(g)(x)) / x.shape[0]

def l2_basis_i(f, i, x):
    # compute the inner product of f with the ith basis function
    return l2_product(f, lambda x: basis_fn(x, i), x)

def get_phi_hat(f, x):
    # compute the inner product of f with all basis functions
    return vmap(l2_basis_i, (None, 0, None))(f, n_s, x)
    
# check for p:
p_hat = get_phi_hat(p, x_q)
print(
    jnp.sum((vmap(p)(x_q) - phi_v(x_q, p_hat))**2) / (nx-1)**2
)

# %%

# Solve the Poisson equation
# 1) By assembling the stiffness matrix
# Δu = p
# => ∫ Δu(x) * ϕ_i(x) dx = ∫ p(x) * ϕ_i(x) dx for all i.
# Substitute u = ∑ w_j * ϕ_j:
# ∑ w_j ∫ Δϕ_j(x) * ϕ_i(x) dx = ∫ p(x) * ϕ_i(x) dx
# Computing Δϕ_j is too expensive, so instead compute 
# -∫ ∇ϕ_j(x)·∇ϕ_i(x) dx
# No boundary term because of the Neumann boundary conditions
# => ∑_j K_ij w_j = p_hat_i

def _K(i,j,x):
    # compute the inner product of the Laplacian of the ith basis function
    # with the jth basis function
    gradϕ_j = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, j)
    gradϕ_i = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, i)
    return -jnp.sum(gradϕ_i * gradϕ_j) / x.shape[0]

# assemble
K = vmap(vmap(_K, (0, None, None)), (None, 0, None))(n_s, n_s, x_q)
plt.imshow(K)
plt.colorbar()
# Note: Diagonal matrix, why?


# %%
# K[0,0] is zero, this corresponds to the un-defined constant.
# To set it to zero, enforce w[0] = 0
K = K.at[0,0].set(1)
print("condition of K: ", jnp.linalg.cond(K))
K = K.at[0,0].set(1)
p_hat = get_phi_hat(p, x_q)
p_hat = p_hat.at[0].set(0)

# Solve the system for w:
w = jnp.linalg.solve(K, p_hat)

u_h = vmap(lambda x: phi(x, w))

# plot the solution and print error
plt.contourf(x_1, x_2, u_h(x).reshape(nx, nx), levels=50)
plt.colorbar()
print(
    "u_h error calculated with direct solve: ",
    jnp.sum((u_h(x_q) -
             - 1/(25 * jnp.pi**2) * phi_v(x_q, p_hat))**2) / (nx-1)**2
)
# %%

# Solve the Poisson equation
# 2) By minimizing the residual
def residual(w, p, x):
    w = w.at[0].set(0) # hard-code the constant in
    u_h = lambda x: phi(x, w)
    def laplace_u_h(x):
        H = jax.hessian(u_h)(x)
        return jnp.trace(H)
    laplace_u_h_at_x = vmap(laplace_u_h)(x)
    # grad_u_h_at_x = vmap(grad(u_h))(x)
    # grad_u_h_sq = jnp.sum(grad_u_h_at_x**2, axis=1)
    p_at_x = vmap(p)(x)
    
    # we do not need to consider the boundary condition 
    # because our basis functions automatically satisfies it
    # normally one would add a penalization term or something
    # (that is commonly referred to as a PINN)
    
    return jnp.sum( (laplace_u_h_at_x - p_at_x)**2 ) / x.shape[0]
    # return jnp.sum( (- grad_u_h_sq - p_at_x)**2 ) / x.shape[0]

# sanity check: the gradient of the residual at the solution should be zero

print( "Gradient of the residual at the solution: ",
      jnp.sum ((jax.grad(residual, argnums=0)(w, p, x_q)**2) ) / w.shape[0] 
      )

# %%

import optax
from jax import value_and_grad

solver = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(
                        max_backtracking_steps=50,
                        store_grad=True
                        )
                    )
# have to jit this for performance
@jit
def objective(params):
    return residual(params, p, x_q)
# initial guess, can take the L2 projection
params = get_phi_hat(p, x).at[0].set(0)
opt_state = solver.init(params)
value_and_grad = optax.value_and_grad_from_state(objective)

# optimization loop 
# (this takes forever because of the laplacian, we only do 100 iterations)
for _ in range(100):
    value, grad = value_and_grad(params, state=opt_state)
    if jnp.sum( grad**2 ) < 1e-6:
        break
    updates, opt_state = solver.update(
        grad, opt_state, params, value=value, 
        grad=grad, value_fn=objective
    )
    params = optax.apply_updates(params, updates)

# %%
# plot this solution
u_h_optax = vmap(lambda x: phi(x, params))
plt.contourf(x_1, x_2, u_h_optax(x).reshape(nx, nx), levels=50)
plt.colorbar()
print(
    "u_h error calculated with least squares: ",
    jnp.sum((u_h_optax(x_q) -
             - 1/(25 * jnp.pi**2) * phi_v(x_q, p_hat))**2) / (nx-1)**2
)

# %%
# Do a more interesting problem (a few localized sources)
key = jax.random.PRNGKey(1)
def q(x):
    nc = 8 # has to be even
    cs = jax.random.uniform(key, (nc, 2), minval=0.1, maxval=0.9)
    r = 0
    # could vmap this to parallelize... but ok.
    i = 0
    for c in cs:
        r += (-1)**(i%2) * jnp.exp(- 1/(2 * 0.001) * ((x[0] - c[0])**2 + (x[1] - c[1])**2))
        i += 1
    return r
plt.contourf(x_1, x_2, vmap(q)(x).reshape(nx, nx), levels=50)
plt.colorbar()

# # %%
# # project the source term
# q_hat = get_phi_hat(q, x_q)
# # set the undefined constant to zero
# q_hat = q_hat.at[0].set(0)
# # solve the system
# w = jnp.linalg.solve(K, q_hat)
# # plot
# u_h = vmap(lambda x: phi(x, w))
# plt.contourf(x_1, x_2, u_h(x).reshape(nx, nx), levels=50)
# plt.colorbar()
# # %%

# def get_random_function(key, smoothness):
#     coeffs = jax.random.normal(key, (N,))
#     lin_is = jnp.arange(N)
#     k_sqd = (lin_is + 1)**2
#     psi_hat = coeffs / (k_sqd**(smoothness/2))
#     psi_hat = psi_hat.at[0].set(0)
    
#     def basis_fn_x(x, i):
#         return jnp.sqrt(2) * jnp.cos(jnp.pi * i * x)
    
#     def psi(x):
#         i_s = jnp.arange(N)
#         psi_i = vmap(basis_fn_x, (None, 0))(x, i_s)
#         return jnp.sum(psi_hat * psi_i)
#     return psi

# key = jax.random.PRNGKey(1)
# f = get_random_function(key, 1)
# n_fine = 1000
# h_fine = 1/n_fine
# x_fine = jnp.linspace(h_fine/2, 1-h_fine/2, 1000)

# def get_f_hat(f, i):
#     return h_fine * jnp.sum(vmap(f)(x_fine) * basis_fn_x(x_fine, i))

# f_hat = vmap(lambda i: get_f_hat(f, i))(jnp.arange(N))

# def f_h(x, f_hat):
#     i_s = jnp.arange(len(f_hat))
#     psi_i = vmap(basis_fn_x, (None, 0))(x, i_s)
#     return jnp.sum(f_hat * psi_i)

# # %%
# plt.plot(x_fine, vmap(f)(x_fine), label="f(x)")
# plt.legend()
# plt.xlabel("x")
# for i in range(10):
#     plt.plot(x_fine, vmap(f_h, (0, None))(x_fine, f_hat[:i+1]), label=f"f_h_N={i}(x)")
# # %%
# plt.plot(x_fine, vmap(f_h, (0, None))(x_fine, f_hat))
# # %%
# plt.plot(x_fine, vmap(f)(x_fine), label="f(x)")
# plt.legend()
# plt.xlabel("x")
# # %%

# import ipywidgets as widgets
# from IPython.display import display

# def update(i):
#     plt.plot(x_fine, vmap(f)(x_fine), label="f(x)")
#     plt.plot(x_fine, vmap(f_h, (0, None))(x_fine, f_hat[:i+1]), label=f"f_h(x) with N={i}")
#     plt.legend()
#     plt.xlabel("x")
#     plt.ylim(-2,2)
#     plt.show()
    
# i_slider = widgets.IntSlider(
#     value=10,
#     min=0,    
#     max=8*N//10,   
#     description='N',
#     continuous_update=True
# )

# interactive = widgets.interactive(update, i=i_slider)

# display(interactive)
# %%
