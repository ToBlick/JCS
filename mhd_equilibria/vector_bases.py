import jax
import jax.numpy as jnp
from jax import jit, vmap

from mhd_equilibria.bases import get_tensor_basis_fn, get_trig_fn
from mhd_equilibria.splines import get_spline
from functools import partial

# #TODO: This feels very inelegant
# def get_scalar_basis_fn(basis, shape):
#     def basis_fn(x, i):
#         return jnp.ones(1)*(basis(x, i))
#     return basis_fn

def unravel_vector_index(I, shapes):
    """
    Unravel a linear index into a tuple of indices for a 3D vector field.
    This function takes a linear index `I` and a tuple of shapes `shapes` for 
    three components of a 3D vector field. It returns a tuple `(d, i, j, k)` 
    where `d` is the component index (0, 1, or 2) and `(i, j, k)` are the 
    unravelled indices for that component.
    Parameters:
    I (int): The linear index to unravel.
    shapes (tuple): A tuple of three shapes, each representing the shape of 
                    one component of the 3D vector field. Each shape is a 
                    tuple of three integers.
    Returns:
    tuple: A tuple `(d, i, j, k)` where `d` is the component index and 
           `(i, j, k)` are the unravelled indices for that component.
    """
    shape_1, shape_2, shape_3 = shapes
    N1 = shape_1[0] * shape_1[1] * shape_1[2]
    N2 = shape_2[0] * shape_2[1] * shape_2[2]
    
    def dijk_if_I_1(x):
        i, j, k = jnp.unravel_index(I, shape_1)
        return 0, i, j, k
    def dijk_if_I_2(x):
        i, j, k = jnp.unravel_index(I, shape_2)
        return 1, i, j, k
    def dijk_if_I_3(x):
        i, j, k = jnp.unravel_index(I, shape_3)
        return 2, i, j, k
    def dijk_if_I_1or2(x):
        return jax.lax.cond(I < N1, dijk_if_I_1, dijk_if_I_2, None)
    return jax.lax.cond(I < N1 + N2, dijk_if_I_1or2, dijk_if_I_3, None)

def ravel_vector_index(dijk, shapes):
    """
    Computes a raveled (flattened) index for a given vector component and its multi-dimensional index.
    Parameters:
    dijk (tuple): A tuple (d, i, j, k) where:
        d (int): The vector component index (0, 1, or 2).
        i (int): The index along the first dimension.
        j (int): The index along the second dimension.
        k (int): The index along the third dimension.
    shapes (tuple): A tuple of three shapes (shape_1, shape_2, shape_3) where each shape is a tuple of three integers
                    representing the dimensions of the respective component.
    Returns:
    int: The raveled index corresponding to the input multi-dimensional index and vector component.
    """
    shape_1, shape_2, shape_3 = shapes
    N1 = shape_1[0] * shape_1[1] * shape_1[2]
    N2 = shape_2[0] * shape_2[1] * shape_2[2]
    d, i, j, k = dijk
    def I_if_d_is_zero(x):
        return jnp.ravel_multi_index((i, j, k), shape_1, mode='clip')
    def I_if_d_is_one(x):
        return jnp.ravel_multi_index((i, j, k), shape_2, mode='clip') + N1
    def I_if_d_is_two(x):
        return jnp.ravel_multi_index((i, j, k), shape_3, mode='clip') + N1 + N2
    def I_if_d_is_zero_or_one(x):
        return jax.lax.cond(d == 0, I_if_d_is_zero, I_if_d_is_one, None)
    return jax.lax.cond(d == 2, I_if_d_is_two, I_if_d_is_zero_or_one, None)

def get_vector_basis_fn(bases, shape):
    """
    Creates a vector basis function that returns a 3-dimensional vector with one of its components
    set based on the provided basis functions and shape.
    Args:
        bases (tuple of callables): A list of three basis functions. Each basis function should take
                                    two arguments (x, i) and return a scalar value.
        shape (tuple of int): A tuple of three integers representing the shape of the basis functions.
    Returns:
        callable: A function that takes two arguments (x, I) and returns a 3-dimensional vector.
                  The vector will have one of its components set based on the basis functions and
                  the index I, while the other components will be zero.
    """
    def basis_fn_0(x, i):
        return jnp.zeros(3).at[0].set(bases[0](x, i))
    def basis_fn_1(x, i):
        return jnp.zeros(3).at[1].set(bases[1](x, i - shape[0]))
    def basis_fn_2(x, i):
        return jnp.zeros(3).at[2].set(bases[2](x, i - shape[0] - shape[1]))
    def basis_fn(x, I):
        return jax.lax.cond(I < shape[0], basis_fn_0, basis_fn_1_or_2, x, I)
    def basis_fn_1_or_2(x, I):
        return jax.lax.cond(I < shape[0] + shape[1], basis_fn_1, basis_fn_2, x, I)
    return basis_fn

def get_zero_form_basis(ns, ps, types, boundary):
    """
    Generates a zero-form basis for a given set of parameters.
    Parameters:
    ns (tuple): A tuple of integers (n_r, n_θ, n_ζ) representing the number of basis functions in each dimension.
    ps (tuple): A tuple of integers (p_r, p_θ, p_ζ) representing the polynomial degrees in each dimension.
    types (tuple): A tuple of strings ('type_r', 'type_θ', 'type_ζ') indicating the type of basis functions ('fourier', 'clamped', 'periodic', ...).
    boundary (tuple): A tuple of strings ('boundary_r', 'boundary_θ', 'boundary_ζ') indicating the boundary conditions ('dirichlet' or other). Only the 'dirichlet' boundary condition actually does something: It removes the first and last clamped spline from the basis.
    Returns:
    tuple: A tuple containing:
        - basis0 (function): The tensor basis function.
        - shape0 (jnp.array): An array representing the shape of the basis.
        - N0 (int): The total number of basis functions.
    """
    n_r, n_θ, n_ζ = ns
    p_r, p_θ, p_ζ = ps
    
    Omega = ((0, 1), (0, 1), (0, 1))
    
    if types[0] == 'fourier':
        basis_r = get_trig_fn(n_r, *Omega[0])
    else:
        if (boundary[0] == 'dirichlet'):
            _basis_r = get_spline(n_r, p_r, types[0])
            def basis_r(x, i):
                return _basis_r(x, i+1)
            n_r = n_r - 2
        else:
            basis_r = get_spline(n_r, p_r, types[0])
    
    if types[1] == 'fourier':
        basis_θ = get_trig_fn(n_θ, *Omega[1])
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_θ = (get_spline(n_θ, p_θ, types[1]))
            def basis_θ(x, i):
                return _basis_θ(x, i+1)
            n_θ = n_θ - 2
        else:
            basis_θ = (get_spline(n_θ, p_θ, types[1]))
        
    if types[2] == 'fourier':
        basis_ζ = get_trig_fn(n_ζ, *Omega[2])
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
            def basis_ζ(x, i):
                return _basis_ζ(x, i+1)
            n_ζ = n_ζ - 2
        else:
            basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
        
    shape0 = jnp.array([n_r, n_θ, n_ζ])
    
    basis0 = get_tensor_basis_fn(
                    (basis_r, basis_θ, basis_ζ), 
                    shape0)
    N0 = jnp.prod(shape0)
    return basis0, shape0, N0

def get_one_form_basis(ns, ps, types, boundary):
    """
    Generates a basis for one-forms given the specifications for the number of basis functions,
    polynomial degrees, types of basis functions, and boundary conditions.
    Parameters:
    ns (tuple): A tuple (n_r, n_θ, n_ζ) specifying the number of basis functions in each dimension.
    ps (tuple): A tuple (p_r, p_θ, p_ζ) specifying the polynomial degrees in each dimension.
    types (tuple): A tuple (type_r, type_θ, type_ζ) specifying the type of basis functions ('fourier' or 'spline') in each dimension.
    boundary (tuple): A tuple (boundary_r, boundary_θ, boundary_ζ) specifying the boundary conditions ('dirichlet' or other) in each dimension. Only the 'dirichlet' boundary condition actually does something: It removes the first and last clamped spline from the basis.
    Returns:
    tuple: A tuple containing:
        - basis1: The generated basis function for one-forms. Takes two arguments (x, I) and returns a 3D vector.
        - shapes1: A 2D array where each row represents the shape of the basis functions for a component of the one-form. If this is all splines, it will be ( (n_r-1, n_θ, n_ζ), (n_r, n_θ-1, n_ζ), (n_r, n_θ, n_ζ-1) ). This is different for Fourier bases: there is "reduction in size" for Fourier bases.
        - N1: The total number of basis functions, for example (n_r-1) * n_θ * n_ζ + n_r * (n_θ-1) * n_ζ + n_r * n_θ * (n_ζ-1).
    """
    n_r, n_θ, n_ζ = ns
    p_r, p_θ, p_ζ = ps
    
    Omega = ((0, 1), (0, 1), (0, 1))
    
    if types[0] == 'fourier':
        basis_r = get_trig_fn(n_r, *Omega[0])
        basis_dr = basis_r
        n_dr = n_r
    else:
        if (boundary[0] == 'dirichlet'):
            _basis_r = get_spline(n_r, p_r, types[0])
            def basis_r(x, i):
                return _basis_r(x, i+1)
            n_r = n_r - 2
        else:
            basis_r = get_spline(n_r, p_r, types[0])
        basis_dr = (get_spline(n_r - 1, p_r - 1, types[0]))
        n_dr = n_r - 1
    
    if types[1] == 'fourier':
        basis_θ = get_trig_fn(n_θ, *Omega[1])
        basis_dθ = basis_θ
        n_dθ = n_θ
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_θ = (get_spline(n_θ, p_θ, types[1]))
            def basis_θ(x, i):
                return _basis_θ(x, i+1)
            n_θ = n_θ - 2
        else:
            basis_θ = (get_spline(n_θ, p_θ, types[1]))
        basis_dθ = (get_spline(n_θ - 1, p_θ - 1, types[1]))
        n_dθ = n_θ - 1
        
    if types[2] == 'fourier':
        basis_ζ = get_trig_fn(n_ζ, *Omega[2])
        basis_dζ = basis_ζ
        n_dζ = n_ζ
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
            def basis_ζ(x, i):
                return _basis_ζ(x, i+1)
            n_ζ = n_ζ - 2
        else:
            basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
        basis_dζ = (get_spline(n_ζ - 1, p_ζ - 1, types[2]))
        n_dζ = n_ζ - 1
        
    shapes1 = jnp.array([ [n_dr, n_θ, n_ζ],
                          [n_r, n_dθ, n_ζ],
                          [n_r, n_θ, n_dζ] ])
        
    basis1_1 = get_tensor_basis_fn( (basis_dr, basis_θ, basis_ζ), shapes1[0])
    basis1_2 = get_tensor_basis_fn( (basis_r, basis_dθ, basis_ζ), shapes1[1])
    basis1_3 = get_tensor_basis_fn( (basis_r, basis_θ, basis_dζ), shapes1[2])
    
    basis1 = get_vector_basis_fn( (basis1_1, basis1_2, basis1_3), jnp.prod(shapes1, axis=1))

    N1 = jnp.sum(jnp.prod(shapes1, axis=1))
    return basis1, shapes1, N1

def get_two_form_basis(ns, ps, types, boundary):
    n_r, n_θ, n_ζ = ns
    p_r, p_θ, p_ζ = ps
    
    Omega = ((0, 1), (0, 1), (0, 1))
    
    if types[0] == 'fourier':
        basis_r = get_trig_fn(n_r, *Omega[0])
        basis_dr = basis_r
        n_dr = n_r
    else:
        if (boundary[0] == 'dirichlet'):
            _basis_r = get_spline(n_r, p_r, types[0])
            def basis_r(x, i):
                return _basis_r(x, i+1)
            n_r = n_r - 2
        else:
            basis_r = get_spline(n_r, p_r, types[0])
        basis_dr = (get_spline(n_r - 1, p_r - 1, types[0]))
        n_dr = n_r - 1
    
    if types[1] == 'fourier':
        basis_θ = get_trig_fn(n_θ, *Omega[1])
        basis_dθ = basis_θ
        n_dθ = n_θ
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_θ = (get_spline(n_θ, p_θ, types[1]))
            def basis_θ(x, i):
                return _basis_θ(x, i+1)
            n_θ = n_θ - 2
        else:
            basis_θ = (get_spline(n_θ, p_θ, types[1]))
        basis_dθ = (get_spline(n_θ - 1, p_θ - 1, types[1]))
        n_dθ = n_θ - 1
        
    if types[2] == 'fourier':
        basis_ζ = get_trig_fn(n_ζ, *Omega[2])
        basis_dζ = basis_ζ
        n_dζ = n_ζ
    else:
        if (boundary[1] == 'dirichlet'):
            _basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
            def basis_ζ(x, i):
                return _basis_ζ(x, i+1)
            n_ζ = n_ζ - 2
        else:
            basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
        basis_dζ = (get_spline(n_ζ - 1, p_ζ - 1, types[2]))
        n_dζ = n_ζ - 1
        
    shapes = jnp.array([ [n_r, n_dθ, n_dζ],
                         [n_dr, n_θ, n_dζ],
                         [n_dr, n_dθ, n_ζ] ])
        
    basis_1 = get_tensor_basis_fn( (basis_r, basis_dθ, basis_dζ), shapes[0])
    basis_2 = get_tensor_basis_fn( (basis_dr, basis_θ, basis_dζ), shapes[1])
    basis_3 = get_tensor_basis_fn( (basis_dr, basis_dθ, basis_ζ), shapes[2])
    
    basis = get_vector_basis_fn( (basis_1, basis_2, basis_3), jnp.prod(shapes, axis=1))

    N = jnp.sum(jnp.prod(shapes, axis=1))
    return basis, shapes, N

def get_three_form_basis(ns, ps, types, boundary):
    n_r, n_θ, n_ζ = ns
    p_r, p_θ, p_ζ = ps
    
    Omega = ((0, 1), (0, 1), (0, 1))
    
    if types[0] == 'fourier':
        basis_r = get_trig_fn(n_r, *Omega[0])
        basis_dr = basis_r
        n_dr = n_r
    else:
        basis_r = (get_spline(n_r, p_r, types[0]))
        basis_dr = (get_spline(n_r - 1, p_r - 1, types[0]))
        n_dr = n_r - 1
    
    if types[1] == 'fourier':
        basis_θ = get_trig_fn(n_θ, *Omega[1])
        basis_dθ = basis_θ
        n_dθ = n_θ
    else:
        basis_θ = (get_spline(n_θ, p_θ, types[1]))
        basis_dθ = (get_spline(n_θ - 1, p_θ - 1, types[1]))
        n_dθ = n_θ - 1
        
    if types[2] == 'fourier':
        basis_ζ = get_trig_fn(n_ζ, *Omega[2])
        basis_dζ = basis_ζ
        n_dζ = n_ζ
    else:
        basis_ζ = (get_spline(n_ζ, p_ζ, types[2]))
        basis_dζ = (get_spline(n_ζ - 1, p_ζ - 1, types[2]))
        n_dζ = n_ζ - 1

    shape = jnp.array([n_dr, n_dθ, n_dζ])
    
    basis = get_tensor_basis_fn(
                    (basis_dr, basis_dθ, basis_dζ), 
                    shape)
    N = jnp.prod(shape)
    return basis, shape, N