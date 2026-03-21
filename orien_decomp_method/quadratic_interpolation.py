"""
triangle_quadratic_interpolation.py
==================================

Quadratic (second-order) interpolation utilities for 2D triangular elements.

This module implements area-coordinate-based quadratic interpolation for
six-node triangular (T6) elements. Given the coordinates of the three
triangle vertices and an arbitrary point inside (or outside) the triangle,
the module evaluates barycentric coordinates, quadratic shape functions,
and performs interpolation of scalar, vector, or tensor-valued fields.

The implementation is independent of any finite element framework and is
suitable for post-processing, homogenization, field reconstruction, and
micromechanics applications where higher-order spatial interpolation is
required.

Typical applications include:
    - Quadratic field reconstruction on unstructured triangular meshes
    - Post-processing of FEM or FFT-based homogenization results
    - Interpolation of stress, strain, or orientation tensors
    - RVE-scale field evaluation at arbitrary spatial locations

Mathematical background:
    The interpolation is based on area (barycentric) coordinates and the
    standard six-node quadratic triangular (T6) shape functions, which
    provide second-order completeness and ensure exact reproduction of
    quadratic fields.

References:
    - Zienkiewicz, O. C., & Taylor, R. L. (2000).
      *The Finite Element Method*, Vol. 1. Butterworth-Heinemann.
    - Bathe, K. J. (1996).
      *Finite Element Procedures*. Prentice Hall.
    - Hughes, T. J. R. (1987).
      *The Finite Element Method*. Dover.

Author:
    Wenlong Tian
    Northwestern Polytechnical University

Date:
    2025-11-11
"""

import numpy as np


def area(p, q, r):
    """
    Compute the signed area of a triangle defined by three points.

    Parameters
    ----------
    p, q, r : array_like, shape (2,)
        Coordinates of the three points.

    Returns
    -------
    float
        Signed area of the triangle (p, q, r). The sign depends on the
        orientation of the point ordering.
    """
    return 0.5 * (
        (q[0] - p[0]) * (r[1] - p[1])
        - (q[1] - p[1]) * (r[0] - p[0])
    )


def barycentric_coords(p, a, b, c):
    """
    Compute barycentric (area) coordinates of a point in a triangle.

    Parameters
    ----------
    p : array_like, shape (2,)
        Coordinates of the interpolation point.
    a, b, c : array_like, shape (2,)
        Coordinates of the triangle vertices.

    Returns
    -------
    l1, l2, l3 : float
        Barycentric (area) coordinates corresponding to vertices
        a, b, and c, respectively.
    """
    A = area(a, b, c)
    l1 = area(p, b, c) / A
    l2 = area(p, c, a) / A
    l3 = area(p, a, b) / A
    return l1, l2, l3


def t6_shape_functions(l1, l2, l3):
    """
    Evaluate quadratic T6 shape functions.

    Parameters
    ----------
    l1, l2, l3 : float
        Barycentric coordinates of the evaluation point.

    Returns
    -------
    N : ndarray, shape (6,)
        Quadratic shape function values corresponding to the
        three vertices and three edge-midpoint nodes of a T6 element.
    """
    N = np.empty(6)
    N[0] = l1 * (2.0 * l1 - 1.0)
    N[1] = l2 * (2.0 * l2 - 1.0)
    N[2] = l3 * (2.0 * l3 - 1.0)
    N[3] = 4.0 * l1 * l2
    N[4] = 4.0 * l2 * l3
    N[5] = 4.0 * l3 * l1
    return N


def t6_interpolate(p):
    """
    Perform quadratic T6 interpolation at a given point.

    Parameters
    ----------
    p : array_like, shape (2,)
        Coordinates of the interpolation point.
    a, b, c : array_like, shape (2,)
        Coordinates of the triangle vertices.

    Returns
    -------
    interpolated_value : ndarray or float
        Interpolated value of the field at point p.
    """
    a = np.array([1.0, 0.0])
    b = np.array([0.5, 0.5])
    c = np.array([1/3, 1/3])

    l1, l2, l3 = barycentric_coords(p, a, b, c)
    N = t6_shape_functions(l1, l2, l3)
    return N

def piecewise_linear_interp(p):
    """
    Piecewise-linear interpolation weights for a point inside a T6 triangle
    subdivided into 4 linear sub-triangles.

    Parameters
    ----------
    p : array_like, shape (2,)
        Coordinates of the interpolation point.
    node_coords : ndarray, shape (6,2)
        Coordinates of T6 nodes in order: [A,B,C,D,E,F]
        where A,B,C are vertices, D,E,F are edge midpoints (AB, BC, CA).

    Returns
    -------
    weights : ndarray, shape (3, )
        Linear interpolation weights for the sub-triangle vertices.
    tri_nodes : ndarray, shape (3, )
        Indices of the T6 nodes forming the sub-triangle used for interpolation.
    """
    node_coords = np.array([[1.0, 0], [0.5, 0.5], [1/3, 1/3], [3/4, 1/4], [5/12, 5/12], [2/3, 1/6]])

    # Node indices
    A, B, C, D, E, F = range(6)

    # Define the 4 sub-triangles in terms of T6 node indices
    sub_tris = [
        [A, D, F],  # lower-left
        [D, B, E],  # lower-right
        [F, E, C],  # top
        [D, E, F]   # center
    ]

    for tri in sub_tris:
        a, b, c = [node_coords[i] for i in tri]
        l1, l2, l3 = barycentric_coords(p, a, b, c)

        if np.all(np.array([l1, l2, l3]) >= -1e-12):
            weights = np.array([l1, l2, l3])
            return weights, np.array(tri)


# ---------------------------------------------------------------------------------------------------------- #
# Function summary
# ---------------------------------------------------------------------------------------------------------- #
"""
Summary of Functions
--------------------

area(p, q, r)
    Compute the signed area of a triangle defined by three 2D points.

barycentric_coords(p, a, b, c)
    Evaluate barycentric (area) coordinates of an arbitrary point
    with respect to a triangular element.

t6_shape_functions(l1, l2, l3)
    Compute the quadratic shape functions for a six-node triangular
    (T6) finite element based on barycentric coordinates.

t6_interpolate(p, a, b, c, values)
    Perform second-order (quadratic) interpolation of scalar, vector,
    or tensor fields at a given point using T6 shape functions.
"""

# End of module triangle_quadratic_interpolation.py