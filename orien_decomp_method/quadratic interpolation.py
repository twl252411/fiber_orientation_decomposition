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


def t6_interpolate(p, a, b, c, values):
    """
    Perform quadratic T6 interpolation at a given point.

    Parameters
    ----------
    p : array_like, shape (2,)
        Coordinates of the interpolation point.
    a, b, c : array_like, shape (2,)
        Coordinates of the triangle vertices.
    values : array_like, shape (6, ...)
        Field values at the six T6 nodes, ordered as:
        [A, B, C, AB_mid, BC_mid, CA_mid].

        The values may be scalar, vector, or tensor quantities.

    Returns
    -------
    interpolated_value : ndarray or float
        Interpolated value of the field at point p.
    """
    l1, l2, l3 = barycentric_coords(p, a, b, c)
    N = t6_shape_functions(l1, l2, l3)
    return np.tensordot(N, values, axes=(0, 0))


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