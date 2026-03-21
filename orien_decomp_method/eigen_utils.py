"""
eigen_utils.py
============

A collection of utilities for 2D triangle operations, barycentric coordinate computation,
and eigenvalue sorting for small matrices.

This module provides:
    - triangle area computation
    - barycentric coordinates relative to a reference triangle
    - eigenvalues and eigenvectors sorting

These functions are useful in micromechanics, finite element preprocessing, and
orientation averaging methods.

Author:
    [Wenlong Tian], [Northwestern Polytechnical University]
    Date: 2025-11-11
"""

import numpy as np

# ==================================================================================================================== #
def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the area of a triangle given its three vertices.

    Args:
        p1, p2, p3 (np.ndarray): Vertex coordinates, shape (2,) or (1,2).

    Returns:
        float: Area of the triangle.
    """
    p1 = np.asarray(p1).ravel()
    p2 = np.asarray(p2).ravel()
    p3 = np.asarray(p3).ravel()
    return 0.5 * abs(
        p1[0] * (p2[1] - p3[1]) +
        p2[0] * (p3[1] - p1[1]) +
        p3[0] * (p1[1] - p2[1])
    )


# ==================================================================================================================== #
def find_coefficients(pn: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of a point inside a reference triangle.

    Args:
        pn (np.ndarray): The point coordinates (2,).

    Returns:
        np.ndarray: Barycentric coordinates [alpha1, alpha2, alpha3] of pn
            with respect to the reference triangle.

    Notes:
        - Reference triangle vertices are:
            P1 = [1, 0], P2 = [0.5, 0.5], P3 = [1/3, 1/3]
        - Coordinates are computed using area ratios.
    """
    p1, p2, p3 = np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([1/3, 1/3])
    s0 = triangle_area(p1, p2, p3)
    s1 = triangle_area(pn, p2, p3)
    s2 = triangle_area(p1, pn, p3)
    s3 = triangle_area(p1, p2, pn)
    return np.array([s1 / s0, s2 / s0, s3 / s0])


# ==================================================================================================================== #
def sorted_eigens(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a 3x3 matrix and sort them in descending order.

    Args:
        matrix (np.ndarray): Square matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: (eig_vectors_sorted, eig_values_diag_sorted)
            - eig_vectors_sorted: eigenvectors as columns, sorted by descending eigenvalues.
            - eig_values_diag_sorted: diagonal matrix of sorted eigenvalues.
    """
    eig_vals, eig_vectors = np.linalg.eig(matrix)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals_sorted = np.diag(eig_vals[idx])
    eig_vectors_sorted = eig_vectors[:, idx]
    return eig_vectors_sorted, eig_vals_sorted


# -------------------------------------------------------------------------------------------------------------------- #
# Module summary and export control
# -------------------------------------------------------------------------------------------------------------------- #

__all__ = [
    "triangle_area",
    "find_coefficients",
    "sorted_eigens",
]

# ---------------------------------------------------------------------------------------------------------- #
# Function summary
# ---------------------------------------------------------------------------------------------------------- #
"""
Summary of Functions
--------------------

triangle_area(p1, p2, p3)
    Compute the area of a triangle from its three vertices.

find_coefficients(pn)
    Compute barycentric coordinates of a point with respect to a reference triangle.

sorted_eigens(matrix)
    Compute eigenvalues and eigenvectors of a 3x3 matrix and sort them in descending order.
"""

# End of module eigen_utils.py
