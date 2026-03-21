"""
mean_field_utils.py
===================

Functions for mean-field micromechanics of composites.

This module provides functions for:
    - Eshelby tensor computation for ellipsoidal inclusions
    - Phase stiffness generation for anisotropic materials
    - Two-phase composite homogenization using Mori-Tanaka or Double-Inclusion schemes
    - Orientation averaging for fiber composites
    - Closure approximations to obtain 4th-order orientation tensors

References:
    - Mura, T. (1987). *Micromechanics of Defects in Solids*. Springer.
    - Advani, S. G., Tucker, C. L. (1987). "The use of tensors to describe fiber orientation in short-fiber composites."
      *J. Rheol.*, 31(8), 751–784.

Author:
    [Wenlong Tian], [Northwestern Polytechnical University]
    Date: 2025-11-11
"""

import numpy as np
import tensor_utils as tu


# ==================================================================================================================== #
def eshelby_ellip_mech(aspect_ratio: float, poisson_matrix: float, axis: str = 'z') -> np.ndarray:
    """
    Compute the Eshelby tensor for a prolate spheroidal inclusion in an isotropic matrix.

    Args:
        aspect_ratio (float): Aspect ratio (a/c) of the ellipsoid.
        poisson_matrix (float): Poisson’s ratio of the matrix.
        axis (str, optional): Major axis orientation ('x', 'y', 'z'). Defaults to 'z'.

    Returns:
        np.ndarray: 4th-order Eshelby tensor (3x3x3x3).
    """
    ar, pvm = aspect_ratio, poisson_matrix
    eshelby = np.zeros((3, 3, 3, 3))

    t1, t2 = ar**2 - 1.0, 1.0 - pvm
    ga = ar * (ar * np.sqrt(t1) - np.arccosh(ar)) / (t1**1.5)

    # --- Major axis along z ---
    eshelby[2, 2, 2, 2] = 0.5 / t2 * ((4*ar**2-2)/t1 - 2*pvm - ga*(1-2*pvm+3*ar**2/t1))
    eshelby[0, 0, 0, 0] = eshelby[1, 1, 1, 1] = 0.25 / t2 * (3*ar**2/(2*t1) + ga*(1-2*pvm-9/(4*t1)))
    eshelby[2, 2, 0, 0] = eshelby[2, 2, 1, 1] = 0.5 / t2 * ((-ar**2)/t1 + 2*pvm + ga*(1-2*pvm+3/(2*t1)))
    eshelby[0, 0, 2, 2] = eshelby[1, 1, 2, 2] = 0.5 / t2 * ((-ar**2)/t1 + 0.5*ga*(3*ar**2/t1 -1 +2*pvm))
    eshelby[0, 0, 1, 1] = eshelby[1, 1, 0, 0] = 0.25 / t2 * (ar**2/(2*t1) - ga*(1-2*pvm+3/(4*t1)))

    shear = 0.25 / t2 * ((-2)/t1 - 2*pvm - 0.5*ga*(1-2*pvm-3*(ar**2+1)/t1))
    for (i,j) in [(2,0),(0,2),(2,1),(1,2)]:
        eshelby[i,j,i,j] = eshelby[i,j,j,i] = eshelby[j,i,i,j] = eshelby[j,i,j,i] = shear

    eshelby[0,1,0,1] = eshelby[1,0,0,1] = eshelby[0,1,1,0] = eshelby[1,0,1,0] = (eshelby[0,0,0,0]-eshelby[0,0,1,1])/2

    eshelby = 0.5*(eshelby + np.transpose(eshelby, (1,0,2,3)))
    eshelby = 0.5*(eshelby + np.transpose(eshelby, (0,1,3,2)))

    if axis.lower() in ['x','y']:
        perm = [2,0,1] if axis.lower()=='x' else [1,2,0]
        eshelby = eshelby[np.ix_(perm,perm,perm,perm)]

    return eshelby


# ==================================================================================================================== #
def eshelby_ellip_etc(aspect_ratio: float, semi_minor: float,
                      k_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Eshelby-like tensors (eshelby_s, eshelby_c, eshelby_d)
    for a prolate spheroidal inclusion aligned with the z-axis.

    This simplified version assumes:
        - The inclusion is prolate (aspect_ratio > 1).
        - The fiber (symmetry) axis is the z-axis.
        - Only diagonal components of `k_matrix` are used.

    Args:
        aspect_ratio (float):
            Aspect ratio of the inclusion (a3 / a1), where a3 is along z.
            Must satisfy aspect_ratio > 1 for prolate geometry.
        semi_minor (float):
            Semi-axis length in the transverse directions (x, y).
        k_matrix (ndarray of shape (3, 3)):
            Matrix property tensor (e.g., conductivity or stiffness).
            Only k_matrix[0,0] and k_matrix[2,2] are used for scaling.

    Returns:
        tuple of ndarray:
            - **eshelby_s** (3×3): Eshelby S-like tensor.

    Notes:
        - This implementation follows the classical Eshelby formulation
          for a spheroidal inclusion embedded in an infinite isotropic medium.
        - A small epsilon is added to denominators for numerical stability.

    References:
        Eshelby, J.D. (1957). *Proc. R. Soc. A*, 241, 376–396.
    """
    eps = 1e-16

    # Initialize tensors
    eshelby_s = np.zeros((3, 3))
    eshelby_c = np.zeros((3, 3))
    eshelby_d = np.zeros((3, 3))

    # Semi-axes
    a1 = semi_minor          # x, y
    a3 = aspect_ratio * a1   # z (fiber direction)

    # Scaling variables
    lambda1 = np.sqrt(a1**2 / (k_matrix[0, 0] + eps))
    lambda3 = np.sqrt(a3**2 / (k_matrix[2, 2] + eps))

    # Eshelby S tensor (for prolate spheroid, aspect_ratio > 1)
    numerator = (lambda3 * lambda1**2 * np.arccosh(lambda3 / (lambda1 + eps))
                 - lambda1**2 * np.sqrt(lambda3**2 - lambda1**2))
    denominator = (lambda3**2 - lambda1**2)**1.5 + eps
    eshelby_s[2, 2] = numerator / denominator
    eshelby_s[0, 0] = eshelby_s[1, 1] = (1.0 - eshelby_s[2, 2]) / 2.0

    # Eshelby C tensor (similar geometric form)
    numerator = (3.0 * a1 * a3**2 * np.arccos(a1 / (a3 + eps))
                 - 3.0 * a1**2 * np.sqrt(a3**2 - a1**2))
    denominator = 2.0 * a3 * (a3**2 - a1**2)**1.5 + eps
    eshelby_c[2, 2] = numerator / denominator
    eshelby_c[0, 0] = eshelby_c[1, 1] = (1.0 - eshelby_c[2, 2]) / 2.0

    # Eshelby D tensor
    eshelby_d[0, 0] = eshelby_c[1, 1] + eshelby_c[2, 2]
    eshelby_d[1, 1] = eshelby_c[0, 0] + eshelby_c[2, 2]
    eshelby_d[2, 2] = eshelby_c[0, 0] + eshelby_c[1, 1]

    return eshelby_s


# ==================================================================================================================== #
def phase_stiffess(e11: float, e22: float, e33: float, g12: float, g13: float, g23: float,
                   nu12: float, nu13: float, nu23: float) -> np.ndarray:
    """
    Generate 4th-order stiffness tensor for a transversely isotropic material.

    Args:
        e11, e22, e33: Young's moduli
        g12, g13, g23: Shear moduli
        nu12, nu13, nu23: Poisson ratios

    Returns:
        np.ndarray: 4th-order stiffness tensor (3x3x3x3)
    """
    compl_matrix = np.zeros((6,6))
    compl_matrix[0,0] = 1/e11
    compl_matrix[1,1] = 1/e22
    compl_matrix[2,2] = 1/e33
    compl_matrix[0,1] = compl_matrix[1,0] = -nu12/e11
    compl_matrix[0,2] = compl_matrix[2,0] = -nu13/e11
    compl_matrix[1,2] = compl_matrix[2,1] = -nu23/e22
    compl_matrix[3,3] = 1/g23
    compl_matrix[4,4] = 1/g13
    compl_matrix[5,5] = 1/g12

    stiff_voigt = np.linalg.inv(compl_matrix)
    stiff_tensor = tu.tensor_voigt(stiff_voigt)

    return stiff_tensor


# ==================================================================================================================== #
def two_phase_etc(etc_mat: np.ndarray, etc_inc: np.ndarray, inc_vf: float, inc_ar: float, semi_minor: float,
                  model_type: str = "mori-tanaka") -> np.ndarray:
    """
    Compute the effective thermal conductivity tensor of a two-phase composite
    using either the Mori–Tanaka or Double-Inclusion mean-field scheme.

    Parameters
    ----------
    ect_mat : np.ndarray (3×3)
        Thermal conductivity tensor of the matrix phase.
    ect_inc : np.ndarray (3×3)
        Thermal conductivity tensor of the inclusion phase.
    inc_vf : float
        Volume fraction of the inclusion phase.
    inc_ar : float
        Aspect ratio of the ellipsoidal inclusion (a3/a1, assumed > 1, aligned with z-axis).
    semi_minor : float
        Semi-minor axis length (a1 = a2) of the ellipsoidal inclusion.
    model_type : str, optional
        Mean-field homogenization model. Options:
        - 'mori-tanaka'
        - 'double-inclusion'
        Default is 'mori-tanaka'.

    Returns
    -------
    np.ndarray (3×3)
        Effective thermal conductivity tensor of the composite.

    Notes
    -----
    - The Eshelby tensor for an ellipsoidal inclusion embedded in an anisotropic matrix
      is computed internally by `eshelby_ellip_etc()`.
    - The implementation follows the analytical framework of the classical
      mean-field homogenization approach, assuming isotropic inclusions and matrices.
    """

    # Eshelby tensors for matrix and inclusion
    eshelby_mat = eshelby_ellip_etc(inc_ar, semi_minor, etc_mat)
    eshelby_inc = eshelby_ellip_etc(inc_ar, semi_minor, etc_inc)

    # Thermal resistivity tensors and contrast
    h_mat = np.linalg.inv(etc_mat)
    h_inc = np.linalg.inv(etc_inc)
    delta_k = etc_inc - etc_mat

    # Phase interaction tensors
    b_mat = np.linalg.inv(np.eye(3) + eshelby_mat @ h_mat @ delta_k)
    b_inc = np.eye(3) - eshelby_inc @ h_inc @ delta_k

    # Model-dependent interaction correction
    if model_type.lower() == "mori-tanaka":
        eta = 0.0
    elif model_type.lower() == "double-inclusion":
        eta = 0.5 * inc_vf * (1 + inc_vf)
    else:
        raise ValueError("model_type must be 'mori-tanaka' or 'double-inclusion'")

    # Effective interaction and homogenized property
    b_eff = np.linalg.inv((1 - eta) * np.linalg.inv(b_mat) + eta * np.linalg.inv(b_inc))
    b1_eff = b_eff @ np.linalg.inv((1 - inc_vf) * np.eye(3) + inc_vf * b_eff)
    uni_ect = etc_mat + inc_vf * delta_k @ b1_eff

    return uni_ect


# ==================================================================================================================== #
def two_phase_stiffness(stiff_mat: np.ndarray, stiff_inc: np.ndarray, inc_vf: float, aspect_ratio: float,
                        poisson_mat: float, poisson_inc: float, beta_mat: float, beta_inc: float,
                        model_type: str="mori-tanaka") -> np.ndarray:
    """
    Compute effective stiffness of a two-phase composite.

    Args:
        stiff_mat, stiff_inc: 4th-order stiffness tensors
        inc_vf: inclusion volume fraction
        aspect_ratio: inclusion aspect ratio
        poisson_matrix, poisson_inclusion: Poisson ratios
        beta_mat, beta_inc: 4th-order beta tensors, beta = stiff:cte
        model_type: 'mori-tanaka' or 'double-inclusion'

    Returns:
        np.ndarray: Effective 4th-order stiffness tensor
    """
    iden, iden_vol, iden_dev = tu.tensor_identity()

    eshelby_mat = eshelby_ellip_mech(aspect_ratio, poisson_mat)
    eshelby_inc = eshelby_ellip_mech(aspect_ratio, poisson_inc)

    b_mat = tu.tensor_inverse(iden + tu.tensor_double_dot(eshelby_mat,
        tu.tensor_double_dot(tu.tensor_inverse(stiff_mat), stiff_inc - stiff_mat)))
    b_inc = tu.tensor_inverse(iden + tu.tensor_double_dot(eshelby_inc,
        tu.tensor_double_dot(tu.tensor_inverse(stiff_inc), stiff_mat - stiff_inc)))

    if model_type.lower() == "mori-tanaka":
        eta = 0.0
    elif model_type.lower() == "double-inclusion":
        eta = 0.5 * inc_vf * (1 + inc_vf)
    else:
        raise ValueError("model_type must be 'mori-tanaka' or 'double-inclusion'")

    b_effective = tu.tensor_inverse((1-eta)*tu.tensor_inverse(b_mat) + eta*tu.tensor_inverse(b_inc))

    uni_stiffness = tu.tensor_double_dot((1-inc_vf)*stiff_mat + tu.tensor_double_dot(inc_vf*stiff_inc, b_effective),
                                        tu.tensor_inverse((1-inc_vf)*iden + inc_vf*b_effective))

    b1_effective = tu.tensor_double_dot(b_effective, tu.tensor_inverse(inc_vf * b_effective + (1 - inc_vf) * iden))
    tmp_term1 = tu.tensor_double_dot(stiff_inc - stiff_mat, b1_effective - iden)
    tmp_term2 = tu.tensor_double_dot(tmp_term1 ,tu.tensor_inverse(stiff_inc - stiff_mat))
    uni_gamma = tu.tensor_double_dot(tmp_term2, beta_inc - beta_mat)

    return uni_stiffness, uni_gamma


# ==================================================================================================================== #
def orien_closure_approx(ori_ten_2: np.ndarray, dim: str='3D', zero_p: int=None) -> np.ndarray:
    """
    Compute 4th-order orientation tensor from 2nd-order tensor using closure approximation.

    Args:
        ori_ten_2: 2nd-order orientation tensor
        dim: '1D','2D','3D'
        zero_p: zero direction index for 2D planar case

    Returns:
        np.ndarray: 4th-order orientation tensor (3x3x3x3)
    """
    if dim == '1D':

        return np.einsum('ij,kl->ijkl', ori_ten_2, ori_ten_2)

    elif dim == '2D':

        alpha, beta = -1/24, 1/6
        if zero_p is None: raise ValueError("zero_p must be specified for 2D")
        mask = np.ones(3, dtype=bool)
        mask[zero_p] = False

        ori_ten_2 = ori_ten_2[np.ix_(mask,mask)]
        delta = np.eye(2)

        ori_ten_4 = np.zeros((2,2,2,2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        term1 = beta*(ori_ten_2[i,j]*delta[k,l] + ori_ten_2[i,k]*delta[j,l] +
                                      ori_ten_2[i,l]*delta[j,k] + ori_ten_2[k,l]*delta[i,j] +
                                      ori_ten_2[j,l]*delta[i,k] + ori_ten_2[j,k]*delta[i,l])
                        term2 = alpha*(delta[i,j]*delta[k,l] + delta[i,k]*delta[j,l] +
                                       delta[i,l]*delta[j,k])
                        ori_ten_4[i,j,k,l] = term1 + term2

        ori_ten_4_full = np.zeros((3,3,3,3))
        idx = np.arange(3)
        idx = idx[idx!=zero_p]
        for ii, i in enumerate(idx):
            for jj, j in enumerate(idx):
                for kk, k in enumerate(idx):
                    for ll, l in enumerate(idx):
                        ori_ten_4_full[i,j,k,l] = ori_ten_4[ii,jj,kk,ll]

        return ori_ten_4_full

    elif dim == '3D':

        alpha, beta = -1/35, 1/7
        delta = np.eye(3)

        ori_ten_4 = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        term1 = beta*(ori_ten_2[i,j]*delta[k,l] + ori_ten_2[i,k]*delta[j,l] +
                                      ori_ten_2[i,l]*delta[j,k] + ori_ten_2[k,l]*delta[i,j] +
                                      ori_ten_2[j,l]*delta[i,k] + ori_ten_2[j,k]*delta[i,l])
                        term2 = alpha*(delta[i,j]*delta[k,l] + delta[i,k]*delta[j,l] +
                                       delta[i,l]*delta[j,k])
                        ori_ten_4[i,j,k,l] = term1 + term2

        return ori_ten_4

    else:
        raise ValueError("dim must be '1D','2D','3D'")

# -------------------------------------------------------------------------------------------------------------------- #
# Module summary and export control
# -------------------------------------------------------------------------------------------------------------------- #
__all__ = [
    "eshelby_ellip_mech",
    "eshelby_ellip_etc",
    "phase_stiffess",
    "two_phase_stiffness",
    "two_phase_etc",
    "orien_closure_approx",
]

# ---------------------------------------------------------------------------------------------------------- #
# Function summary
# ---------------------------------------------------------------------------------------------------------- #
"""
Summary of Functions
--------------------

eshelby_ellip_mech(aspect_ratio, poisson_matrix, axis='z')
    Compute 4th-order Eshelby tensor for prolate spheroidal inclusion.

eshelby_ellip_etc(aspect_ratio, semi_minor, k_matrix)
    Compute Eshelby-like tensors (S, C, D) for a spheroidal inclusion in a conductive medium.

phase_stiffess(e11, e22, e33, g12, g13, g23, nu12, nu13, nu23)
    Generate 4th-order stiffness tensor for transversely isotropic materials.

two_phase_stiffness(matrix, inclusion, volume_fraction, aspect_ratio, poisson_matrix, poisson_inclusion, model_type)
    Compute effective 4th-order stiffness tensor of a two-phase composite.

two_phase_etc(etc_mat, etc_inc, inc_vf, inc_ar, semi_minor, model_type)
    Compute effective 3×3 conductivity tensor of a two-phase composite using Mori–Tanaka or Double-Inclusion scheme.

orien_closure_approx(ori_ten_2, dim='3D', zero_p=None)
    Compute 4th-order orientation tensor using closure approximation.
"""

# End of module mean_field_utils.py
