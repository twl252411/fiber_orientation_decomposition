"""
tensor_utils.py
===============

A collection of tensor operations for continuum mechanics and composite material modeling.

This module provides functions for handling 2nd- and 4th-order tensors in 3D elasticity,
including coordinate transformations, Voigt conversions, double-dot products, tensor inversion,
and generation of identity projection operators.

These operations are widely used in micromechanics (e.g., Mori–Tanaka, self-consistent, and
orientation averaging methods), finite element homogenization, and the analysis of
anisotropic or composite materials.

References:
    - Mura, T. (1987). *Micromechanics of Defects in Solids*. Springer.
    - Hill, R. (1965). "A self-consistent mechanics of composite materials." *J. Mech. Phys. Solids*, 13(4), 213–222.
    - Weng, G. J. (1990). "The theoretical connection between Mori–Tanaka and Ponte Castañeda–Willis models."
      *Mech. Mater.*, 8(2–3), 117–128.

Main Functions:
    tensor_orien_trans: Rotate a 4th-order tensor according to (θ, φ).
    tensor_eigen_trans: Transform 2nd- or 4th-order tensors via rotation matrices.
    tensor_voigt: Convert between 4th-order tensors and 6×6 Voigt matrices.
    tensor_inverse: Compute inverse of a symmetric 4th-order tensor (e.g., stiffness ↔ compliance).
    tensor_double_dot: Perform tensor double-dot operations.
    tensor_identity: Generate identity, volumetric, and deviatoric 4th-order tensors.

Author:
    [Wenlong Tian], [Northwestern Polytechnical University]
    Date: 2025-11-11
"""

import numpy as np


# ==================================================================================================================== #
def tensor_orien_trans(tensor: np.ndarray, theta: float, phi: float) -> np.ndarray:
    """
    Rotate a 4th-order stiffness tensor according to a specified orientation (theta, phi).

    This function applies a 3D rotation defined by spherical angles (θ, φ)
    to transform a 4th-order elastic tensor, often representing material stiffness.

    Args:
        tensor (np.ndarray): Input tensor, either:
            - 2nd-order tensor (3x3)
            - 4th-order tensor (3x3x3x3)
        theta (float): Polar angle (radians), measured from the z-axis.
        phi (float): Azimuthal angle (radians), measured in the x–y plane.

    Returns:
        np.ndarray: Rotated tensor with the same order as the input.

    Notes:
        The rotation follows the standard spherical coordinate convention.
        This transformation is useful for materials with preferred orientations
        such as unidirectional composites or transversely isotropic media.
    """
    # --- Rotation matrix for spherical angles --- #
    rotation = np.array([
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        [-np.sin(phi), np.cos(phi), 0],
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    ])
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        tensor_rotated = np.einsum('ai,bj,ab->ij', rotation, rotation, tensor)
    if tensor.ndim == 4 and tensor.shape == (3, 3, 3, 3):
    # --- Rotate 4th-order tensor using Einstein summation --- #
        tensor_rotated = np.einsum('ai,bj,ck,dl,abcd->ijkl', rotation, rotation, rotation, rotation, tensor)
    return tensor_rotated


# ==================================================================================================================== #
def tensor_eigen_trans(tensor: np.ndarray, eigen_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a 2nd- or 4th-order tensor using a rotation (eigen) matrix.

    Args:
        tensor (np.ndarray): Input tensor, either:
            - 2nd-order tensor (3x3)
            - 4th-order tensor (3x3x3x3)
        eigen_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotated tensor with the same order as the input.

    Raises:
        ValueError: If the input tensor is not 2nd- or 4th-order.

    Notes:
        This function performs basis transformation using the relation:
        T' = R · T · Rᵀ for 2nd-order tensors,
        and T' = R ⊗ R ⊗ R ⊗ R : T for 4th-order tensors.
    """
    eigen_matrix = np.asarray(eigen_matrix)
    if eigen_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")

    if tensor.ndim == 2 and tensor.shape == (3, 3):
        return np.einsum('ai,bj,ij->ab', eigen_matrix, eigen_matrix, tensor)
    elif tensor.ndim == 4 and tensor.shape == (3, 3, 3, 3):
        return np.einsum('ai,bj,ck,dl,ijkl->abcd', eigen_matrix, eigen_matrix, eigen_matrix, eigen_matrix, tensor)
    else:
        raise ValueError("Tensor must be 2nd- or 4th-order.")


# ==================================================================================================================== #
def tensor_voigt(tensor: np.ndarray) -> np.ndarray:
    """
    Convert between a 4th-order elasticity tensor and a 6x6 Voigt matrix.

    Args:
        tensor (np.ndarray): Input tensor or matrix:
            - 4th-order tensor (3x3x3x3) → converts to 6x6 Voigt matrix.
            - 6x6 matrix → converts to 4th-order tensor (3x3x3x3).

    Returns:
        np.ndarray: Converted representation of the tensor.

    Raises:
        ValueError: If the input does not have shape (3,3,3,3) or (6,6).

    Notes:
        - The Voigt mapping follows engineering convention (not Mandel form).
        - Shear components are included with appropriate symmetry factors.
    """
    voigt_map = {
        (0, 0): 0, (1, 1): 1, (2, 2): 2,
        (0, 1): 3, (1, 0): 3,
        (0, 2): 4, (2, 0): 4,
        (1, 2): 5, (2, 1): 5
    }

    if tensor.ndim == 4 and tensor.shape == (3, 3, 3, 3):
        mtx = np.zeros((6, 6))
        for (i, j), m in voigt_map.items():
            for (k, l), n in voigt_map.items():
                mtx[m, n] = tensor[i, j, k, l]
        return mtx

    elif tensor.ndim == 2 and tensor.shape == (6, 6):
        ten = np.zeros((3, 3, 3, 3))
        for (i, j), m in voigt_map.items():
            for (k, l), n in voigt_map.items():
                ten[i, j, k, l] = tensor[m, n]
        return ten

    else:
        raise ValueError("Input must have shape (3,3,3,3) or (6,6).")


# ==================================================================================================================== #
def tensor_inverse(inp_tensor: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a symmetric 4th-order tensor using Voigt notation.

    The tensor is converted to a 6x6 Voigt matrix, inverted, and mapped back to 4th-order form.
    Shear components are appropriately scaled using √2 factors.

    Args:
        inp_tensor (np.ndarray): Symmetric 4th-order tensor of shape (3,3,3,3).

    Returns:
        np.ndarray: Inverse 4th-order tensor of shape (3,3,3,3).

    Notes:
        - This method assumes major and minor symmetries.
        - Commonly used for compliance tensor computation from stiffness tensors.
    """
    voigt_map = {
        (0, 0): 0, (1, 1): 1, (2, 2): 2,
        (0, 1): 3, (1, 0): 3,
        (0, 2): 4, (2, 0): 4,
        (1, 2): 5, (2, 1): 5
    }
    shear_scale = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])

    voigt_tensor = np.zeros((6, 6))
    for (i, j), m in voigt_map.items():
        for (k, l), n in voigt_map.items():
            scale = shear_scale[m] * shear_scale[n]
            voigt_tensor[m, n] = inp_tensor[i, j, k, l] * scale

    voigt_inv = np.linalg.inv(voigt_tensor)

    inv_tensor = np.zeros((3, 3, 3, 3))
    for (i, j), m in voigt_map.items():
        for (k, l), n in voigt_map.items():
            scale = shear_scale[m] * shear_scale[n]
            inv_tensor[i, j, k, l] = voigt_inv[m, n] / scale

    return inv_tensor


# ==================================================================================================================== #
def tensor_double_dot(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compute the double-dot product between two tensors.

    Args:
        T1 (np.ndarray): First tensor (4th-order tensor, 3x3x3x3).
        T2 (np.ndarray): Second tensor (2nd- or 4th-order tensor).

    Returns:
        np.ndarray: Resulting tensor:
            - (4th·2nd) → 2nd-order tensor (3x3)
            - (4th·4th) → 4th-order tensor (3x3x3x3)

    Raises:
        ValueError: If the tensor orders are incompatible.
    """
    if T1.shape[-2:] != T2.shape[:2]:
        raise ValueError(f"Incompatible shapes: T1{T1.shape} and T2{T2.shape}")

    if T2.ndim == 2:
        return np.einsum('ijkl,kl->ij', T1, T2)
    elif T2.ndim == 4:
        return np.einsum('ijkl,klmn->ijmn', T1, T2)
    else:
        raise ValueError("T2 must be 2nd- or 4th-order tensor.")


# ==================================================================================================================== #
def tensor_identity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate identity and projection 4th-order tensors for 3D elasticity.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - iden_tensor (np.ndarray): Symmetric identity tensor (I₄)
            - vol_tensor (np.ndarray): Volumetric projection tensor (J₄)
            - dev_tensor (np.ndarray): Deviatoric projection tensor (K₄ = I₄ - J₄)

    Notes:
        These tensors satisfy:
            - I₄ : A = A (identity operation)
            - J₄ projects onto volumetric subspace (tr(A)/3 δᵢⱼ)
            - K₄ projects onto deviatoric subspace
    """
    eye3 = np.eye(3)

    iden_tensor = 0.5 * (
        eye3[:, None, :, None] * eye3[None, :, None, :] +
        eye3[:, None, None, :] * eye3[None, :, :, None]
    )

    vol_tensor = (1/3) * (eye3[:, :, None, None] * eye3[None, None, :, :])
    dev_tensor = iden_tensor - vol_tensor

    return iden_tensor, vol_tensor, dev_tensor


# -------------------------------------------------------------------------------------------------------------------- #
# Module summary and export control
# -------------------------------------------------------------------------------------------------------------------- #

__all__ = [
    "tensor_orien_trans",
    "tensor_eigen_trans",
    "tensor_voigt",
    "tensor_inverse",
    "tensor_double_dot",
    "tensor_identity",
]

# ---------------------------------------------------------------------------------------------------------- #
# Function summary
# ---------------------------------------------------------------------------------------------------------- #
"""
Summary of Functions
--------------------

tensor_orien_trans(tensor, theta, phi)
    Rotate a 4th-order stiffness tensor by specified polar (θ) and azimuthal (φ) angles.

tensor_eigen_trans(tensor, eigen_matrix)
    Transform a 2nd- or 4th-order tensor using a general rotation matrix (e.g., from eigenvectors).

tensor_voigt(tensor)
    Convert between full 4th-order elastic tensor (3×3×3×3) and 6×6 Voigt matrix representation.

tensor_inverse(inp_tensor)
    Compute the inverse of a symmetric 4th-order tensor using Voigt transformation.

tensor_double_dot(T1, T2)
    Perform tensor double-dot operation between 4th×2nd or 4th×4th tensors.

tensor_identity()
    Generate 4th-order symmetric identity tensor and its volumetric/deviatoric projectors.
"""

# End of module tensor_utils.py
