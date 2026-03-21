"""
orien_average.py
================

Orientation averaging utilities for fiber-reinforced composites.

This module provides functions to compute orientation-averaged 4th-order
stiffness tensors for short-fiber or unidirectional composites using
either probability density functions (PDFs) over orientation space or
structural tensor-based averaging (Advani–Tucker method).

Dependencies:
    - numpy
    - tensor_utils as tp

Author:
    [Wenlong Tian], [Northwestern Polytechnical University]
    Date: 2025-11-11
"""

import numpy as np
import tensor_utils as tu


# ==================================================================================================================== #
def orien_pdf_average(uni_tensor: np.ndarray, dangle: float = 1.0, dim: str = '3D') -> np.ndarray:
    """
    Compute the orientation-averaged 4th-order stiffness tensor using PDF integration.

    Performs numerical integration over orientation space to compute the averaged
    stiffness tensor for fibers that are either randomly oriented in 3D or
    distributed in a 2D plane.

    Args:
        uni_tensor (np.ndarray): 2th or 4th-order unidirectional ect / stiffness tensor.
        dangle (float, optional): Angular integration step in degrees (default 1°).
        dim (str, optional): Orientation space:
            - '3D': Random 3D fiber orientation (default)
            - '2D': In-plane fiber orientation (θ = π/2)

    Returns:
        np.ndarray: Orientation-averaged 4th-order etc / stiffness tensor.

    Notes:
        - Uses uniform angular sampling with appropriate weighting for spherical or planar distributions.
        - Requires `tensor_orien_trans` from tensor_utils for rotation of 4th-order tensors.
    """
    multi_tensor = np.zeros(uni_tensor.shape)

    dtheta = np.radians(dangle)
    dphi = np.radians(dangle)

    if dim.upper() == '3D':
        n_theta = int(180 / dangle) + 1

        for i in range(n_theta):
            theta = i * dtheta

            if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
                seq = np.pi * (1 - np.cos(dtheta / 2))
                n_phi = 1
            else:
                seq = 2 * np.pi / (180 / dangle) * np.sin(dtheta / 2)
                n_phi = int(np.ceil((180 / dangle) * np.sin(theta)))

            for j in range(n_phi + 1):
                phi = j * np.pi / n_phi if n_phi > 0 else 0.0
                weight = 1 / (4 * np.pi)
                tensor_rot = tu.tensor_orien_trans(uni_tensor, theta, phi)
                multi_tensor += 2 * tensor_rot * weight * seq

    elif dim.upper() == '2D':

        theta = np.pi / 2.0
        n_phi = int(180 / dangle) + 1

        for j in range(n_phi):
            phi = j * dphi
            weight = 1 / (2 * np.pi)
            tensor_rot = tu.tensor_orien_trans(uni_tensor, theta, phi)
            multi_tensor += 2 * dphi * tensor_rot * weight

    return multi_tensor


# ==================================================================================================================== #
def orien_ten_average(uni_c_tensor: np.ndarray, ori_ten_2: np.ndarray, ori_ten_4: np.ndarray) -> np.ndarray:
    """
    Compute orientation-averaged stiffness tensor using structural tensors.

    Implements Advani–Tucker type orientation averaging for short-fiber composites.

    Args:
        uni_c_tensor (np.ndarray): 4th-order stiffness tensor of unidirectional composite (3x3x3x3).
        ori_ten_2 (np.ndarray): 2nd-order orientation tensor <n ⊗ n> (3x3).
        ori_ten_4 (np.ndarray): 4th-order orientation tensor <n ⊗ n ⊗ n ⊗ n> (3x3x3x3).

    Returns:
        np.ndarray: Orientation-averaged 4th-order stiffness tensor (3x3x3x3).

    Notes:
        - Constructs effective stiffness tensor as linear combination of orientation tensors
          and Kronecker deltas with coefficients derived from unidirectional stiffness.
    """
    delta = np.eye(3)
    uni_c_tensor = tu.tensor_orien_trans(uni_c_tensor, np.pi / 2, 0)

    C1111 = uni_c_tensor[0, 0, 0, 0]
    C2222 = uni_c_tensor[1, 1, 1, 1]
    C1122 = uni_c_tensor[0, 0, 1, 1]
    C1212 = uni_c_tensor[0, 1, 0, 1]
    C2233 = uni_c_tensor[1, 1, 2, 2]

    B1 = C1111 + C2222 - 2 * C1122 - 4 * C1212
    B2 = C1122 - C2233
    B3 = C1212 + 0.5 * (C2233 - C2222)
    B4 = C2233
    B5 = 0.5 * (C2222 - C2233)

    multi_c_tensor = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    multi_c_tensor[i, j, k, l] = (
                        B1 * ori_ten_4[i, j, k, l] +
                        B2 * (ori_ten_2[i, j] * delta[k, l] + ori_ten_2[k, l] * delta[i, j]) +
                        B3 * (ori_ten_2[i, k] * delta[j, l] + ori_ten_2[i, l] * delta[j, k] +
                              ori_ten_2[j, k] * delta[i, l] + ori_ten_2[j, l] * delta[i, k]) +
                        B4 * delta[i, j] * delta[k, l] +
                        B5 * (delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k])
                    )

    return multi_c_tensor


# -------------------------------------------------------------------------------------------------------------------- #
# Module summary and export control
# -------------------------------------------------------------------------------------------------------------------- #

__all__ = [
    "orien_pdf_average",
    "orien_ten_average",
]

# ---------------------------------------------------------------------------------------------------------- #
# Function summary
# ---------------------------------------------------------------------------------------------------------- #
"""
Summary of Functions
--------------------

orien_pdf_average(uni_c_tensor, dangle=1.0, dim='3D')
    Compute orientation-averaged 4th-order stiffness tensor using PDF integration over fiber orientations.

orien_ten_average(uni_c_tensor, ori_ten_2, ori_ten_4)
    Compute orientation-averaged 4th-order stiffness tensor using structural tensor invariants (Advani–Tucker method).
"""

# End of module orien_average.py
