# ==================================================================================================== #
# multi_phase_main.py — Effective stiffness and conductivity of multi-phase composites
# ==================================================================================================== #
# ==================================================================================================== #
# multi_phase_main.py — Effective stiffness, CTE and conductivity of multi-phase composites
# ==================================================================================================== #
import numpy as np
import tensor_utils as tu
import eigen_utils as eu
import quadratic_interpolation as qi
import sys


# ---------------------------------------------------------------------------------------------------- #
# Utility functions
# ---------------------------------------------------------------------------------------------------- #
def weighted_sum(alpha, tensors):
    """Weighted sum of tensors."""
    return sum(w * T for w, T in zip(alpha, tensors))

# ---------------------------------------------------------------------------------------------------- #
# Mechanical properties: stiffness + CTE
# ---------------------------------------------------------------------------------------------------- #
def multi_phase_mechanical(alpha, eig_vecs, label, method):
    """
    Effective stiffness and CTE using eigen-based Voigt/Reuss averaging.

    Returns
    -------
    C_V, C_R : 4th-order tensors
        Voigt and Reuss effective stiffness tensors
    alpha_V, alpha_R : 2nd-order tensors
        Voigt and Reuss effective CTE tensors
    """
    # ------------------- Load MF results -------------------
    stiff_list = [tu.tensor_voigt(np.genfromtxt(f'elastic-{i+1}d.txt', delimiter=',', dtype=float)) for i in label]
    cte_list = [tu.tensor_cte(np.genfromtxt(f'cte-{i+1}d.txt', delimiter=',', dtype=float)) for i in label]

    beta_list = [tu.tensor_double_dot(C, a) for C, a in zip(stiff_list, cte_list)]

    # ------------------- Voigt averaging -------------------
    C_V = weighted_sum(alpha, stiff_list)
    beta_V = weighted_sum(alpha, beta_list)

    C_V = tu.tensor_eigen_trans(C_V, eig_vecs)
    beta_V = tu.tensor_eigen_trans(beta_V, eig_vecs)
    alpha_V = tu.tensor_double_dot(tu.tensor_inverse(C_V), beta_V)

    # ------------------- Reuss averaging -------------------
    S_list = [tu.tensor_inverse(C) for C in stiff_list]
    S_R = weighted_sum(alpha, S_list)
    S_R = tu.tensor_eigen_trans(S_R, eig_vecs)
    C_R = tu.tensor_inverse(S_R)

    alpha_R = weighted_sum(alpha, cte_list)
    alpha_R = tu.tensor_eigen_trans(alpha_R, eig_vecs)

    # ------------------- Save results -------------------
    stiffv = np.array([tu.tensor_voigt(C_V)[0,0], tu.tensor_voigt(C_V)[0,1], tu.tensor_voigt(C_V)[0,2],
                      tu.tensor_voigt(C_V)[1,1], tu.tensor_voigt(C_V)[1,2], tu.tensor_voigt(C_V)[2,2],
                      tu.tensor_voigt(C_V)[3,3], tu.tensor_voigt(C_V)[4,4], tu.tensor_voigt(C_V)[5,5]])
    np.savetxt(f"{method}_stiffness_matrixV.txt", stiffv, fmt="%.6e", delimiter=",")
    stiffr = np.array([tu.tensor_voigt(C_R)[0,0], tu.tensor_voigt(C_R)[0,1], tu.tensor_voigt(C_R)[0,2],
                      tu.tensor_voigt(C_R)[1,1], tu.tensor_voigt(C_R)[1,2], tu.tensor_voigt(C_R)[2,2],
                      tu.tensor_voigt(C_R)[3,3], tu.tensor_voigt(C_R)[4,4], tu.tensor_voigt(C_R)[5,5]])
    np.savetxt(f"{method}_stiffness_matrixR.txt", stiffr, fmt="%.6e", delimiter=",")
    cetv = np.array([alpha_V[0,0], alpha_V[1,1], alpha_V[2,2], alpha_V[0,1], alpha_V[0,2], alpha_V[1,2]])
    np.savetxt(f"{method}_cte_matrixV.txt", cetv, fmt="%.6e", delimiter=",")
    cetr = np.array([alpha_R[0,0], alpha_R[1,1], alpha_R[2,2], alpha_R[0,1], alpha_R[0,2], alpha_R[1,2]])
    np.savetxt(f"{method}_cte_matrixR.txt", cetr, fmt="%.6e", delimiter=",")

    return C_V, C_R, alpha_V, alpha_R


# ---------------------------------------------------------------------------------------------------- #
# Thermal conductivity
# ---------------------------------------------------------------------------------------------------- #
def multi_phase_etc(alpha, eig_vecs, label, method):
    """
    Effective thermal conductivity using eigen-based Voigt/Reuss averaging.
    """
    # ------------------- Load MF results -------------------
    k_list = [np.genfromtxt(f'etc-{i+1}d.txt', delimiter=',', dtype=float) for i in label]

    # Voigt
    K_V = weighted_sum(alpha, k_list)
    K_V = tu.tensor_eigen_trans(K_V, eig_vecs)

    # Reuss
    K_R_inv = weighted_sum(alpha, [np.linalg.inv(k) for k in k_list])
    K_R = np.linalg.inv(tu.tensor_eigen_trans(K_R_inv, eig_vecs))

    etcv = np.array([K_V[0,0], K_V[1,1], K_V[2,2], K_V[0,1], K_V[0,2], K_V[1,2]])
    np.savetxt(f"{method}_etc_matrixV.txt", K_V, fmt="%.6e", delimiter=",")
    etcr = np.array([K_R[0,0], K_R[1,1], K_R[2,2], K_R[0,1], K_R[0,2], K_R[1,2]])
    np.savetxt(f"{method}_etc_matrixR.txt", K_R, fmt="%.6e", delimiter=",")

    return K_V, K_R


# ==================================================================================================== #
# Main execution
# ==================================================================================================== #
if __name__ == "__main__":

    # ------------------- Orientation tensor -------------------
    ori_2_ay = [
        np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]]),
        np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]]),
        np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]])][2]

    # ------------------- Eigen-decomposition -------------------
    index = 2
    method = ['linear', 'quadratic', 'multi_linear'][index]
    eig_vecs, eig_vals = eu.sorted_eigens(ori_2_ay)
    pn = np.array([eig_vals[0, 0], eig_vals[1, 1]])
    if index == 0:
        alpha = eu.find_coefficients(pn)
        label = np.array([0, 1, 2])
    elif index == 1:
        alpha = qi.t6_interpolate(pn)
        label = np.array([0, 1, 2, 3, 4, 5])
    else:
        alpha, label = qi.piecewise_linear_interp(pn)

    # ------------------- Homogenization -------------------
    C_V, C_R, alpha_V, alpha_R = multi_phase_mechanical(alpha, eig_vecs, label, method)
    K_V, K_R = multi_phase_etc(alpha, eig_vecs, label, method)

    # ------------------- Screen output -------------------
    np.savetxt(sys.stdout, tu.tensor_voigt(C_V), fmt="%.3f", delimiter=", ")
    np.savetxt(sys.stdout, alpha_V, fmt="%.3f", delimiter=", ")
    np.savetxt(sys.stdout, K_V, fmt="%.3f", delimiter=", ")


# ---------------------------------------------------------------------------------------------------- #
# Export control
# ---------------------------------------------------------------------------------------------------- #
__all__ = ["multi_phase_mechanical", "multi_phase_etc"]

"""
Summary of Functions
--------------------

multi_phase_stiffness()
    Compute effective elastic stiffness of a two-phase composite with random fiber orientations.

multi_phase_etc()
    Compute effective thermal conductivity of a two-phase composite with random inclusion orientations.
"""
