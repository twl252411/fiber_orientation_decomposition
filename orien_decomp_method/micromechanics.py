# ==================================================================================================== #
# multi_phase_main.py — Effective stiffness and conductivity of multi-phase composites
# ==================================================================================================== #
import numpy as np
import tensor_utils as tu
import mean_field_utils as mf
import eigen_utils as eu
import orien_average as oa
import sys


# ---------------------------------------------------------------------------------------------------- #
# Function: multi_phase_stiffness
# ---------------------------------------------------------------------------------------------------- #
def multi_phase_mechnical() -> np.ndarray:
    """
    Compute the effective stiffness matrix (6×6, Voigt form) for a two-phase composite
    reinforced with randomly oriented fibers using eigen-based orientation averaging.

    Returns
    -------
    np.ndarray
        Effective 6×6 stiffness matrix (Voigt notation).

    Notes
    -----
    - The matrix and inclusion are both treated as transversely isotropic solids.
    - The unidirectional composite stiffness is obtained by mean-field homogenization
      (Mori–Tanaka or Double-Inclusion models).
    - The final stiffness tensor is orientation-averaged through eigen-based
      interpolation among 1D, 2D, and 3D orientation states.
    """

    # ------------------- Matrix properties -------------------
    e11_mat = e22_mat = e33_mat = 3
    nu12_mat = nu13_mat = nu23_mat = 0.35
    g12_mat = g13_mat = g23_mat = 0.5 * e11_mat / (1 + nu12_mat)
    cte_mat = np.eye(3) * 70.0

    # ------------------- Inclusion properties -------------------
    e11_inc = 172
    e22_inc = e33_inc = 172
    nu12_inc = nu13_inc = 0.2
    g12_inc = g13_inc = e33_inc/2/(1+nu12_inc)
    g23_inc = g13_inc
    nu23_inc = nu12_inc
    cte_inc = np.array([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1]])

    # ------------------- Inclusion aspect ratio and volume fraction -------------------
    inc_ar = 15 * 1.25
    inc_vf = 0.10

    # ------------------- 4th-order stiffness and beta tensors -------------------
    stiff_mat = mf.phase_stiffess(e11_mat, e22_mat, e33_mat, g12_mat, g13_mat, g23_mat, nu12_mat, nu13_mat, nu23_mat)
    stiff_inc = mf.phase_stiffess(e11_inc, e22_inc, e33_inc, g12_inc, g13_inc, g23_inc, nu12_inc, nu13_inc, nu23_inc)
    stiff_inc = tu.tensor_orien_trans(stiff_inc, -np.pi/2, 0)
    #
    beta_mat = tu.tensor_double_dot(stiff_mat, cte_mat)
    beta_inc = tu.tensor_double_dot(stiff_inc, cte_inc)

    # ------------------- Unidirectional composite stiffness and cte -------------------
    uni_stiff, uni_gamma = mf.two_phase_stiffness(stiff_mat, stiff_inc, inc_vf, inc_ar, nu12_mat, nu12_inc,
                                                  beta_mat, beta_inc, model_type="mori-tanaka")
    uni_beta = (1.0 - inc_vf) * beta_mat + inc_vf * (beta_inc + uni_gamma)
    uni_cte = tu.tensor_double_dot(tu.tensor_inverse(uni_stiff), uni_beta)

    # ------------------- Orientation tensors -------------------
    ori_2_ay = np.array([[0.19, 0.028, 0.0],
                         [0.028, 0.81, 0.00],
                         [0.0, 0.00, 0.0]])

    ori_2_1d = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])

    ori_2_2d = np.array([[0.5, 0.0, 0.0],
                         [0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0]])

    ori_2_3d = np.eye(3) / 3

    # ------------------- Eigen-decomposition -------------------
    eig_vecs, eig_vals = eu.sorted_eigens(ori_2_ay)
    pn = np.array([eig_vals[0, 0], eig_vals[1, 1]])
    alpha = eu.find_coefficients(pn)

    # ------------------- Orientation-averaged tensors -------------------
    ori_4_1d = mf.orien_closure_approx(ori_2_1d, '1D')
    ori_4_2d = mf.orien_closure_approx(ori_2_2d, '2D', zero_p=2)
    ori_4_3d = mf.orien_closure_approx(ori_2_3d, '3D')

    average_stiff_1d = oa.orien_ten_average(uni_stiff, ori_2_1d, ori_4_1d)
    average_stiff_2d = oa.orien_ten_average(uni_stiff, ori_2_2d, ori_4_2d)
    average_stiff_3d = oa.orien_ten_average(uni_stiff, ori_2_3d, ori_4_3d)

    average_cte_1d = tu.tensor_orien_trans(uni_cte, np.pi / 2.0, 0.0)
    average_cte_2d = oa.orien_pdf_average(uni_cte, dangle=0.5, dim='2D')
    average_cte_3d = oa.orien_pdf_average(uni_cte, dangle=0.5, dim='3D')

    average_beta_1d = tu.tensor_orien_trans(uni_beta, np.pi / 2.0, 0.0)
    average_beta_2d = oa.orien_pdf_average(uni_beta, dangle=0.5, dim='2D')
    average_beta_3d = oa.orien_pdf_average(uni_beta, dangle=0.5, dim='3D')

    # ------------------- Voigt Eigen-based interpolation -------------------
    average_stiff = alpha[0] * average_stiff_1d + alpha[1] * average_stiff_2d + alpha[2] * average_stiff_3d
    multi_stiffv = tu.tensor_eigen_trans(average_stiff, eig_vecs)

    average_beta = alpha[0] * average_beta_1d + alpha[1] * average_beta_2d + alpha[2] * average_beta_3d
    multi_beta = tu.tensor_eigen_trans(average_beta, eig_vecs)
    multi_ctev = tu.tensor_double_dot(tu.tensor_inverse(multi_stiffv), multi_beta)

    # ------------------- Ruess Eigen-based interpolation -------------------
    average_compli = (alpha[0] * tu.tensor_inverse(average_stiff_1d) + alpha[1] * tu.tensor_inverse(average_stiff_2d)
                      + alpha[2] * tu.tensor_inverse(average_stiff_3d))
    multi_stiffr = tu.tensor_inverse(tu.tensor_eigen_trans(average_compli, eig_vecs))

    average_cte = alpha[0] * average_cte_1d + alpha[1] * average_cte_2d + alpha[2] * average_cte_3d
    multi_cter = tu.tensor_eigen_trans(average_cte, eig_vecs)

    # ------------------- Convert to 6×6 Voigt form and save -------------------
    multi_stiff_voigtv = tu.tensor_voigt(multi_stiffv)
    multi_stiff_voigtr = tu.tensor_voigt(multi_stiffr)

    # ------------------- Write the results to the text files -------------------
    np.savetxt('stiffness_matrixv.txt', multi_stiff_voigtv, delimiter=',', fmt='%.6e')
    np.savetxt('stiffness_matrixr.txt', multi_stiff_voigtr, delimiter=',', fmt='%.6e')
    np.savetxt('cte_matrixv.txt', multi_ctev, delimiter=',', fmt='%.6e')
    np.savetxt('cte_matrixr.txt', multi_cter, delimiter=',', fmt='%.6e')

    return tu.tensor_voigt(average_stiff_1d), tu.tensor_voigt(average_stiff_2d), tu.tensor_voigt(average_stiff_3d)

# ---------------------------------------------------------------------------------------------------- #
# Function: multi_phase_etc
# ---------------------------------------------------------------------------------------------------- #
def multi_phase_etc() -> np.ndarray:
    """
    Compute the effective thermal conductivity tensor (3×3) for a two-phase composite
    with randomly oriented inclusions using eigen-based orientation averaging.

    Returns
    -------
    np.ndarray
        Effective 3×3 thermal conductivity tensor.

    Notes
    -----
    - The computation assumes ellipsoidal inclusions with given aspect ratio and volume fraction.
    - The unidirectional tensor is first evaluated using mean-field theory, then averaged
      over all orientations via eigen-decomposition and PDF-based interpolation.
    """

    # ------------------- Matrix and inclusion properties -------------------
    etc_mat = np.eye(3) * 0.22
    etc_inc = np.array([[2.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 8.8]])

    # ------------------- Inclusion aspect ratio and volume fraction -------------------
    inc_ar = 15*1.25
    inc_vf = 0.10
    semi_minor = 2.5E-6

    # ------------------- Unidirectional composite conductivity -------------------
    uni_tensor = mf.two_phase_etc(etc_mat, etc_inc, inc_vf, inc_ar, semi_minor)

    # ------------------- Orientation tensor -------------------
    ori_2_ay = np.array([[0.58, 0.019, -0.015],
                         [0.019, 0.17, -0.012],
                         [-0.015, -0.012, 0.25]])

    # ------------------- Eigen-decomposition -------------------
    eig_vecs, eig_vals = eu.sorted_eigens(ori_2_ay)
    pn = np.array([eig_vals[0, 0], eig_vals[1, 1]])
    alpha = eu.find_coefficients(pn)

    # ------------------- Orientation-averaged tensors -------------------
    average_tensor_1d = tu.tensor_orien_trans(uni_tensor, np.pi / 2.0, 0.0)
    average_tensor_2d = oa.orien_pdf_average(uni_tensor, dangle=1.0, dim='2D')
    average_tensor_3d = oa.orien_pdf_average(uni_tensor, dangle=1.0, dim='3D')

    # ------------------- Eigen-based interpolation -------------------
    average_tensorv = alpha[0] * average_tensor_1d \
                   + alpha[1] * average_tensor_2d \
                   + alpha[2] * average_tensor_3d
    multi_etcv = tu.tensor_eigen_trans(average_tensorv, eig_vecs)

    average_tensorr = alpha[0] * np.linalg.inv(average_tensor_1d) \
                   + alpha[1] * np.linalg.inv(average_tensor_2d) \
                   + alpha[2] * np.linalg.inv(average_tensor_3d)
    multi_etcr = np.linalg.inv(tu.tensor_eigen_trans(average_tensorr, eig_vecs))
    # ------------------- Save result -------------------
    np.savetxt('etc_matrixv.txt', multi_etcv, delimiter=',', fmt='%.6e')
    np.savetxt('etc_matrixr.txt', multi_etcr, delimiter=',', fmt='%.6e')

    return multi_etcv


# ==================================================================================================== #
# Main execution
# ==================================================================================================== #
if __name__ == '__main__':
    multi_stiff_voigt, multi_cte, multi_cte1 = multi_phase_mechnical()
    multi_phase_etc()
    np.savetxt(sys.stdout, multi_stiff_voigt, fmt="%.3f", delimiter=", ")
    np.savetxt(sys.stdout, multi_cte, fmt="%.3f", delimiter=", ")
    np.savetxt(sys.stdout, multi_cte1, fmt="%.3f", delimiter=", ")


# -------------------------------------------------------------------------------------------------------------------- #
# Module summary and export control
# -------------------------------------------------------------------------------------------------------------------- #
__all__ = ["multi_phase_mechnical", "multi_phase_etc"]

"""
Summary of Functions
--------------------

multi_phase_stiffness()
    Compute effective elastic stiffness of a two-phase composite with random fiber orientations.

multi_phase_etc()
    Compute effective thermal conductivity of a two-phase composite with random inclusion orientations.
"""
