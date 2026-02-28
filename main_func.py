import time
import numpy as np
from scipy.spatial import KDTree
from distance3d import colliders, reorientation
from distance3d import mpr, periodic_image
from pytransform3d import rotations

start_time = time.time()  # Start timing

# ========================== Parameters =============================
#
ori_2_ay = [
        np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]]),
        np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]]),
        np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]])][2]

rve_size = np.array([150.0, 150.0, 150.0])  # RVE dimensions
inc_size, inc_vf, inc_enl = np.array([5.0/2, 75.0]), 0.10, 1.05  # Parameters of inclusions
inc_ori_tensor2 = ori_2_ay #np.array([[3.0/4, 0.0, 0.0], [0.0, 1.0/4, 0.0], [0.0, 0.0, 0.0]]) * 1.0
inc_volume = np.pi * inc_size[0] ** 2 * inc_size[1]  # volume of cylinder
inc_num = int(np.prod(rve_size) * inc_vf / inc_volume)  # Calculate the number of inclusions
inc_size = inc_size * inc_enl  # Enlarged inclusion size
inc_cutoff = np.linalg.norm(inc_size)  # Cut-off distance of KDTree method
alpha, beta, tolerance = 0.2, 1.0, inc_size[0] * 1.e-3  # Damp coefficients of gradient descent method

# ====================== centers and angles =========================
#
points = np.random.uniform([0, 0, 0], rve_size, size=(inc_num, 3))
angles = np.random.uniform([0, 0, 0], np.pi * np.array([2, 1, 1]), size=(inc_num, 3))
inc_ori_vecs = np.column_stack((np.sin(angles[:, 1]) * np.cos(angles[:, 0]),
    np.sin(angles[:, 1]) * np.sin(angles[:, 0]), np.cos(angles[:, 1])))

# ================== Predefined orientation ==========================
#
pred_ori_tensor4 = reorientation.ori_tensor4_recon(inc_ori_tensor2)
inc_ori_vecs = reorientation.orivector_optimization(inc_ori_vecs, pred_ori_tensor4, beta)
# t = pred_ori_tensor4 - reorientation.current_ori_tensor4(inc_ori_vecs)
angles = reorientation.optimized_ori_angles(inc_ori_vecs, angles)

# =========================== Main loop ==============================
#
gradients = np.zeros_like(points)
cylinder2origin_1, cylinder2origin_2 = np.zeros([4, 4]), np.zeros([4, 4])
cylinder2origin_1[3, 3], cylinder2origin_2[3, 3] = 1.0, 1.0

for iter_num in range(int(1e4)):  # Set maximum iteration limit

    # Calculate gradients and potentials
    gradients.fill(0)
    potential = 0.0

    # Loop over each cell in the cutoff pairs
    tree = KDTree(points, boxsize=rve_size)
    pairs = tree.query_pairs(r=inc_cutoff)  # Find all pairs of points within the interaction radius

    # Loop over each cell in the grid
    for p1, p2 in pairs:
        point_1 = points[p1]
        cylinder2origin_1[0:3, 0:3] = rotations.matrix_from_euler(angles[p1], 2, 1, 2, False)
        cylinder2origin_1[:3, 3] = point_1
        c1 = colliders.Cylinder(cylinder2origin_1, inc_size[0], inc_size[1])
        point_2 = points[p2]
        iden_vec = np.round((point_1 - point_2) / rve_size)  # Compute periodic boundary vector
        point_2 += iden_vec * rve_size  # Adjust positions for periodic boundaries
        cylinder2origin_2[0:3, 0:3] = rotations.matrix_from_euler(angles[p2], 2, 1, 2, False)
        cylinder2origin_2[:3, 3] = point_2
        c2 = colliders.Cylinder(cylinder2origin_2, inc_size[0], inc_size[1])
        # Check if distance is within the interaction radius
        if np.linalg.norm(point_1 - point_2) < inc_cutoff:
            contact, depth, pene_dir, _ = mpr.mpr_penetration(c1, c2)
            if contact:
                grad = - depth * pene_dir  # Calculate the gradient contribution
                potential += 0.5 * depth ** 2  # Update potential
                gradients[p1] += grad  # Accumulate gradients for both points
                gradients[p2] -= grad

    # Update positions using gradient descent
    points += alpha * gradients
    points %= rve_size  # Apply periodic boundary condition

    # Log potential and check for convergence
    print(f"Iteration {iter_num}: Potential = {potential:.8f}")
    if potential <= np.min(tolerance):
        break  # Exit the loop

# =========================== Add periodic images ===========================
#
peri_shift = inc_cutoff / 2.0
peri_points, peri_angles = periodic_image.generate_periodic_images(points, angles, rve_size, peri_shift)

# =========================== Combine and save results ======================
#
np.savetxt('points-a3.txt', points, delimiter=',', fmt='%.6f')
np.savetxt('angles-a3.txt', angles, delimiter=',', fmt='%.6f')
np.savetxt('peri_points-a3.txt', peri_points, delimiter=',', fmt='%.6f')
np.savetxt('peri_angles-a3.txt', peri_angles, delimiter=',', fmt='%.6f')
print(f"Execution time: {time.time() - start_time:.6f} seconds")