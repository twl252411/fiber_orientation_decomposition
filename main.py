import argparse
import time
from pathlib import Path

import numpy as np
from pytransform3d import rotations
from scipy.spatial import cKDTree
from distance3d import colliders, mpr, periodic_image, reorientation


ORI_2_AY = [
    np.array([[0.58, 0.019, -0.015], [0.019, 0.17, -0.012], [-0.015, -0.012, 0.25]]),
    np.array([[0.40, 0.069, 0.26], [0.069, 0.17, -0.001], [0.26, -0.001, 0.43]]),
    np.array([[0.19, 0.028, 0.00], [0.028, 0.81, 0.0], [0.0, 0.0, 0.0]]), ]

OUTPUT_DIR = Path("point_angle_files")


def parse_args():
    parser = argparse.ArgumentParser(description="Fiber orientation decomposition in periodic RVE.")
    parser.add_argument(
        "--ori-id",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Index for ORI_2_AY[id], valid values: 0, 1, 2.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible initialization.")
    parser.add_argument("--max-iter", type=int, default=int(1e4), help="Maximum iteration count for packing loop.")
    parser.add_argument("--log-interval", type=int, default=50, help="Iteration logging interval. Use 0 to disable.")
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Output filename tag. If omitted, uses a{ori_id+1}.",
    )
    return parser.parse_args()


def _resolve_output_tag(output_tag, output_dir):
    output_files = [
        output_dir / f"points_{output_tag}.txt",
        output_dir / f"angles_{output_tag}.txt",
        output_dir / f"peri_points_{output_tag}.txt",
        output_dir / f"peri_angles_{output_tag}.txt",
    ]
    if any(path.exists() for path in output_files):
        return f"{output_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    return output_tag


def _log(iter_num, potential, log_interval, is_last=False):
    if log_interval <= 0:
        return
    if is_last or (iter_num % log_interval == 0):
        print(f"Iteration {iter_num}: Potential = {potential:.8f}")


def run(ori_id=0, random_seed=42, max_iterations=int(1e4), log_interval=50, output_tag=None):
    start_time = time.time()
    rng = np.random.default_rng(random_seed)

    # ========================== Parameters =============================
    rve_size = np.array([100.0, 100.0, 100.0], dtype=float)
    inc_size_base = np.array([5.0 / 2.0, 50.0], dtype=float)  # [radius, length]
    inc_vf = 0.15
    inc_enl = 1.05
    inc_ori_tensor2 = ORI_2_AY[ori_id]

    inc_volume = np.pi * inc_size_base[0] ** 2 * inc_size_base[1]
    inc_num = int(np.prod(rve_size) * inc_vf / inc_volume)
    inc_size = inc_size_base * inc_enl
    inc_cutoff = np.sqrt(inc_size[1] ** 2 + 4.0 * inc_size[0] ** 2)
    inv_rve_size = 1.0 / rve_size

    # Damping coefficients
    alpha = 0.5
    beta = 1.0
    tolerance = inc_size[0] * 1.0e-3

    # ====================== centers and angles =========================
    points = rng.uniform([0.0, 0.0, 0.0], rve_size, size=(inc_num, 3))
    angles = rng.uniform([0.0, 0.0, 0.0], np.pi * np.array([2.0, 1.0, 1.0]), size=(inc_num, 3))
    inc_ori_vecs = np.column_stack(
        (np.sin(angles[:, 1]) * np.cos(angles[:, 0]),
         np.sin(angles[:, 1]) * np.sin(angles[:, 0]),
         np.cos(angles[:, 1]), ))

    # ================== Predefined orientation ==========================
    pred_ori_tensor4 = reorientation.ori_tensor4_recon(inc_ori_tensor2)
    inc_ori_vecs = reorientation.orivector_optimization(
        inc_ori_vecs, pred_ori_tensor4, beta, log_interval=10, )
    angles = reorientation.optimized_ori_angles(inc_ori_vecs, angles)

    # Angles are fixed in packing loop, so rotation matrices can be precomputed once.
    rotation_mats = np.empty((inc_num, 3, 3), dtype=float)
    for i in range(inc_num):
        rotation_mats[i] = rotations.matrix_from_euler(angles[i], 2, 1, 2, False)

    # =========================== Main loop ==============================
    gradients = np.zeros_like(points)
    cylinder2origin_1 = np.eye(4, dtype=float)
    cylinder2origin_2 = np.eye(4, dtype=float)
    c1 = colliders.Cylinder(cylinder2origin_1, inc_size[0], inc_size[1])
    c2 = colliders.Cylinder(cylinder2origin_2, inc_size[0], inc_size[1])

    final_iteration = max_iterations - 1
    final_potential = 0.0

    for iter_num in range(max_iterations):
        gradients.fill(0.0)
        potential = 0.0

        pairs = cKDTree(points, boxsize=rve_size).query_pairs(r=inc_cutoff, output_type="ndarray")

        for p1, p2 in pairs:
            point_1 = points[p1]

            # Copy is required; in-place shift must not mutate points[p2].
            point_2 = points[p2].copy()
            iden_vec = np.rint((point_1 - point_2) * inv_rve_size)
            point_2 += iden_vec * rve_size

            cylinder2origin_1[0:3, 0:3] = rotation_mats[p1]
            cylinder2origin_1[:3, 3] = point_1
            cylinder2origin_2[0:3, 0:3] = rotation_mats[p2]
            cylinder2origin_2[:3, 3] = point_2

            contact, depth, pene_dir, _ = mpr.mpr_penetration(c1, c2)
            if contact:
                grad = -depth * pene_dir
                potential += 0.5 * depth * depth
                gradients[p1] += grad
                gradients[p2] -= grad

        points += alpha * gradients
        points %= rve_size

        final_iteration = iter_num
        final_potential = potential
        _log(iter_num, potential, log_interval, is_last=(iter_num == max_iterations - 1))
        if potential <= tolerance:
            break

    # =========================== Add periodic images ===========================
    peri_shift = inc_cutoff / 2.0
    peri_points, peri_angles = periodic_image.generate_periodic_images(points, angles, rve_size, peri_shift)

    # =========================== Combine and save results ======================
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = _resolve_output_tag(f"a{ori_id + 1}" if output_tag is None else output_tag, output_dir)

    angles_deg = np.rad2deg(angles)
    peri_angles_deg = np.rad2deg(peri_angles)

    np.savetxt(output_dir / f"points_{output_tag}.txt", points, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / f"angles_{output_tag}.txt", angles_deg, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / f"peri_points_{output_tag}.txt", peri_points, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / f"peri_angles_{output_tag}.txt", peri_angles_deg, delimiter=",", fmt="%.6f")

    print(f"Final iteration: {final_iteration}, potential: {final_potential:.8f}")
    print(f"Execution time: {time.time() - start_time:.6f} seconds")
    print(f"Orientation tensor id: {ori_id}")
    print(f"Output directory: {output_dir}")
    print(f"Output tag: {output_tag}")


if __name__ == "__main__":
    args = parse_args()
    run(
        ori_id=args.ori_id,
        random_seed=args.seed,
        max_iterations=args.max_iter,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )
