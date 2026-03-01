import numpy as np
#
def generate_periodic_images(points, angles, rve_size, perio_shift):
    """
    Generate periodic images of points and their associated angles within an RVE.

    Parameters:
    -----------
    points : np.ndarray, shape (n, 3)
        Array of 3D coordinates for `n` points.

    angles : np.ndarray, shape (n, 3)
        Array of angles associated with each point, where each row corresponds to a point.

    rve_size : np.ndarray, shape (3)
        Size of the representative volume element (RVE).

    perio_shift : float
        Tolerance value for filtering points near the RVE boundary.

    Returns:
    --------
    np.ndarray, np.ndarray
        Filtered points and their corresponding angles, both adjusted for periodic boundary conditions.
    """
    # Generate periodic shifts for all combinations of [-1, 0, 1] along each axis
    shifts = np.array(np.meshgrid(*[np.array([-1, 0, 1]) for _ in range(3)])).T.reshape(-1, 3) * rve_size
    # Apply all shifts to the original points to generate periodic images
    points_images = np.vstack([points + shift for shift in shifts])
    # Duplicate angles for all generated periodic images
    angles_images = np.tile(angles, (len(shifts), 1))
    # Filter points outside the RVE boundary considering the periodic shift
    half_rve = rve_size / 2.0
    mask = ~np.any(((points_images - half_rve) / (half_rve + perio_shift)).astype(int) != 0, axis=1)
    points_filtered = points_images[mask]
    angles_filtered = angles_images[mask]

    return points_filtered, angles_filtered