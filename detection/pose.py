"""Estimate 3D armor-plate corners from image corners and calibration."""

import cv2
import numpy as np


LARGE_PATTERNS = {"1", "base_big"}


def load_calibration(path):
    """Read the camera matrix and distortion coefficients from the YAML file."""
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Could not open calibration file: {path}")

    # This assignment's YAML stores matrices as nested numeric sequences, so
    # read each scalar from the sequence nodes instead of calling node.mat().
    matrix_node = storage.getNode("camera_matrix")
    camera_matrix = np.array(
        [
            [matrix_node.at(row).at(col).real() for col in range(matrix_node.at(row).size())]
            for row in range(matrix_node.size())
        ],
        dtype=np.float64,
    )
    distortion_node = storage.getNode("distort_coeffs")
    distortion = np.array(
        [distortion_node.at(index).real() for index in range(distortion_node.size())],
        dtype=np.float64,
    )
    storage.release()
    return camera_matrix, distortion


def object_corners(pattern):
    """Return the four physical plate corners in the assignment's order."""
    width = 0.230 if pattern in LARGE_PATTERNS else 0.135
    height = 0.056
    return np.array(
        [
            [-width / 2, -height / 2, 0],  # upper-left
            [-width / 2, +height / 2, 0],  # lower-left
            [+width / 2, +height / 2, 0],  # lower-right
            [+width / 2, -height / 2, 0],  # upper-right
        ],
        dtype=np.float32,
    )


def estimate_pose(candidate, camera_matrix, distortion):
    """Return camera-space 3D corners and reprojection RMS, or (None, None)."""
    physical_corners = object_corners(candidate["pattern"])
    image_corners = np.asarray(candidate["corners_px"], dtype=np.float32)

    # IPPE handles a general planar rectangle and can return two pose choices.
    result = cv2.solvePnPGeneric(
        physical_corners,
        image_corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    success, rotation_vectors, translation_vectors = result[:3]
    if not success:
        return None, None

    solutions = []
    for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
        rotation, _ = cv2.Rodrigues(rotation_vector)
        corners_camera = (rotation @ physical_corners.T).T + translation_vector.reshape(1, 3)

        # Reject the mirror solution if it places any physical corner behind
        # the camera. Then score remaining solutions by their pixel reprojection.
        if not np.isfinite(corners_camera).all() or np.any(corners_camera[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(
            physical_corners, rotation_vector, translation_vector, camera_matrix, distortion
        )
        residual = projected.reshape(4, 2) - image_corners
        rms = float(np.sqrt(np.sum(residual * residual) / 4.0))
        solutions.append((rms, corners_camera))

    if not solutions:
        return None, None
    rms, corners_camera = min(solutions, key=lambda solution: solution[0])
    return corners_camera.tolist(), rms
