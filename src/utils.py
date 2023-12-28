import torch
import numpy as np
from kornia.geometry import epipolar
from kornia.geometry.linalg import transform_points
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.geometry.epipolar import motion_from_essential_choose_solution, find_fundamental

def project_2d_to_3d(intrinsics, extrinsics, points_3d):
    P = intrinsics @ extrinsics
    return convert_points_from_homogeneous(transform_points(P, points_3d))

def params_to_extrinsic(x):
    """
    x = [Batch size, 7]
    """

    B = x.shape[0]

    rotation_matrix = quaternion_to_rotation_matrix_batch(x[:, :4])
    rt = torch.cat([
        rotation_matrix,
        x[:, 4:].unsqueeze(2)
    ], dim=2
    )
    extrinsic_batch = torch.cat(
        [rt, torch.tensor([0, 0, 0, 1], device=x.device).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)], dim=1)

    return extrinsic_batch


def create_intrinsic(K):
    # intrinsic
    intrinsic = torch.nn.functional.pad(K, (0, 1, 0, 1))
    intrinsic[:, -1, -1] = 1
    return intrinsic


def extrinsics_to_params(extrinsics):
    """
    extrinsics = tensor [B, 4, 4]
    """

    rotations = torch.tensor([rotation_matrix_to_quaternion(e).tolist() for e in extrinsics[:, :3, :3]])
    translations = extrinsics[:,:3, 3]
    extrinsics_params = torch.cat([rotations, translations], dim=1)
    return extrinsics_params


def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion."""
    # Ensure the matrix is 3x3
    if R.shape != (3, 3):
        raise ValueError("The rotation matrix must be 3x3")

    # Allocate space for the quaternion
    q = np.empty((4, ))

    # Compute the quaternion components
    q[0] = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[1] = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[3] = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2

    # Compute the signs of the quaternion components
    q[1] *= np.sign(q[1] * (R[2, 1] - R[1, 2]))
    q[2] *= np.sign(q[2] * (R[0, 2] - R[2, 0]))
    q[3] *= np.sign(q[3] * (R[1, 0] - R[0, 1]))

    return q


def quaternion_to_rotation_matrix_batch(qvec_batch):
    """Convert a batch of quaternions to rotation matrices."""
    # Validate input shape
    if qvec_batch.ndim != 2 or qvec_batch.shape[1] != 4:
        raise ValueError("Input must be a batch of quaternions with shape (n, 4)")

    # Extract components
    w, x, y, z = qvec_batch[:, 0], qvec_batch[:, 1], qvec_batch[:, 2], qvec_batch[:, 3]

    # Precompute repeated terms
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Allocate space for all rotation matrices
    batch_size = qvec_batch.shape[0]
    rotation_matrices = torch.zeros((batch_size, 3, 3), device=qvec_batch.device)

    # Compute rotation matrices
    rotation_matrices[:, 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrices[:, 0, 1] = 2 * (xy - wz)
    rotation_matrices[:, 0, 2] = 2 * (xz + wy)
    rotation_matrices[:, 1, 0] = 2 * (xy + wz)
    rotation_matrices[:, 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrices[:, 1, 2] = 2 * (yz - wx)
    rotation_matrices[:, 2, 0] = 2 * (xz - wy)
    rotation_matrices[:, 2, 1] = 2 * (yz + wx)
    rotation_matrices[:, 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrices
