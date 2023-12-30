
import torch
import random
from tqdm import tqdm
from kornia.geometry import find_fundamental, motion_from_essential_choose_solution

import plotly
import plotly.express as px
import plotly.graph_objs as go

def estimate_camera_poses(image1_features, image2_features, camera_matrix):
    """
    Estimates camera poses and triangulates points from matching features between two images.

    Args:
        image1_features: Tensor of keypoint features from image 1 (B, N, C)
        image2_features: Tensor of keypoint features from image 2 (B, N, C)
        camera_matrix: Tensor of camera intrinsics (B, 3, 3)

    Returns:
        rotations: Tensor of rotation matrices for each camera (B, 3, 3)
        translations: Tensor of translation vectors for each camera (B, 3)
        points_3d: Tensor of triangulated 3D points (B, N, 3)
    """

    # Find fundamental matrix
    fundamental_matrix = find_fundamental(image1_features, image2_features)

    # Extract essential matrix
    essential_matrix = (camera_matrix.transpose(-2, -1) @ fundamental_matrix @ camera_matrix)

    # Choose solution with positive depth and triangulate points
    rotations, translations, points_3d = motion_from_essential_choose_solution(essential_matrix, camera_matrix,
                                                                               camera_matrix,
                                                                               image1_features, image2_features)

    return rotations, translations, points_3d


def create_intrinsic(K):
    # intrinsic
    intrinsic = torch.nn.functional.pad(K, (0, 1, 0, 1))
    intrinsic[:, -1, -1] = 1
    return intrinsic


def create_extrinsics(rotations, translations):
    extrinsics = torch.cat([rotations, translations], dim=-1)

    N = extrinsics.shape[0]
    extrinsics = extrinsics.repeat(N, 1, 1)
    addition = torch.tensor([[[0, 0, 0, 1]]]).repeat(N, 1, 1)

    extrinsics = torch.cat([extrinsics, addition], dim=1)
    return extrinsics


def move_points_to_camera_world_space(extrinsic_dst, points_3d):
    points_3d_hom = torch.cat([points_3d, torch.ones([1, points_3d.shape[1], 1])], dim=2)

    extrinsic_dst_inv = torch.linalg.inv(extrinsic_dst)
    points_3d_new = torch.bmm(
        extrinsic_dst_inv,
        points_3d_hom.transpose(1, 2)
    ).transpose(1, 2)[..., :3]

    return points_3d_new


def create_visibility_matrix_from_pairs(matching_pairs):
    """
    return: matching_matrix storch.size(N points with confirmed identities, each visible across three cameras or more, N_cameras)
    the values are the indices of the points given camera.
    """

    matching_matrix = torch.arange(matching_pairs[0]['0']['points'].shape[0]).unsqueeze(1).long()

    for i in tqdm(range(matching_pairs.shape[0] - 1)):

        # match points
        matchind_indices = torch.where(
            torch.eq(
                matching_pairs[i]['1']['points'].unsqueeze(1),
                matching_pairs[i + 1]['0']['points'].unsqueeze(0),
            ).all(dim=2)
        )
        matching_matrix_temp = torch.stack(matchind_indices, dim=1)

        # find matching points among pairs
        new_col = torch.zeros(matching_matrix.shape[0]).unsqueeze(1).long() - 1
        for item in matching_matrix_temp:
            cond = item[0] == matching_matrix[:, i]
            new_col[cond] = item[1]

        # add new points with no matches
        new_items = [[i] for i in range(matching_pairs[i + 1]['0']['points'].shape[0]) if i not in new_col.flatten()]
        new_items = torch.cat([
            torch.zeros([len(new_items), i + 1], dtype=torch.long) - 1,
            torch.tensor(new_items, dtype=torch.long)],
            dim=1)

        # add connected points
        matching_matrix = torch.cat([matching_matrix, new_col], dim=1)

        # add new points
        if new_items.shape[0] > 0:
            matching_matrix = torch.cat([matching_matrix, new_items], dim=0)

        # remove isolated (points shared between 2 cameras only)
        # has_only_one_item = (matching_matrix != -1).sum(dim=1) == 1
        # has_last_item = matching_matrix[:, -1] != -1
        # matching_matrix = matching_matrix[~has_only_one_item | has_last_item]

    # remove points shared only between two cameras
    has_only_one_item = ((matching_matrix != -1).sum(dim=1) == 1)
    matching_matrix = matching_matrix[~has_only_one_item]

    return matching_matrix


def show(Rs, Ts, points_3d, colors, onlypoints=False):
    # Calculate camera position (assuming T is translation vector)
    # camera_position = -R.T @ T
    camera_traces = []
    ray_traces = []
    for R, T in zip(Rs, Ts):
        camera_position = torch.matmul(-R.T, T)

        # camera_position = T

        camera_direction = R.T[:, 2]
        camera_up_direction = R.T[:, 1]
        camera_right_direction = R.T[:, 0]

        line_length = 5  # Length of the direction line

        camera_trace = go.Scatter3d(
            x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
            mode='markers',
            marker=dict(size=6, color='red', symbol='diamond')
        )

        ray_trace = go.Scatter3d(
            x=[camera_position[0], camera_position[0] + camera_direction[0] * line_length],
            y=[camera_position[1], camera_position[1] + camera_direction[1] * line_length],
            z=[camera_position[2], camera_position[2] + camera_direction[2] * line_length],
            mode='lines',
            line=dict(color=random.choice(["green", "red", "lightgreen"]), width=5)
        )

        line_length = 0.5
        ray_up_trace = go.Scatter3d(
            x=[camera_position[0], camera_position[0] + camera_up_direction[0] * line_length],
            y=[camera_position[1], camera_position[1] + camera_up_direction[1] * line_length],
            z=[camera_position[2], camera_position[2] + camera_up_direction[2] * line_length],
            mode='lines',
            line=dict(color=random.choice(["green", "red", "lightgreen"]), width=5)
        )

        line_length = 0.25
        ray_right_trace = go.Scatter3d(
            x=[camera_position[0], camera_position[0] + camera_right_direction[0] * line_length],
            y=[camera_position[1], camera_position[1] + camera_right_direction[1] * line_length],
            z=[camera_position[2], camera_position[2] + camera_right_direction[2] * line_length],
            mode='lines',
            line=dict(color=random.choice(["green", "red", "lightgreen"]), width=5)
        )

        camera_traces.append(camera_trace)
        ray_traces.append(ray_trace)
        ray_traces.append(ray_up_trace)
        ray_traces.append(ray_right_trace)

    # Create traces
    points_trace = go.Scatter3d(
        x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
        mode='markers',
        marker=dict(size=4, color=colors, opacity=1.0)
    )

    # Layout
    layout = go.Layout(
        title='3D Points and Camera Visualization',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    # Figure
    if not onlypoints:
        fig = go.Figure(data=[points_trace] + camera_traces + ray_traces, layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(autorange='reversed'),  # This reverses the X-axis
            )
        )
    else:
        fig = go.Figure(data=[points_trace], layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(autorange='reversed'),  # This reverses the X-axis
            )
        )

    # Show the figure
    fig.show()
