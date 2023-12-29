import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils import project_2d_to_3d, params_to_extrinsic, create_intrinsic, extrinsics_to_params
from src.visibility_utils import estimate_camera_poses, create_extrinsics, move_points_to_camera_world_space, \
    create_intrinsic, create_visibility_matrix_from_pairs


def extract_3d_from_pairs(matching_pairs, imgs_path, height, K):
    zero_pair = {
        '0': {
            "name": matching_pairs[0]['0']['name'],
            "points": matching_pairs[0]['0']['points']
        },
        '1': {
            "name": matching_pairs[0]['0']['name'],
            "points": matching_pairs[0]['0']['points']
        }
    }
    matching_pairs = np.concatenate([np.array([zero_pair]), matching_pairs])

    visibility_matrix = create_visibility_matrix_from_pairs(matching_pairs)

    image0_features = matching_pairs[1]["0"]["points"].clone().cpu().unsqueeze(0)
    image1_features = matching_pairs[1]["1"]["points"].clone().cpu().unsqueeze(0)

    rotations_0to1, translations_0to1, points_3d_0to1 = estimate_camera_poses(image0_features, image1_features, K)

    image1 = torch.tensor(plt.imread(os.path.join(imgs_path, matching_pairs[0]['1']['name']))[..., :3])
    colorindices = image1_features[0].round().type(torch.long)
    colors = image1[height - colorindices[:, 1], colorindices[:, 0]]
    colors = colors.unsqueeze(0)

    # extrinsic_0to1 = create_extrinsics(rotations_0to1, translations_0to1)

    camera_views = [
        {
            "3dpoints": points_3d_0to1.clone(),
            "2dpoints": image0_features,
            "colors": colors,
            "extrinsic": torch.eye(4).unsqueeze(0),
            "intrinsic": create_intrinsic(K),
            "visibility": visibility_matrix[:, 0].not_equal(-1)
        }
    ]

    # ---

    for i in range(len(matching_pairs) - 1):
        # for i in range(15):

        # get coordinates of points
        image1_features_2 = matching_pairs[i]["1"]["points"].clone().cpu().unsqueeze(0)
        image2_features = matching_pairs[i + 1]["1"]["points"].clone().cpu().unsqueeze(0)

        # filter for points that are visible in both cameras
        visible_from_cameras_1_and_2 = (visibility_matrix != -1)[:, i:i + 2].all(dim=1)
        visible_3_cameras = visibility_matrix[visible_from_cameras_1_and_2][:, i:i + 2]

        image1_features_2 = image1_features_2[0:1, visible_3_cameras[:, 0], :]
        image2_features = image2_features[0:1, visible_3_cameras[:, 1], :]

        # 3d estimation using matching points
        rotations_1to2, translations_1to2, points_3d_1to2 = estimate_camera_poses(image1_features_2, image2_features, K)

        # find colors of 3d points
        image1 = torch.tensor(plt.imread(os.path.join(imgs_path, matching_pairs[i + 1]['0']['name']))[..., :3])
        colorindices = image1_features_2[0].round().type(torch.long)
        colors2 = image1[height - colorindices[:, 1], colorindices[:, 0]]
        colors2 = colors2.unsqueeze(0)

        # place camera in the same world space as the rest of the cameras
        extrinsic_1to2 = create_extrinsics(rotations_1to2, translations_1to2)

        points_3d_new = move_points_to_camera_world_space(camera_views[-1]["extrinsic"], points_3d_1to2)
        extrinsic_new_0 = torch.bmm(torch.eye(4).unsqueeze(0), camera_views[-1]["extrinsic"])
        extrinsic_new = torch.bmm(extrinsic_1to2, camera_views[-1]["extrinsic"])

        # add camera and points to list
        camera_views.append(
            {
                "3dpoints": points_3d_new,
                "2dpoints": image2_features,
                "colors": colors2,
                "extrinsic": extrinsic_new,
                "intrinsic": create_intrinsic(K),
                "visibility": visible_from_cameras_1_and_2  # saves which point is visible to camera
            }
        )

    return camera_views, visibility_matrix


def prepare_data(camera_views, visibility_matrix, filter_visible=-1):
    batch_point3d = []
    batch_point2d = []
    batch_colors = []
    batch_visibility = []

    # iterate over points
    for i in tqdm(range(visibility_matrix.shape[0])):

        temp_points3d = []
        temp_colors = []

        temp_points2d = torch.zeros([len(camera_views), 2])
        temp_visibility = torch.zeros([len(camera_views)])

        # iterate over cameras
        for j, camera in enumerate(camera_views):

            mask = torch.where(camera['visibility'])[0].eq(i)

            if ~mask.any():
                continue

            relevant_point3d = camera['3dpoints'][0, mask][0]
            relevant_point2d = camera["2dpoints"][0, mask][0]
            relevant_color = camera["colors"][0, mask][0]

            temp_points3d.append(relevant_point3d)
            temp_colors.append(relevant_color)

            temp_points2d[j] = relevant_point2d
            temp_visibility[j] = 1

        if len(temp_points3d) == 0:
            continue

        batch_point3d.append(torch.stack(temp_points3d))
        batch_colors.append(torch.stack(temp_colors))
        batch_point2d.append(temp_points2d)
        batch_visibility.append(temp_visibility)

    # batch_point3d = torch.stack(batch_point3d)
    # batch_colors = torch.stack(batch_colors)
    batch_point2d = torch.stack(batch_point2d)
    batch_visibility = torch.stack(batch_visibility).type(torch.bool)

    # filters for points visible by filter_visible cameras or more
    if filter_visible > 0:
        mask = batch_visibility.sum(dim=1) >= filter_visible  # use
        batch_visibility = batch_visibility[mask]
        batch_point3d = [p for p, m in zip(batch_point3d, mask) if m]
        batch_point2d = batch_point2d[mask, :]
        batch_colors = [p for p, m in zip(batch_colors, mask) if m]

    return batch_visibility, batch_point3d, batch_point2d, batch_colors


def fit(camera_extrinsic, camera_intrinsic, batch_point2d, batch_point3d, batch_visibility, steps=3000, lr=0.005,
        train_angle=False, camera_indecies_to_train=[], fit_cam_only=False, device="cuda"):
    # define model

    model = ProjectionModel(camera_intrinsic.clone(), camera_extrinsic.clone(), batch_point3d.clone(),
                            batch_visibility.clone(), device=device)
    if len(camera_indecies_to_train) > 0:
        if fit_cam_only:
            model.create_hooks(camera_indecies_to_train, mask_extrinsics=True)
        else:
            model.create_hooks(camera_indecies_to_train, mask_extrinsics=True, mask_points=True)

    # fit

    labels = [batch_point2d[batch_visibility[:, index], index].to(device) for index in range(batch_visibility.shape[1])]

    params = [model.extrinsics_params, model.points3d]
    if train_angle: params.append(model.camera_angle_x)

    # optimizer = torch.optim.Adam([model.extrinsics_params, model.points3d], lr=0.005)
    optimizer = torch.optim.Adamax(params, lr=lr)
    # optimizer = torch.optim.SGD([model.extrinsics_params, model.points3d], lr=0.007)

    lossgraph = []

    # training loop
    try:
        with tqdm(total=steps) as pbar:
            for _ in range(steps):
                optimizer.zero_grad()
                projections = model.forward()

                loss = sum([torch.nn.functional.mse_loss(projections[index], labels[index]) for index in
                            range(len(projections))]) / len(projections)
                loss.backward()

                optimizer.step()
                lossgraph.append(loss.detach().item())

                pbar.update()
                pbar.set_description(f"current loss: {loss.detach().cpu().item():.4f}")
    except KeyboardInterrupt:
        pass

    return model, lossgraph


class ProjectionModel(torch.nn.Module):
    def __init__(self, intrinsics, extrinsics, points3d, visibility, width=800, height=800, shared_camera=True,
                 device=None):
        """
        intrinsics: torch.Size([3, 4, 4])
        extrinsics: torch.Size([3, 4, 4])
        project_3d: torch.Size([803, 3])
        """
        super().__init__()

        # set device
        avilable_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device if (device is not None) else avilable_device

        # focal length
        self.camera_angle_x = torch.nn.Parameter(torch.tensor([0.6911112070083618]).to(self.device), requires_grad=True)
        self.shared_camera = shared_camera

        # dimentions
        self.width = width
        self.height = height

        # camera parameters
        if self.shared_camera:
            self.intrinsics = None
        else:
            self.intrinsics = torch.nn.Parameter(intrinsics.to(self.device), requires_grad=True)

        self.extrinsics_params = torch.nn.Parameter(extrinsics_to_params(extrinsics).to(self.device),
                                                    requires_grad=True)
        self.points3d = torch.nn.Parameter(points3d.to(self.device), requires_grad=True)
        self.visibility = visibility.to(self.device)


    def preprocess_points3d(self, points3d):
        batch_points3d = []
        num_points_visible_for_camera = self.visibility.sum(0)
        pad_vals = num_points_visible_for_camera.max() - num_points_visible_for_camera

        for index in range(self.visibility.shape[1]):
            mask = self.visibility[:, index]
            points3d_camera = points3d[mask]
            points3d_camera = torch.nn.functional.pad(points3d_camera, (0, 0, 0, pad_vals[index]), "constant", 0)
            batch_points3d.append(points3d_camera)

        return torch.stack(batch_points3d), num_points_visible_for_camera


    def get_intrinsics(self):

        if not self.shared_camera:
            return self.intrinsics

        f = self.width / (2 * torch.tan(self.camera_angle_x / 2))

        K = torch.eye(3).unsqueeze(0).to(self.device)
        K[0, 0, 2] = self.width / 2
        K[0, 1, 2] = self.height / 2
        K[0, 0, 0] = f
        K[0, 1, 1] = f
        intrinsics = create_intrinsic(K).repeat(self.extrinsics_params.shape[0], 1, 1)
        return intrinsics

    def forward(self):

        intrinsics = self.get_intrinsics()
        extrinsics = params_to_extrinsic(self.extrinsics_params)

        points3d, num_points_visible_for_camera = self.preprocess_points3d(self.points3d)

        points3d_projected = project_2d_to_3d(intrinsics, extrinsics, points3d)
        points3d_projected = [p[:num_points_visible_for_camera[i]] for i, p in enumerate(points3d_projected)]

        return points3d_projected

    def create_hooks(self, indecies, mask_intrinsics=False, mask_extrinsics=False, mask_points=False):

        # The actual hook function
        def extrinsic_parameters_grad_hook(grad):
            mask = torch.zeros_like(grad, dtype=torch.bool)
            mask[indecies] = 1
            return grad * mask

        def points3d_grad_hook(grad):
            # optimize only the points visible to at least three of the relevant cameras
            mask = self.visibility[:, indecies].sum(dim=1).ge(3).type(torch.long).unsqueeze(1)

            return grad * mask

        def intrinsic_grad_hook(grad):
            mask = torch.zeros_like(grad, dtype=torch.bool)
            mask[indecies] = 1
            return grad * mask

        if mask_points:
            self.points3d.register_hook(points3d_grad_hook)
        if mask_extrinsics:
            self.extrinsics_params.register_hook(extrinsic_parameters_grad_hook)
        if mask_intrinsics:
            self.intrinsics.register_hook(intrinsic_grad_hook)