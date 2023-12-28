import torch

from src.utils import project_2d_to_3d, params_to_extrinsic, create_intrinsic, extrinsics_to_params


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