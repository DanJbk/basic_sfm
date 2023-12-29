
import torch
import numpy as np

from src.visibility_utils import show
from src.utils import params_to_extrinsic
from src.model import extract_3d_from_pairs, prepare_data, fit

import plotly
import plotly.express as px
import plotly.graph_objs as go


def default_arguments():

    camera_angle_x = 0.6911112070083618
    width = 800
    height = 800

    # data paths
    imgs_path = "D:\\9.programming\\Plenoxels\\data\\lego\\test"
    matching_pairs = np.load("D:\\9.programming\\sfm\\matching_pairs_updated.npy", allow_pickle=True)

    return camera_angle_x, width, height, imgs_path, matching_pairs

def main():
    # camera parameters

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    camera_angle_x, width, height, imgs_path, matching_pairs = default_arguments()

    for i, pair in enumerate(matching_pairs):
        matching_pairs[i]['0']['points'][:, 1] = height - pair['0']['points'][:, 1]
        matching_pairs[i]['1']['points'][:, 1] = height - pair['1']['points'][:, 1]

    # Compute focal length

    camera_angle_x = torch.tensor([camera_angle_x])
    f = width / (2 * torch.tan(camera_angle_x / 2))

    # Construct the camera matrix K
    K = torch.tensor([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ]).unsqueeze(0).type(torch.float32).to("cpu")

    # ---

    camera_views, visibility_matrix = extract_3d_from_pairs(matching_pairs, imgs_path, height, K)

    camera_extrinsic = torch.cat([c["extrinsic"] for c in camera_views])
    camera_intrinsic = torch.cat([c["intrinsic"] for c in camera_views])
    batch_visibility, batch_point3d, batch_point2d, batch_colors = prepare_data(camera_views[:], visibility_matrix,
                                                                                filter_visible=3)

    points_3d = torch.stack([p.mean(dim=0) for p in batch_point3d])
    colors = torch.stack([p.mean(dim=0) for p in batch_colors])

    model, lossgraph = fit(camera_extrinsic, camera_intrinsic, batch_point2d, points_3d, batch_visibility, steps=4000,
        lr=0.005, train_angle=False, camera_indecies_to_train=[], fit_cam_only=False, device=device)

    # visulize

    fig = px.line(x=torch.arange(len(lossgraph)), y=lossgraph, title="training loss")
    fig.show()

    with torch.inference_mode():
        r_display = [r for r in params_to_extrinsic(model.extrinsics_params)[:, :-1, :3].squeeze().clone().detach().cpu()]
        t_display = [t for t in params_to_extrinsic(model.extrinsics_params)[:, :-1, 3:].squeeze().clone().detach().cpu()]
        points3d_model = model.points3d.clone().detach().cpu().numpy()

    show(r_display, t_display, points3d_model, colors, onlypoints=False)


if __name__ == "__main__":
    main()
