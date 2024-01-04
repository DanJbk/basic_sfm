import argparse
import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np

from src.feature_matching import match_features
from src.visibility_utils import show
from src.utils import params_to_extrinsic
from src.model import extract_3d_from_pairs, prepare_data, fit

import plotly
import plotly.express as px
import plotly.graph_objs as go


def main(camera_angle_x, width, height, imgs_path, matching_pairs_path, visibility_thrshold=6, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # camera parameters
    matching_pairs = np.load(matching_pairs_path, allow_pickle=True)

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
    batch_visibility, batch_point3d, batch_point2d, batch_colors = prepare_data(
        camera_views[:],
        visibility_matrix,
        filter_visible=visibility_thrshold
    )

    points_3d = torch.stack([p.mean(dim=0) for p in batch_point3d])
    colors = torch.stack([p.mean(dim=0) for p in batch_colors])

    labels = [batch_point2d[batch_visibility[:, index], index].to(device) for index in range(batch_visibility.shape[1])]

    # remove cameras with no relevant points
    cameras_with_points_indices = torch.tensor([len(l) for l in labels]).greater(0)
    camera_extrinsic = camera_extrinsic[cameras_with_points_indices]
    camera_intrinsic = camera_intrinsic[cameras_with_points_indices]
    batch_visibility = batch_visibility[:, cameras_with_points_indices]
    labels = [l for l in labels if len(l) > 0]

    model, lossgraph = fit(
        camera_extrinsic,
        camera_intrinsic,
        labels,
        points_3d,
        batch_visibility,
        width=width,
        height=height,
        steps=6000,
        lr=0.005,
        train_angle=False,
        camera_indecies_to_train=[],
        fit_cam_only=False,
        device=device
    )

    # visualize

    fig = px.line(x=torch.arange(len(lossgraph)), y=lossgraph, title="training loss")
    fig.show()

    with torch.inference_mode():
        r_display = [r for r in params_to_extrinsic(model.extrinsics_params)[:, :-1, :3].squeeze().clone().detach().cpu()]
        t_display = [t for t in params_to_extrinsic(model.extrinsics_params)[:, :-1, 3:].squeeze().clone().detach().cpu()]
        points3d_model = model.points3d.clone().detach().cpu().numpy()

    np.save(
        "bundle_adjusted",
        {
            "extrinsic_matrices": params_to_extrinsic(model.extrinsics_params).clone().detach().cpu().numpy(),
            "intrinsic_matrices": model.get_intrinsics().clone().detach().cpu().numpy(),
            "3d_points": model.points3d.clone().detach().cpu().numpy()
        },
        allow_pickle=True
    )

    show(r_display, t_display, points3d_model, colors, onlypoints=False)


def __main__():
    parser = argparse.ArgumentParser(description='Basic sfm pipeline.')

    # Add arguments
    parser.add_argument('--camera_angle_x', help='camera angle in the x axis',
                        type=int, default=0.6911112070083618)
    parser.add_argument('--width', help='width of image', type=int, default=800)
    parser.add_argument('--height', help='height of image', type=int, default=800)
    parser.add_argument('--visibility_thr', help='filter points that are not visible to a certain amount of cameras'
                        , type=int, default=7)
    parser.add_argument('--imgs_path', help='path of images', type=str, default="data\\images")

    # Parse the arguments
    args = parser.parse_args()

    match_features(imgs_path=args.imgs_path, skip_step=2)
    main(args.camera_angle_x, args.width, args.height, args.imgs_path, matching_pairs_path="matching_pairs_kornia.npy",
         visibility_thrshold=args.visibility_thr, device=None)


if __name__ == "__main__":
    __main__()

