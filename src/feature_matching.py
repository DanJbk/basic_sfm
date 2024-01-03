import glob
import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kornia as K
import kornia.feature as KF
from tqdm.auto import tqdm
from torchvision import transforms

trans1 = transforms.ToTensor()

def visualize_laf_matches(laf1, laf2, path1, path2):
    """
    Visualize matches between two sets of Local Affine Frames (LAFs).

    Args:
    laf1 (torch.Tensor): Tensor of LAFs for the first image, shape [N, 2].
    laf2 (torch.Tensor): Tensor of LAFs for the second image, shape [N, 2].
    path1 (str): Path to the first image.
    path2 (str): Path to the second image.
    """
    if laf1.size(0) != laf2.size(0):
        raise ValueError("Number of LAFs must be the same for both images.")

    # Load and convert images
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    # Concatenate images side by side
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    combined_img = Image.new('RGB', (total_width, max_height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_img)
    plt.axis('off')

    # Draw lines between matching LAFs
    for i in range(laf1.size(0)):
        x1, y1 = laf1[i].tolist()
        x2, y2 = laf2[i].tolist()
        plt.plot([x1, x2 + img1.width], [y1, y2], color=tuple(torch.rand([3]).tolist()), alpha=1.0, linewidth=1.0)

    plt.show()


def open_file(path, device="cpu"):
    return transforms.functional.pil_to_tensor(Image.open(path).convert("RGB"))[None, ...].to(device) / 255.0


@torch.inference_mode()
def match_images(path1, path2, lg=None, disk=None, thr=0.9, device="cuda"):

    # img1 = transforms.functional.pil_to_tensor(Image.open(path1).convert("RGB"))[None, ...].to(device) / 255.0
    # img2 = transforms.functional.pil_to_tensor(Image.open(path2).convert("RGB"))[None, ...].to(device) / 255.0

    img1 = open_file(path1, device=device)
    img2 = open_file(path2, device=device)

    num_features = 2048
    if disk is None:
        disk = KF.DISK.from_pretrained("depth").to(device)

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)

    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

    # ---

    if lg is None:
        lg = KF.LightGlue("disk").to(device).eval()

    image0 = {
        "keypoints": features1.keypoints[None],
        "descriptors": features1.descriptors[None],
        "image_size": torch.tensor(img1.shape[-2:][::-1]).view(1, 2).to(device),
    }
    image1 = {
        "keypoints": features2.keypoints[None],
        "descriptors": features2.descriptors[None],
        "image_size": torch.tensor(img2.shape[-2:][::-1]).view(1, 2).to(device),
    }

    with torch.inference_mode():
        out = lg({"image0": image0, "image1": image1})
        idxs = out["matches"][0]

        # print(f"{idxs.shape[0]} tentative matches with DISK LightGlue")

    def get_matching_keypoints(kp1, kp2, idxs):
        mkpts1 = kp1[idxs[:, 0]]
        mkpts2 = kp2[idxs[:, 1]]
        return mkpts1, mkpts2


    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

    Fm, inliers = cv2.findFundamentalMat(
        mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    inliers = inliers > 0
    # print(f"{inliers.sum()} inliers with DISK")

    mask = (out['scores'][0].greater(thr).cpu() & inliers[:, 0]).bool()

    return mkpts1[mask], mkpts2[mask], image0, image1


def visualize_laf_matches(laf1, laf2, path1, path2):
    """
    Visualize matches between two sets of Local Affine Frames (LAFs).

    Args:
    laf1 (torch.Tensor): Tensor of LAFs for the first image, shape [N, 2].
    laf2 (torch.Tensor): Tensor of LAFs for the second image, shape [N, 2].
    path1 (str): Path to the first image.
    path2 (str): Path to the second image.
    """
    if laf1.size(0) != laf2.size(0):
        raise ValueError("Number of LAFs must be the same for both images.")

    # Load and convert images
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    # Concatenate images side by side
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    combined_img = Image.new('RGB', (total_width, max_height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_img)
    plt.axis('off')

    # Draw lines between matching LAFs
    for i in range(laf1.size(0)):
        x1, y1 = laf1[i].tolist()
        x2, y2 = laf2[i].tolist()
        plt.plot([x1, x2 + img1.width], [y1, y2], color=tuple(torch.rand([3]).tolist()), alpha=1.0, linewidth=1.0)

    plt.show()


def match_features(imgs_path="data\\images", skip_step=7):

    # imgs_path = "D:\\9.programming\\Plenoxels\\data\\lego\\test"

    def sortfunc(p): return int(p.split("r_")[-1].split(".png")[0].split("_")[0])

    paths = glob.glob(f"{imgs_path}\*")

    paths = sorted(paths, key=sortfunc)
    paths = [p for p in paths if ("depth" not in p) and ("normal" not in p)]
    relevant_paths = paths[::skip_step]

    matching_pairs = []
    device = "cuda"
    disk = KF.DISK.from_pretrained("depth").to(device).eval()
    lg = KF.LightGlue("disk").to(device).eval()

    with torch.inference_mode():
        for i in tqdm(range(len(relevant_paths) - 1), desc="matching features"):
            p1, p2, image0, image1 = match_images(relevant_paths[i], relevant_paths[i + 1], lg=lg, disk=disk, thr=0.7,
                                                  device=device)

            matching_pairs.append({
                "0": {
                    "name": relevant_paths[i].split("\\")[-1],
                    "points": p1.cpu()
                },
                "1": {
                    "name": relevant_paths[i + 1].split("\\")[-1],
                    "points": p2.cpu()
                }
            })

    np.save("matching_pairs_kornia.npy", matching_pairs)

if __name__ == "__main__":
    os.chdir(os.path.join(os.getcwd(), '..'))
    match_features()
