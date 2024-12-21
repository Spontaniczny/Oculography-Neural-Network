import cv2
import numpy as np
import torch
from skimage.morphology import ellipse as create_ellipse
from skimage.feature import canny
from .ellipse import Ellipse
from typing import Tuple, Optional


def find_circle_radius(mask: np.ndarray) -> int:
    mass = np.sum(mask)
    inner_circle_mass = mass / 8
    radius = np.sqrt(inner_circle_mass / np.pi)
    return int(radius)

def get_rid_of_noise(mask: np.ndarray) -> np.ndarray:
    radius = find_circle_radius(mask)
    circle = create_ellipse(radius, radius).astype("uint8")
    closed_img = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, circle)  # needs to be done first on mask
    opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, circle)
    opened_img = opened_img.astype("int32")
    return opened_img


def find_convex_hull(mask: np.ndarray) -> np.ndarray:
    all_points = np.column_stack(np.where(mask > 0.5))
    edge_points = cv2.convexHull(all_points)
    return edge_points


def find_outline(mask: np.ndarray) -> np.ndarray:
    edges = canny(mask)
    edge_points = np.column_stack(np.where(edges > 0.5))
    return edge_points

def fit_ellipse(
        mask: np.ndarray, 
        edge_detection: str = "convex_hull"
    ) -> Ellipse:
    
    if edge_detection not in ["convex_hull", "canny"]:
        raise ValueError("Edge detection method should be one of ['convex_hull', 'canny']")
    if edge_detection == "convex_hull":
        edge_points = find_convex_hull(mask)
    else:
        edge_points = find_outline(mask)
    
    if edge_points is None or len(edge_points) == 0:
        return Ellipse(mask.shape[0] / 2, mask.shape[1] / 2, 0, 0, 0, mask.shape)
    
    (x_center, y_center), (minor_axis, major_axis), angle = cv2.fitEllipse(edge_points)

    minor_axis, major_axis = minor_axis / 2, major_axis / 2
    ellipse = Ellipse(x_center, y_center, major_axis, minor_axis, angle, mask.shape)
    return ellipse

def remove_blobs_and_fill_areas(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    if len(stats) <= 1:
        return mask
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    largest_blob = np.zeros_like(mask, dtype=np.uint8)
    largest_blob[labels == largest_label] = 255

    contours, _ = cv2.findContours(largest_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(largest_blob)
    cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)

    return filled_image


def remove_noise(mask: torch.Tensor, find_ellipse: bool = True) -> Tuple[torch.Tensor, Optional[Ellipse]]:
    mask = mask.squeeze().numpy()

    # denoised_mask = get_rid_of_noise(mask) if denoise else mask
    denoised_mask = remove_blobs_and_fill_areas(mask)

    if not find_ellipse:
        tensor_mask = torch.Tensor(denoised_mask).unsqueeze(0)
        tensor_mask = tensor_mask > 0.5
        return tensor_mask, None

    ellipse = fit_ellipse(denoised_mask)
    mask = ellipse.draw_ellipse()
    tensor_ellipse = torch.Tensor(mask).unsqueeze(0)
    tensor_mask = tensor_ellipse > 0.5
    return tensor_mask, ellipse
