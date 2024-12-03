from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import torch
from skimage.morphology import ellipse as create_ellipse
from skimage.feature import canny


@dataclass(frozen=True)
class Ellipse:
    x_center: float
    y_center: float
    major_axis: float
    minor_axis: float
    rotate_angle: float
    original_img_shape: tuple[int, int]


    def get_parameters(self) -> tuple[float, float, float, float, float]:
        return self.x_center, self.y_center, self.major_axis, self.minor_axis, self.rotate_angle

    def normalize_parameters(self) -> tuple[float, float, float, float, float]:
        h, w = self.original_img_shape
        x_norm = self.x_center / h
        y_norm = self.y_center / w
        major_norm = self.major_axis / h
        minor_norm = self.minor_axis / w
        angle_norm =  torch.pi * ((self.rotate_angle - 180) / 360)
        return x_norm, y_norm, major_norm, minor_norm, angle_norm

    def draw_ellipse_on_image(
            self, 
            mask: np.ndarray, 
            inplace: bool = False,
        ) -> np.ndarray:

        if not inplace:
            mask = np.copy(mask)
        
        cv2.ellipse(
            mask, 
            (int(self.y_center), int(self.x_center)), 
            (int(self.major_axis), int(self.minor_axis)), 
            -self.rotate_angle, 0, 360, 2, 1
        )
        
        return mask
    

    def draw_ellipse(self, target_shape: Optional[tuple[int, int]] = None):
        if target_shape is not None and len(target_shape) > 2:
            raise ValueError("Shape argument should have 2 dimensions")
        
        ellipse_img = np.zeros(self.original_img_shape)
        cv2.ellipse(
            ellipse_img, 
            (int(self.y_center), int(self.x_center)),
            (int(self.major_axis), int(self.minor_axis)), 
            -self.rotate_angle, 360, 0, 2, -1
        )

        if target_shape is not None:
            ellipse_img = cv2.resize(ellipse_img, target_shape, 0, 0, interpolation = cv2.INTER_NEAREST)
            
        return ellipse_img


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

def find_ellipse(mask: torch.Tensor, denoise: bool = True) -> tuple[torch.Tensor, Ellipse]:
    mask = mask.squeeze().numpy()
    denoised_mask = get_rid_of_noise(mask) if denoise else mask
    ellipse = fit_ellipse(denoised_mask)
    mask = ellipse.draw_ellipse()
    tensor_ellipse = torch.Tensor(mask).unsqueeze(0)
    tensor_mask = tensor_ellipse > 0.5
    return tensor_mask, ellipse
