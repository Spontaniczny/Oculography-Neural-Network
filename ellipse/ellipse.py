from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


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
        angle_norm =  self.rotate_angle / 180
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
    