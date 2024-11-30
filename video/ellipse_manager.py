import numpy as np
import cv2
import math
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint, QRectF

class EllipseManager:
    def __init__(self):
        self.reset()
        self.frame_height = None
        self.frame_width = None

    def set_frame_size(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def reset(self):
        self.ellipse_center = None
        self.ellipse_size = None
        self.ellipse_angle = 0
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.is_dragging_ellipse = False
        self.is_dragging_point = False
        self.is_rotating = False
        self.initial_angle = None
        self.dragged_point_idx = None
        self.binary_mask = None
        self.control_points = []
        self.rotation_handle = None

    def delete_ellipse(self):
        self.reset()

    def has_ellipse(self):
        return self.ellipse_center is not None

    def get_ellipse_info(self):
        return {
            'center_x': self.ellipse_center[0],
            'center_y': self.ellipse_center[1],
            'size_x': self.ellipse_size[0],
            'size_y': self.ellipse_size[1],
            'angle': self.ellipse_angle
        }

    def get_pixmap(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        qimage = QImage(frame_rgb.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        painter.setOpacity(0.5)
        if self.ellipse_center and self.ellipse_size:
            if sum(self.ellipse_size) > 1:
                painter.setBrush(QColor(255, 255, 255, 127))
                painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                rect = QRectF(
                    int(self.ellipse_center[0] - self.ellipse_size[0]),
                    int(self.ellipse_center[1] - self.ellipse_size[1]),
                    int(2 * self.ellipse_size[0]), int(2 * self.ellipse_size[1])
                )
                painter.save()
                painter.translate(int(self.ellipse_center[0]), int(self.ellipse_center[1]))
                painter.rotate(self.ellipse_angle)
                painter.translate(-int(self.ellipse_center[0]), -int(self.ellipse_center[1]))
                painter.drawEllipse(rect)
                painter.restore()
                self.draw_control_points(painter)
        painter.end()
        return pixmap

    def draw_control_points(self, painter):
        self.control_points = self.calculate_corner_control_points()
        painter.setBrush(QColor(0, 0, 255, 255))
        for point in self.control_points:
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 5, 5)
        if self.rotation_handle:
            painter.setBrush(QColor(255, 0, 0, 255))
            painter.drawEllipse(QPoint(int(self.rotation_handle[0]), int(self.rotation_handle[1])), 7, 7)

    def calculate_corner_control_points(self):
        if not self.ellipse_center or not self.ellipse_size:
            return []
        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size
        corners = [(cx - axes_x, cy - axes_y),
                   (cx + axes_x, cy - axes_y),
                   (cx - axes_x, cy + axes_y),
                   (cx + axes_x, cy + axes_y)]
        rotated_corners = [self.rotate_point(p, (cx, cy), self.ellipse_angle) for p in corners]
        top_center = (cx, cy - axes_y)
        rotation_handle = (top_center[0], top_center[1] - 50)
        self.rotation_handle = self.rotate_point(rotation_handle, (cx, cy), self.ellipse_angle)
        return rotated_corners

    def rotate_point(self, point, center, angle):
        px, py = point
        cx, cy = center
        angle_rad = math.radians(angle)
        qx = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
        qy = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)
        return qx, qy

    def inverse_rotate_point(self, point, center, angle):
        return self.rotate_point(point, center, -angle)

    def mousePressEvent(self, event, label_pos):
        pos = event.pos() - label_pos
        if self.rotation_handle and abs(pos.x() - self.rotation_handle[0]) < 10 and abs(pos.y() - self.rotation_handle[1]) < 10:
            self.is_rotating = True
            self.start_point = pos
            self.initial_angle = math.degrees(math.atan2(self.start_point.y() - self.ellipse_center[1],
                                                         self.start_point.x() - self.ellipse_center[0]))
            return
        for idx, point in enumerate(self.control_points):
            if abs(pos.x() - point[0]) < 10 and abs(pos.y() - point[1]) < 10:
                self.is_dragging_point = True
                self.dragged_point_idx = idx
                return
        if self.ellipse_center and self.is_point_inside_ellipse(pos):
            self.is_dragging_ellipse = True
            self.start_point = pos
        elif event.button() == Qt.LeftButton:
            self.start_point = pos
            self.is_drawing = True

    def mouseMoveEvent(self, event, label_pos):
        pos = event.pos() - label_pos
        if self.is_drawing:
            self.end_point = pos
            self.update_drawing_ellipse()
        elif self.is_dragging_point:
            self.adjust_ellipse_from_drag(pos)
        elif self.is_rotating:
            current_angle = math.degrees(math.atan2(pos.y() - self.ellipse_center[1],
                                                    pos.x() - self.ellipse_center[0]))
            angle_difference = current_angle - self.initial_angle
            self.ellipse_angle += angle_difference
            self.initial_angle = current_angle
        elif self.is_dragging_ellipse:
            delta_x = pos.x() - self.start_point.x()
            delta_y = pos.y() - self.start_point.y()
            self.ellipse_center = (self.ellipse_center[0] + delta_x, self.ellipse_center[1] + delta_y)
            self.start_point = pos

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_drawing:
                self.is_drawing = False
                self.finalize_ellipse()
            elif self.is_dragging_point:
                self.is_dragging_point = False
                self.finalize_ellipse()
            elif self.is_rotating:
                self.is_rotating = False
                self.finalize_ellipse()
            elif self.is_dragging_ellipse:
                self.is_dragging_ellipse = False
                self.finalize_ellipse()

    def update_drawing_ellipse(self):
        if self.start_point and self.end_point:
            center = (
                (self.start_point.x() + self.end_point.x()) // 2, (self.start_point.y() + self.end_point.y()) // 2)
            size = (
                abs(self.start_point.x() - self.end_point.x()) // 2,
                abs(self.start_point.y() - self.end_point.y()) // 2)
            self.ellipse_center = center
            self.ellipse_size = size

    def finalize_ellipse(self):
        if self.ellipse_center and self.ellipse_size:
            if sum(self.ellipse_size) > 1:
                # Create binary mask
                self.binary_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                cv2.ellipse(
                    self.binary_mask,
                    (int(self.ellipse_center[0]), int(self.ellipse_center[1])),
                    (int(self.ellipse_size[0]), int(self.ellipse_size[1])),
                    self.ellipse_angle,
                    0,
                    360,
                    [255, 255, 255],
                    -1
                )

    def adjust_ellipse_from_drag(self, pos):
        if self.dragged_point_idx is None or not self.ellipse_center or not self.ellipse_size:
            return
        unrotated_pos = self.inverse_rotate_point((pos.x(), pos.y()), self.ellipse_center, self.ellipse_angle)
        cx, cy = self.ellipse_center
        new_width = abs(unrotated_pos[0] - cx)
        new_height = abs(unrotated_pos[1] - cy)
        self.ellipse_size = (new_width, new_height)

    def is_point_inside_ellipse(self, point):
        if not self.ellipse_center or not self.ellipse_size:
            return False
        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size
        if not axes_x or not axes_y:
            return True
        rel_x, rel_y = self.rotate_point((point.x(), point.y()), (cx, cy), -self.ellipse_angle)
        rel_x, rel_y = rel_x - cx, rel_y - cy
        return (rel_x ** 2) / (axes_x ** 2) + (rel_y ** 2) / (axes_y ** 2) <= 1


