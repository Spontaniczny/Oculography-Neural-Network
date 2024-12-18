import numpy as np
import cv2
import math
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF

class EllipseManager:
    def __init__(self):
        self.reset()
        self.frame_height = None
        self.frame_width = None
        self.ellipse_alpha = 127  # Default alpha

    def set_frame_size(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def set_alpha(self, alpha_value):
        self.ellipse_alpha = alpha_value

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
        self.drawing_points = []  # Points collected when in 'Draw Points' mode

    def delete_ellipse_or_points(self):
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

    def get_pixmap(self, frame, drawing_mode):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        qimage = QImage(frame_rgb.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        painter.setOpacity(0.5)
        if drawing_mode == 'ellipse':
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
        elif drawing_mode == 'points':
            # Draw points
            painter.setBrush(QColor(255, 0, 0, 255))
            painter.setPen(QPen(QColor(255, 0, 0, 255), 2))
            for point in self.drawing_points:
                painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 3, 3)
            # Draw fitted ellipse if available
            if self.ellipse_center and self.ellipse_size:
                if sum(self.ellipse_size) > 1:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(0, 255, 0, 255), 2))
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
        painter.end()
        return pixmap

    def get_frame_pixmap(self, frame):
        # Convert just the frame to a QPixmap without drawing the ellipse or points
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        qimage = QImage(frame_rgb.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def draw_overlay_on_pixmap(self, pixmap, scale_x, scale_y, drawing_mode):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if drawing_mode == 'ellipse' and self.ellipse_center and self.ellipse_size and sum(self.ellipse_size) > 1:
            disp_cx = self.ellipse_center[0] / scale_x
            disp_cy = self.ellipse_center[1] / scale_y
            disp_axes_x = self.ellipse_size[0] / scale_x
            disp_axes_y = self.ellipse_size[1] / scale_y

            painter.setBrush(QColor(255, 255, 255, self.ellipse_alpha))
            painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
            rect = QRectF(disp_cx - disp_axes_x, disp_cy - disp_axes_y, 2 * disp_axes_x, 2 * disp_axes_y)
            painter.save()
            painter.translate(disp_cx, disp_cy)
            painter.rotate(self.ellipse_angle)
            painter.translate(-disp_cx, -disp_cy)
            painter.drawEllipse(rect)
            painter.restore()

            # Draw control points
            self.draw_control_points_on_scaled(painter, scale_x, scale_y)

        elif drawing_mode == 'points':
            # Draw points in scaled coordinates
            painter.setBrush(QColor(255, 0, 0, 255))
            painter.setPen(QPen(QColor(255, 0, 0, 255), 2))
            for p in self.drawing_points:
                disp_x = p[0] / scale_x
                disp_y = p[1] / scale_y
                painter.drawEllipse(QPointF(disp_x, disp_y), 3, 3)

            if self.ellipse_center and self.ellipse_size and sum(self.ellipse_size) > 1:
                disp_cx = self.ellipse_center[0] / scale_x
                disp_cy = self.ellipse_center[1] / scale_y
                disp_axes_x = self.ellipse_size[0] / scale_x
                disp_axes_y = self.ellipse_size[1] / scale_y

                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(0, 255, 0, 255), 2))
                rect = QRectF(disp_cx - disp_axes_x, disp_cy - disp_axes_y, 2 * disp_axes_x, 2 * disp_axes_y)
                painter.save()
                painter.translate(disp_cx, disp_cy)
                painter.rotate(self.ellipse_angle)
                painter.translate(-disp_cx, -disp_cy)
                painter.drawEllipse(rect)
                painter.restore()

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

    def draw_control_points_on_scaled(self, painter, scale_x, scale_y):
        if not self.ellipse_center or not self.ellipse_size:
            return
        corners = self.calculate_corner_control_points()
        cx, cy = self.ellipse_center

        # Scale and rotate corners for drawing
        painter.setBrush(QColor(0, 0, 255, 255))
        painter.setPen(Qt.NoPen)
        for p in corners:
            disp_x = p[0] / scale_x
            disp_y = p[1] / scale_y
            painter.drawEllipse(QPointF(disp_x, disp_y), 5, 5)

        if self.rotation_handle:
            disp_rx = self.rotation_handle[0] / scale_x
            disp_ry = self.rotation_handle[1] / scale_y
            painter.setBrush(QColor(255, 0, 0, 255))
            painter.drawEllipse(QPointF(disp_rx, disp_ry), 7, 7)

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
        self.rotation_handle = self.rotate_point((top_center[0], top_center[1] - 50), (cx, cy), self.ellipse_angle)
        return rotated_corners

    def rotate_point(self, point, center, angle):
        px, py = point
        cx, cy = center
        angle_rad = math.radians(angle)
        qx = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
        qy = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)
        return qx, qy

    def rotate_point_displayed(self, point, center, angle):
        # Rotate a point in display coords
        px, py = point
        cx, cy = center
        angle_rad = math.radians(angle)
        qx = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
        qy = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)
        return qx, qy

    def inverse_rotate_point(self, point, center, angle):
        return self.rotate_point(point, center, -angle)

    def mousePressEvent(self, event, label_pos, drawing_mode):
        pos = event.pos() - label_pos  # pos is now in original coordinates if you've fixed scaling in media_editor

        # Recalculate control_points and rotation_handle before checking for hits
        if self.ellipse_center and self.ellipse_size and sum(self.ellipse_size) > 1:
            self.control_points = self.calculate_corner_control_points()
            # calculate_corner_control_points sets self.rotation_handle too

        if drawing_mode == 'ellipse':
            # Check if user clicked rotation handle
            if self.rotation_handle and abs(pos.x() - self.rotation_handle[0]) < 10 and abs(
                    pos.y() - self.rotation_handle[1]) < 10:
                self.is_rotating = True
                self.start_point = pos
                self.initial_angle = math.degrees(math.atan2(self.start_point.y() - self.ellipse_center[1],
                                                             self.start_point.x() - self.ellipse_center[0]))
                return

            # Check if user clicked on a control point
            for idx, point in enumerate(self.control_points):
                if abs(pos.x() - point[0]) < 10 and abs(pos.y() - point[1]) < 10:
                    self.is_dragging_point = True
                    self.dragged_point_idx = idx
                    return

            # Check if user clicked inside ellipse to drag
            if self.ellipse_center and self.is_point_inside_ellipse(pos):
                self.is_dragging_ellipse = True
                self.start_point = pos
            elif event.button() == Qt.LeftButton:
                # Start drawing a new ellipse
                self.start_point = pos
                self.is_drawing = True
        elif drawing_mode == 'points':
            if event.button() == Qt.LeftButton:
                self.drawing_points.append((pos.x(), pos.y()))


    def mouseMoveEvent(self, event, label_pos, drawing_mode):
        pos = event.pos() - label_pos
        if drawing_mode == 'ellipse':
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
        elif drawing_mode == 'points':
            pass  # No dragging in points mode

    def mouseReleaseEvent(self, event, drawing_mode):
        if drawing_mode == 'ellipse':
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
        elif drawing_mode == 'points':
            pass  # No action needed on mouse release in points mode

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

        # Opposite corners map for the four corners:
        # [top-left (0), top-right (1), bottom-left (2), bottom-right (3)]
        opposite_idx_map = {0: 3, 1: 2, 2: 1, 3: 0}
        opposite_idx = opposite_idx_map[self.dragged_point_idx]

        # Get the current rotated corner points in global coordinates
        current_corners = self.calculate_corner_control_points()
        fixed_corner_global = current_corners[opposite_idx]
        dragged_corner_global = (pos.x(), pos.y())

        old_cx, old_cy = self.ellipse_center
        angle = self.ellipse_angle

        # To transform points into the ellipse's local space (no rotation, ellipse center at (0,0)):
        # Steps to get local coords:
        # 1. Inverse rotate the point around the old ellipse center by -angle.
        # 2. Translate so that old ellipse center becomes (0,0).

        def to_local(global_pt):
            # First inverse rotate around old center, removing angle
            inv_rot = self.inverse_rotate_point(global_pt, (old_cx, old_cy), angle)
            # Now translate so that old_cx, old_cy is the origin
            return (inv_rot[0] - old_cx, inv_rot[1] - old_cy)

        fixed_corner_local = to_local(fixed_corner_global)
        dragged_corner_local = to_local(dragged_corner_global)

        # In local coordinates, determine the new ellipse center and size
        new_local_cx = (fixed_corner_local[0] + dragged_corner_local[0]) / 2.0
        new_local_cy = (fixed_corner_local[1] + dragged_corner_local[1]) / 2.0
        new_width = abs(dragged_corner_local[0] - fixed_corner_local[0]) / 2.0
        new_height = abs(dragged_corner_local[1] - fixed_corner_local[1]) / 2.0

        # Now we have the new ellipse parameters in local coords.
        # To go back to global coordinates:
        # local = (x_local, y_local) is after removing old_cx, old_cy and inverse rotation.
        # Reverse steps:
        # 1. Translate local center back by old_cx, old_cy
        lx = old_cx + new_local_cx
        ly = old_cy + new_local_cy
        # 2. Rotate this point around old center by +angle (the opposite of what we did to go local).
        new_global_center = self.rotate_point((lx, ly), (old_cx, old_cy), angle)

        self.ellipse_center = new_global_center
        self.ellipse_size = (new_width, new_height)

    def is_point_inside_ellipse(self, point):
        if not self.ellipse_center or not self.ellipse_size:
            return False
        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size
        if not axes_x or not axes_y:
            return False
        rel_x, rel_y = self.rotate_point((point.x(), point.y()), (cx, cy), -self.ellipse_angle)
        rel_x, rel_y = rel_x - cx, rel_y - cy
        return (rel_x ** 2) / (axes_x ** 2) + (rel_y ** 2) / (axes_y ** 2) <= 1

    def fit_ellipse_to_points(self):
        if len(self.drawing_points) < 5:
            print("At least 5 points are required to fit an ellipse.")
            return
        # Prepare data for cv2.fitEllipse
        points_array = np.array(self.drawing_points, dtype=np.int32)
        if len(points_array.shape) == 2:
            points_array = points_array.reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(points_array)
        (x, y), (MA, ma), angle = ellipse
        self.ellipse_center = (x, y)
        self.ellipse_size = (MA / 2, ma / 2)  # OpenCV returns full lengths, we need radii
        self.ellipse_angle = angle
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
