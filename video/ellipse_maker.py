import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
import math


class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Editor with Ellipses")
        self.video_path = None
        self.video_cap = None
        self.current_frame = None
        self.frame_count = 0
        self.current_frame_idx = 0
        self.ellipse_center = None  # Center of the ellipse
        self.ellipse_size = None  # Size (axes lengths) of the ellipse
        self.ellipse_angle = 0  # Angle of the ellipse
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.is_dragging_ellipse = False
        self.is_dragging_point = False
        self.is_rotating = False
        self.initial_angle = None  # Initial angle when rotation starts
        self.dragged_point_idx = None
        self.binary_mask = None
        self.control_points = []  # Control points for resizing
        self.rotation_handle = None  # Rotation control handle

        # GUI Elements
        self.video_label = QLabel(self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.change_frame)

        self.frame_input = QLineEdit(self)
        self.frame_input.setPlaceholderText("Go to frame")
        self.frame_input.returnPressed.connect(self.go_to_frame)

        self.frame_label = QLabel(self)

        self.save_button = QPushButton("Save Frame", self)
        self.save_button.clicked.connect(self.save_frame)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)

        self.delete_button = QPushButton("Delete Ellipse", self)
        self.delete_button.clicked.connect(self.delete_ellipse)

        self.max_frames_label = QLabel(self)

        # Layout
        frame_control_layout = QHBoxLayout()
        frame_control_layout.addWidget(self.frame_input)
        frame_control_layout.addWidget(self.max_frames_label)
        frame_control_layout.addWidget(self.frame_label)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)
        layout.addLayout(frame_control_layout)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.delete_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setMouseTracking(True)
        self.video_label.setMouseTracking(True)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.frame_count - 1)
            self.max_frames_label.setText(f"/ {self.frame_count} frames")
            self.change_frame(0)

    def change_frame(self, frame_idx):
        if not self.video_cap:
            return

        self.current_frame_idx = frame_idx
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.current_frame = self.video_cap.read()

        if ret:
            self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)  # Reset binary mask
            self.ellipse_center = None  # Reset ellipse
            self.update_video_display()

        # Update the current frame label
        self.frame_label.setText(f"Current Frame: {frame_idx}")

    def update_video_display(self):
        # Convert frame to RGB for display in QLabel
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        qimage = QImage(frame_rgb.data, width, height, QImage.Format_RGB888)

        # Draw the ellipse and control points if they exist
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        painter.setOpacity(0.5)  # Make ellipses transparent
        if self.ellipse_center:
            # Draw ellipse
            painter.setBrush(QColor(255, 255, 255, 127))
            painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
            rect = QRectF(
                int(self.ellipse_center[0] - self.ellipse_size[0]), int(self.ellipse_center[1] - self.ellipse_size[1]),
                int(2 * self.ellipse_size[0]), int(2 * self.ellipse_size[1])
            )
            painter.save()
            painter.translate(int(self.ellipse_center[0]), int(self.ellipse_center[1]))
            painter.rotate(self.ellipse_angle)
            painter.translate(-int(self.ellipse_center[0]), -int(self.ellipse_center[1]))
            painter.drawEllipse(rect)
            painter.restore()

            # Draw resizing control points and rotation handle
            self.draw_control_points(painter)

        painter.end()
        self.video_label.setPixmap(pixmap)

    def draw_control_points(self, painter):
        self.control_points = self.calculate_corner_control_points()
        painter.setBrush(QColor(0, 0, 255, 255))  # Blue control points for resizing
        for point in self.control_points:
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 5, 5)  # Small circles for control points

        # Draw the rotation handle (above the ellipse)
        if self.rotation_handle:
            painter.setBrush(QColor(255, 0, 0, 255))  # Red control point for rotation
            painter.drawEllipse(QPoint(int(self.rotation_handle[0]), int(self.rotation_handle[1])), 7, 7)

    def calculate_corner_control_points(self):
        # Calculate the four corners of the bounding box for resizing, and rotation handle
        if not self.ellipse_center or not self.ellipse_size:
            return []

        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size

        # Calculate corners of the bounding box (unrotated)
        corners = [(cx - axes_x, cy - axes_y),  # Top-left
                   (cx + axes_x, cy - axes_y),  # Top-right
                   (cx - axes_x, cy + axes_y),  # Bottom-left
                   (cx + axes_x, cy + axes_y)]  # Bottom-right

        # Rotate the corners with the ellipse
        rotated_corners = [self.rotate_point(p, (cx, cy), self.ellipse_angle) for p in corners]

        # Calculate the rotation handle (above the top center of the ellipse, after rotation)
        top_center = (cx, cy - axes_y)  # Top-center point of the ellipse
        rotation_handle = (top_center[0], top_center[1] - 50)  # 50 pixels above the top-center
        self.rotation_handle = self.rotate_point(rotation_handle, (cx, cy), self.ellipse_angle)

        return rotated_corners

    def rotate_point(self, point, center, angle):
        """ Rotate a point around a center by a given angle. """
        px, py = point
        cx, cy = center
        angle_rad = math.radians(angle)
        qx = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
        qy = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)
        return qx, qy

    def inverse_rotate_point(self, point, center, angle):
        """ Apply the inverse of a rotation to a point, i.e., rotate it back by the given angle. """
        return self.rotate_point(point, center, -angle)

    def mousePressEvent(self, event):
        pos = event.pos() - self.video_label.pos()

        # Check if rotating based on proximity to rotation handle
        if self.rotation_handle and abs(pos.x() - self.rotation_handle[0]) < 10 and abs(pos.y() - self.rotation_handle[1]) < 10:
            self.is_rotating = True
            self.start_point = pos
            self.initial_angle = math.degrees(math.atan2(self.start_point.y() - self.ellipse_center[1],
                                                         self.start_point.x() - self.ellipse_center[0]))
            return

        # Check if clicking on a control point for resizing
        for idx, point in enumerate(self.control_points):
            if abs(pos.x() - point[0]) < 10 and abs(pos.y() - point[1]) < 10:
                self.is_dragging_point = True
                self.dragged_point_idx = idx
                return

        # Check if clicking inside the ellipse to drag it
        if self.ellipse_center and self.is_point_inside_ellipse(pos):
            self.is_dragging_ellipse = True
            self.start_point = pos  # Remember the point where the mouse is pressed
        elif event.button() == Qt.LeftButton:
            # Start drawing a new ellipse
            self.start_point = event.pos() - self.video_label.pos()
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        pos = event.pos() - self.video_label.pos()

        if self.is_drawing:
            # Update the end point and redraw the ellipse
            self.end_point = pos
            self.update_drawing_ellipse()
        elif self.is_dragging_point:
            # Dragging corner control point to resize the ellipse
            self.adjust_ellipse_from_drag(pos)
        elif self.is_rotating:
            # Update the rotation based on mouse movement
            current_angle = math.degrees(math.atan2(pos.y() - self.ellipse_center[1],
                                                    pos.x() - self.ellipse_center[0]))
            angle_difference = current_angle - self.initial_angle
            self.ellipse_angle += angle_difference
            self.initial_angle = current_angle  # Update the initial angle for the next movement
            self.update_video_display()
        elif self.is_dragging_ellipse:
            # Update the ellipse position when dragging the whole ellipse
            delta_x = pos.x() - self.start_point.x()
            delta_y = pos.y() - self.start_point.y()
            self.ellipse_center = (self.ellipse_center[0] + delta_x, self.ellipse_center[1] + delta_y)
            self.start_point = pos  # Update the start point to the current position
            self.update_video_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_drawing:
                self.is_drawing = False
                self.finalize_ellipse()
            elif self.is_dragging_point:
                self.is_dragging_point = False
            elif self.is_rotating:
                self.is_rotating = False
            elif self.is_dragging_ellipse:
                self.is_dragging_ellipse = False

    def update_drawing_ellipse(self):
        center = ((self.start_point.x() + self.end_point.x()) // 2, (self.start_point.y() + self.end_point.y()) // 2)
        size = (abs(self.start_point.x() - self.end_point.x()) // 2, abs(self.start_point.y() - self.end_point.y()) // 2)

        self.ellipse_center = center
        self.ellipse_size = size
        self.update_video_display()

    def finalize_ellipse(self):
        if self.ellipse_center and self.ellipse_size:
            # Update the binary mask (clear previous ellipses)
            self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            cv2.ellipse(self.binary_mask, self.ellipse_center, self.ellipse_size, self.ellipse_angle, 0, 360, 255, -1)

    def adjust_ellipse_from_drag(self, pos):
        if self.dragged_point_idx is None or not self.ellipse_center or not self.ellipse_size:
            return

        # Convert the mouse position back to the unrotated space of the ellipse
        unrotated_pos = self.inverse_rotate_point((pos.x(), pos.y()), self.ellipse_center, self.ellipse_angle)

        # Adjust the size based on the unrotated mouse position
        cx, cy = self.ellipse_center
        new_width = abs(unrotated_pos[0] - cx)
        new_height = abs(unrotated_pos[1] - cy)

        # Update the ellipse size
        self.ellipse_size = (new_width, new_height)
        self.update_video_display()

    def update_rotation(self, pos):
        """ Update the rotation angle based on the mouse position. """
        cx, cy = self.ellipse_center
        angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
        self.ellipse_angle = angle
        self.update_video_display()

    def is_point_inside_ellipse(self, point):
        """ Check if a point is inside the ellipse """
        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size

        # Calculate the point's position relative to the ellipse's center and apply the rotation
        rel_x, rel_y = self.rotate_point((point.x(), point.y()), (cx, cy), -self.ellipse_angle)
        rel_x, rel_y = rel_x - cx, rel_y - cy

        # Check if the point is inside the ellipse using the ellipse equation
        return (rel_x ** 2) / (axes_x ** 2) + (rel_y ** 2) / (axes_y ** 2) <= 1

    def delete_ellipse(self):
        self.ellipse_center = None
        self.ellipse_size = None
        self.control_points = []
        self.rotation_handle = None
        self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        self.update_video_display()

    def save_frame(self):
        if self.current_frame is not None and self.ellipse_center is not None:
            # Create the filenames for the current frame
            frame_filename = f"{self.video_path.rsplit('.', 1)[0]}_frame_{self.current_frame_idx}.png"
            binary_filename = f"{self.video_path.rsplit('.', 1)[0]}_frame_{self.current_frame_idx}_binary.png"

            # Save the current frame with ellipses
            cv2.imwrite(frame_filename, self.current_frame)

            # Save the binary mask
            cv2.imwrite(binary_filename, self.binary_mask)

            print(f"Saved {frame_filename} and {binary_filename}")

    def go_to_frame(self):
        frame_num = int(self.frame_input.text())
        if 0 <= frame_num < self.frame_count:
            self.slider.setValue(frame_num)
            self.change_frame(frame_num)


# Run the PyQt application
app = QApplication(sys.argv)
window = VideoEditor()
window.show()
sys.exit(app.exec_())
