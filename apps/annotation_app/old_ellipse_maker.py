import csv
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
import math
import os

class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_variables()
        self.setup_gui_elements()
        self.setup_layout()
        self.setMouseTracking(True)
        self.video_label.setMouseTracking(True)

    def initialize_variables(self):
        self.setWindowTitle("Video Frame Editor with Ellipses")
        self.video_path = None
        self.video_name = None
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

    def setup_gui_elements(self):
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

        self.prev_frame_button = QPushButton("<", self)
        self.prev_frame_button.setFixedWidth(30)
        self.prev_frame_button.clicked.connect(self.prev_frame)

        self.next_frame_button = QPushButton(">", self)
        self.next_frame_button.setFixedWidth(30)
        self.next_frame_button.clicked.connect(self.next_frame)

    def setup_layout(self):
        frame_control_layout = QHBoxLayout()
        frame_control_layout.addWidget(self.prev_frame_button)
        frame_control_layout.addWidget(self.frame_input)
        frame_control_layout.addWidget(self.max_frames_label)
        frame_control_layout.addWidget(self.frame_label)
        frame_control_layout.addWidget(self.next_frame_button)

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

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.frame_count - 1)
            self.max_frames_label.setText(f"/ {self.frame_count} frames")
            self.change_frame(0)

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.slider.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        if self.current_frame_idx < self.frame_count - 1:
            self.slider.setValue(self.current_frame_idx + 1)

    def change_frame(self, frame_idx):
        if not self.video_cap:
            return

        self.current_frame_idx = frame_idx
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.current_frame = self.video_cap.read()

        if ret:
            # self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)  # Reset binary mask
            # self.ellipse_center = None  # Reset ellipse
            self.update_video_display()

        # Update the current frame label
        self.frame_label.setText(f"Current Frame: {frame_idx}")

    def update_video_display(self):
        if self.current_frame is None:
            return

        # Convert frame to RGB for display in QLabel
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        qimage = QImage(frame_rgb.data, width, height, QImage.Format_RGB888)

        # Draw the ellipse and control points if they exist
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        painter.setOpacity(0.5)  # Make ellipses transparent
        if self.ellipse_center and self.ellipse_size:
            if sum(self.ellipse_size) > 1:  # Ensure the ellipse size is valid
                # Draw ellipse
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
        if self.rotation_handle and abs(pos.x() - self.rotation_handle[0]) < 10 and abs(
                pos.y() - self.rotation_handle[1]) < 10:
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

            # Avoid zero-sized or negative-sized ellipses by enforcing a minimum size
            # min_size = 10
            # if size[0] < min_size:
            #     size = (min_size, size[1])
            # if size[1] < min_size:
            #     size = (size[0], min_size)

            self.ellipse_center = center
            self.ellipse_size = size
            self.update_video_display()

    def finalize_ellipse(self):
        if self.ellipse_center and self.ellipse_size:
            # Ensure the ellipse size is valid
            if sum(self.ellipse_size) > 1:
                # Avoid finalizing with too-small ellipses to prevent crashes
                # min_size = 10
                # if self.ellipse_size[0] < min_size or self.ellipse_size[1] < min_size:
                #     print("Ellipse size too small, skipping finalization.")
                #     return

                # Update the binary mask (clear previous ellipses)
                self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
                cv2.ellipse(
                    self.binary_mask,
                    (int(self.ellipse_center[0]), int(self.ellipse_center[1])),  # Center in integer form
                    (int(self.ellipse_size[0]), int(self.ellipse_size[1])),  # Axes in integer form
                    self.ellipse_angle,  # Angle for rotation
                    0,  # Starting angle of the arc
                    360,  # Ending angle of the arc
                    [255, 255, 255],  # White color for the mask
                    -1  # Thickness (-1 to fill the ellipse)
                )
            else:
                print("Invalid ellipse size, skipping finalization.")

    def adjust_ellipse_from_drag(self, pos):
        if self.dragged_point_idx is None or not self.ellipse_center or not self.ellipse_size:
            return

        # Convert the mouse position back to the unrotated space of the ellipse
        unrotated_pos = self.inverse_rotate_point((pos.x(), pos.y()), self.ellipse_center, self.ellipse_angle)

        # Adjust the size based on the unrotated mouse position
        cx, cy = self.ellipse_center
        new_width = abs(unrotated_pos[0] - cx)
        new_height = abs(unrotated_pos[1] - cy)

        # Ensure that the width and height remain greater than a minimum value to avoid zero or negative sizes
        # min_size = 10  # You can set this value based on your needs
        # if new_width < min_size:
        #     new_width = min_size
        # if new_height < min_size:
        #     new_height = min_size

        # Update the ellipse size
        self.ellipse_size = (new_width, new_height)
        self.update_video_display()

    def update_rotation(self, pos):
        """ Update the rotation angle based on the mouse position. """
        if not self.ellipse_center:
            return
        cx, cy = self.ellipse_center
        angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
        self.ellipse_angle = angle
        self.update_video_display()

    def is_point_inside_ellipse(self, point):
        """ Check if a point is inside the ellipse """
        if not self.ellipse_center or not self.ellipse_size:
            return False

        cx, cy = self.ellipse_center
        axes_x, axes_y = self.ellipse_size

        # Avoid division by zero, always enable moving the ellipse
        if not axes_x or not axes_y:
            return True

        # Calculate the point's position relative to the ellipse's center and apply the rotation
        rel_x, rel_y = self.rotate_point((point.x(), point.y()), (cx, cy), -self.ellipse_angle)
        rel_x, rel_y = rel_x - cx, rel_y - cy

        # Check if the point is inside the ellipse using the ellipse equation
        return (rel_x ** 2) / (axes_x ** 2) + (rel_y ** 2) / (axes_y ** 2) <= 1

    def delete_ellipse(self):
        self.ellipse_center = None
        self.ellipse_size = None
        self.ellipse_angle = 0
        self.control_points = []
        self.rotation_handle = None
        self.binary_mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        self.update_video_display()

    def save_frame(self):
        if self.current_frame is not None and self.ellipse_center is not None:
            # Create directories if they do not exist
            data_dir = os.path.join(os.path.dirname(self.video_path), 'data')
            frames_dir = os.path.join(data_dir, 'frames')
            annotations_dir = os.path.join(data_dir, 'annotations')
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)

            # Create the filenames for the current frame
            frame_filename = os.path.join(frames_dir, f"{self.video_name}_frame_{self.current_frame_idx}.png")
            binary_filename = os.path.join(annotations_dir, f"{self.video_name}_frame_{self.current_frame_idx}.png")

            # Save the current frame with ellipses
            cv2.imwrite(frame_filename, self.current_frame)

            # Save the binary mask
            cv2.imwrite(binary_filename, self.binary_mask)

            self.save_to_csv(data_dir)

            print(f"Saved {frame_filename} and {binary_filename}")

    def save_to_csv(self, data_dir):
        csv_filename = os.path.join(data_dir, 'ellipse_info.csv')
        rows = []
        new_entry = [
            f"{self.video_name}_frame_{self.current_frame_idx}",
            self.ellipse_center[0],
            self.ellipse_center[1],
            self.ellipse_size[0],
            self.ellipse_size[1],
            self.ellipse_angle
        ]

        if not os.path.exists(csv_filename):
            rows.append(["frame_name", "center_x", "center_y", "size_x", "size_y", "angle"])

        # Read existing entries
        if os.path.exists(csv_filename):
            with open(csv_filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] != new_entry[0]:
                        rows.append(row)

        # Add the new entry
        rows.append(new_entry)

        # Write the updated entries back to the CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

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
