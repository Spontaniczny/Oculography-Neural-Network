import os
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QSlider, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt
from video_player import VideoPlayer
from ellipse_manager import EllipseManager
from data_manager import DataManager

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
        self.video_label = QLabel(self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.frame_input = QLineEdit(self)
        self.frame_label = QLabel(self)
        self.save_button = QPushButton("Save Frame", self)
        self.load_button = QPushButton("Load Video", self)
        self.delete_button = QPushButton("Delete Ellipse", self)
        self.max_frames_label = QLabel(self)
        self.prev_frame_button = QPushButton("<", self)
        self.next_frame_button = QPushButton(">", self)
        # Initialize video player, ellipse manager, and data manager
        self.video_player = VideoPlayer()
        self.ellipse_manager = EllipseManager()
        self.data_manager = DataManager()
        # Connect signals and slots
        self.slider.valueChanged.connect(self.change_frame)
        self.frame_input.returnPressed.connect(self.go_to_frame)
        self.save_button.clicked.connect(self.save_frame)
        self.load_button.clicked.connect(self.load_video)
        self.delete_button.clicked.connect(self.delete_ellipse)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        # Other initializations
        self.current_frame = None
        self.current_frame_idx = 0

    def setup_gui_elements(self):
        pass  # Elements are initialized in initialize_variables()

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
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if video_path:
            self.video_player.load_video(video_path)
            self.max_frames_label.setText(f"/ {self.video_player.frame_count} frames")
            self.slider.setMaximum(self.video_player.frame_count - 1)
            self.change_frame(0)
            self.ellipse_manager.set_frame_size(self.current_frame.shape[1], self.current_frame.shape[0])

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.slider.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        if self.current_frame_idx < self.video_player.frame_count - 1:
            self.slider.setValue(self.current_frame_idx + 1)

    def change_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.current_frame = self.video_player.get_frame(frame_idx)
        if self.current_frame is not None:
            self.ellipse_manager.reset()
            self.update_video_display()
        self.frame_label.setText(f"Current Frame: {frame_idx}")

    def update_video_display(self):
        if self.current_frame is None:
            return
        pixmap = self.ellipse_manager.get_pixmap(self.current_frame)
        self.video_label.setPixmap(pixmap)

    def delete_ellipse(self):
        self.ellipse_manager.delete_ellipse()
        self.update_video_display()

    def save_frame(self):
        if self.current_frame is not None and self.ellipse_manager.has_ellipse():
            data_dir = os.path.join(os.path.dirname(self.video_player.video_path), 'data')
            self.data_manager.save_frame(
                self.video_player.video_name,
                self.current_frame_idx,
                self.current_frame,
                self.ellipse_manager.binary_mask,
                self.ellipse_manager.get_ellipse_info(),
                data_dir
            )
            print(f"Saved frame {self.current_frame_idx}")

    def go_to_frame(self):
        frame_num = int(self.frame_input.text())
        if 0 <= frame_num < self.video_player.frame_count:
            self.slider.setValue(frame_num)
            self.change_frame(frame_num)

    # Mouse events need to be forwarded to ellipse manager
    def mousePressEvent(self, event):
        self.ellipse_manager.mousePressEvent(event, self.video_label.pos())
        self.update_video_display()

    def mouseMoveEvent(self, event):
        self.ellipse_manager.mouseMoveEvent(event, self.video_label.pos())
        self.update_video_display()

    def mouseReleaseEvent(self, event):
        self.current_frame.shape
        self.ellipse_manager.mouseReleaseEvent(event)
        self.update_video_display()
