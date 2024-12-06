from PyQt5.QtCore import Qt
from gui import MediaEditorGUI
from video_player import VideoPlayer
from image_player import ImagePlayer
from ellipse_manager import EllipseManager
from data_manager import DataManager
from PyQt5.QtWidgets import QFileDialog
import os

class MediaEditor(MediaEditorGUI):
    def __init__(self):
        super().__init__()
        self.initialize_logic()
        self.connect_signals()

    def initialize_logic(self):
        # Initialize media players and managers
        self.media_player = None  # Will be either VideoPlayer or ImagePlayer
        self.ellipse_manager = EllipseManager()
        self.data_manager = DataManager()
        self.current_frame = None
        self.current_frame_idx = 0
        self.setMouseTracking(True)
        self.video_label.setMouseTracking(True)

        # Drawing mode: 'ellipse' or 'points'
        self.drawing_mode = 'ellipse'

    def connect_signals(self):
        self.slider.valueChanged.connect(self.change_frame)
        self.frame_input.returnPressed.connect(self.go_to_frame)
        self.save_button.clicked.connect(self.save_frame)
        self.load_video_button.clicked.connect(self.load_video)
        self.load_images_button.clicked.connect(self.load_images)
        self.delete_button.clicked.connect(self.delete_ellipse_or_points)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.draw_ellipse_radio.toggled.connect(self.change_drawing_mode)
        self.fit_ellipse_button.clicked.connect(self.fit_ellipse)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", filter="Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_path:
            self.media_player = VideoPlayer()
            self.media_player.load_video(video_path)
            self.setup_media()

    def load_images(self):
        image_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if image_dir:
            self.media_player = ImagePlayer()
            self.media_player.load_images(image_dir)
            self.setup_media()

    def setup_media(self):
        self.max_frames_label.setText(f"/ {self.media_player.frame_count} frames")
        self.slider.setMaximum(self.media_player.frame_count - 1)
        self.change_frame(0)

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.slider.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        if self.current_frame_idx < self.media_player.frame_count - 1:
            self.slider.setValue(self.current_frame_idx + 1)

    def change_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.current_frame = self.media_player.get_frame(frame_idx)
        self.ellipse_manager.set_frame_size(self.current_frame.shape[1], self.current_frame.shape[0])
        if self.current_frame is not None:
            self.ellipse_manager.reset()
            self.update_video_display()
        self.frame_label.setText(f"Current Frame: {frame_idx}")

    def update_video_display(self):
        if self.current_frame is None:
            return
        pixmap = self.ellipse_manager.get_pixmap(self.current_frame, self.drawing_mode)
        self.video_label.setPixmap(pixmap)

    def delete_ellipse_or_points(self):
        self.ellipse_manager.delete_ellipse_or_points()
        self.update_video_display()

    def save_frame(self):
        if self.current_frame is not None and self.ellipse_manager.has_ellipse():
            data_dir = os.path.join(os.path.dirname(self.media_player.media_path), 'data')
            self.data_manager.save_frame(
                self.media_player.media_name,
                self.current_frame_idx,
                self.current_frame,
                self.ellipse_manager.binary_mask,
                self.ellipse_manager.get_ellipse_info(),
                data_dir
            )
            print(f"Saved frame {self.current_frame_idx}")

    def go_to_frame(self):
        frame_num = int(self.frame_input.text())
        if 0 <= frame_num < self.media_player.frame_count:
            self.slider.setValue(frame_num)
            self.change_frame(frame_num)

    def change_drawing_mode(self):
        if self.draw_ellipse_radio.isChecked():
            self.drawing_mode = 'ellipse'
            self.fit_ellipse_button.setEnabled(False)
        else:
            self.drawing_mode = 'points'
            self.fit_ellipse_button.setEnabled(True)
            self.ellipse_manager.reset()
        self.update_video_display()

    def fit_ellipse(self):
        self.ellipse_manager.fit_ellipse_to_points()
        self.update_video_display()

    # Mouse events need to be forwarded to ellipse manager
    def mousePressEvent(self, event):
        self.ellipse_manager.mousePressEvent(event, self.video_label.pos(), self.drawing_mode)
        self.update_video_display()

    def mouseMoveEvent(self, event):
        self.ellipse_manager.mouseMoveEvent(event, self.video_label.pos(), self.drawing_mode)
        self.update_video_display()

    def mouseReleaseEvent(self, event):
        self.ellipse_manager.mouseReleaseEvent(event, self.drawing_mode)
        self.update_video_display()
