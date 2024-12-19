from PyQt5.QtCore import Qt, QPointF, QPoint
from gui import MediaEditorGUI
from video_player import VideoPlayer
from image_player import ImagePlayer
from ellipse_manager import EllipseManager
from data_manager import DataManager
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QMouseEvent
import os

class MediaEditor(MediaEditorGUI):
    def __init__(self):
        super().__init__()
        self.window_width = 600
        self.window_height = 600
        self.video_label.setFixedSize(self.window_width, self.window_height)
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

        # Connect alpha slider to update ellipse alpha
        self.alpha_slider.valueChanged.connect(self.update_ellipse_alpha)

    def update_ellipse_alpha(self, value):
        self.ellipse_manager.set_alpha(value)
        self.update_video_display()

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

        # Set frame size to ellipse manager
        if self.current_frame is not None:
            height, width, _ = self.current_frame.shape
            self.ellipse_manager.set_frame_size(width, height)

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.slider.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        if self.current_frame_idx < self.media_player.frame_count - 1:
            self.slider.setValue(self.current_frame_idx + 1)

    def change_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.current_frame = self.media_player.get_frame(frame_idx)
        if self.current_frame is not None:
            self.ellipse_manager.reset()
            # Update frame size in ellipse manager
            height, width, _ = self.current_frame.shape
            self.ellipse_manager.set_frame_size(width, height)
            self.update_video_display()
        self.frame_label.setText(f"Current Frame: {frame_idx}")

    def update_video_display(self):
        if self.current_frame is None:
            return

        height, width, _ = self.current_frame.shape
        frame_pixmap = self.ellipse_manager.get_frame_pixmap(self.current_frame)
        scaled_pixmap = frame_pixmap.scaled(self.window_width, self.window_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.displayed_width = scaled_pixmap.width()
        self.displayed_height = scaled_pixmap.height()

        self.scale_x = width / self.displayed_width
        self.scale_y = height / self.displayed_height

        # If centered alignment:
        self.offset_x = (self.window_width - self.displayed_width) / 2
        self.offset_y = (self.window_height - self.displayed_height) / 2

        # Set scale factors in ellipse manager
        self.ellipse_manager.set_scale_factors(self.scale_x, self.scale_y)

        final_pixmap = self.ellipse_manager.draw_overlay_on_pixmap(scaled_pixmap, self.scale_x, self.scale_y,
                                                                   self.drawing_mode)
        self.video_label.setPixmap(final_pixmap)


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
        label_pos = event.pos() - self.video_label.pos()
        img_x = label_pos.x() - self.offset_x
        img_y = label_pos.y() - self.offset_y

        if 0 <= img_x < self.displayed_width and 0 <= img_y < self.displayed_height:
            original_x = img_x * self.scale_x
            original_y = img_y * self.scale_y
            new_event = QMouseEvent(event.type(),
                                    QPointF(original_x, original_y),
                                    event.button(),
                                    event.buttons(),
                                    event.modifiers())
            self.ellipse_manager.mousePressEvent(new_event, QPoint(0, 0), self.drawing_mode)
        self.update_video_display()

    def mouseMoveEvent(self, event):
        label_pos = event.pos() - self.video_label.pos()
        img_x = label_pos.x() - self.offset_x
        img_y = label_pos.y() - self.offset_y
        if 0 <= img_x < self.displayed_width and 0 <= img_y < self.displayed_height:
            original_x = img_x * self.scale_x
            original_y = img_y * self.scale_y
            new_event = QMouseEvent(event.type(),
                                    QPointF(original_x, original_y),
                                    event.button(),
                                    event.buttons(),
                                    event.modifiers())
            self.ellipse_manager.mouseMoveEvent(new_event, QPoint(0, 0), self.drawing_mode)
        self.update_video_display()

    def mouseReleaseEvent(self, event):
        label_pos = event.pos() - self.video_label.pos()
        img_x = label_pos.x() - self.offset_x
        img_y = label_pos.y() - self.offset_y
        if 0 <= img_x < self.displayed_width and 0 <= img_y < self.displayed_height:
            original_x = img_x * self.scale_x
            original_y = img_y * self.scale_y
            new_event = QMouseEvent(event.type(),
                                    QPointF(original_x, original_y),
                                    event.button(),
                                    event.buttons(),
                                    event.modifiers())
            self.ellipse_manager.mouseReleaseEvent(new_event, self.drawing_mode)
        else:
            # If the user releases outside the displayed area, just finalize normally
            self.ellipse_manager.mouseReleaseEvent(event, self.drawing_mode)
        self.update_video_display()
