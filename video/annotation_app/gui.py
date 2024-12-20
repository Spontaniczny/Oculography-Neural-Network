from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QSlider, QVBoxLayout, QPushButton, QWidget, QLineEdit, QHBoxLayout, QRadioButton, QButtonGroup, QCheckBox
)
from PyQt5.QtCore import Qt

class MediaEditorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_variables()
        self.setup_layout()

    def initialize_variables(self):
        self.setWindowTitle("Media Frame Editor with Ellipses")
        self.video_label = QLabel(self)
        # self.video_label.setFixedSize(400, 400)  # Fixed 400x400 display area

        self.slider = QSlider(Qt.Horizontal, self)
        self.frame_input = QLineEdit(self)
        self.frame_label = QLabel(self)
        self.media_label = QLabel(self)
        self.save_button = QPushButton("Save Frame", self)
        self.load_video_button = QPushButton("Load Video", self)
        self.load_images_button = QPushButton("Load Images", self)
        self.delete_button = QPushButton("Delete Ellipse/Points", self)
        self.max_frames_label = QLabel(self)
        self.prev_frame_button = QPushButton("<", self)
        self.next_frame_button = QPushButton(">", self)
        self.prev_frame_button.setFixedWidth(30)
        self.next_frame_button.setFixedWidth(30)

        # Drawing mode selection
        self.draw_ellipse_radio = QRadioButton("Draw Ellipse")
        self.draw_points_radio = QRadioButton("Draw Points")
        self.draw_ellipse_radio.setChecked(True)  # Default mode
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.draw_ellipse_radio)
        self.mode_button_group.addButton(self.draw_points_radio)

        self.fit_ellipse_button = QPushButton("Fit Ellipse")
        self.fit_ellipse_button.setEnabled(False)  # Disabled by default

        # Alpha slider for ellipse interior
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(255)
        self.alpha_slider.setValue(80)  # Default alpha
        self.alpha_label = QLabel("Ellipse Alpha:")

        # New alpha slider for ellipse edge
        self.edge_alpha_slider = QSlider(Qt.Horizontal)
        self.edge_alpha_slider.setMinimum(0)
        self.edge_alpha_slider.setMaximum(255)
        self.edge_alpha_slider.setValue(255)  # Default full opacity
        self.edge_alpha_label = QLabel("Ellipse Edge Alpha:")

        # Gamma slider
        self.gamma_label = QLabel("Gamma:")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(50)
        self.gamma_slider.setMaximum(150)
        self.gamma_slider.setValue(100)  # 100 = gamma 1.0

        # Contrast slider
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(150)
        self.contrast_slider.setValue(100)  # 100 = contrast 1.0

        # Checkbox to go to next frame after saving
        self.next_frame_checkbox = QCheckBox("Go to next frame after saving")
        self.next_frame_checkbox.setChecked(False)


    def setup_layout(self):
        frame_control_layout = QHBoxLayout()
        frame_control_layout.addWidget(self.prev_frame_button)
        frame_control_layout.addWidget(self.frame_input)
        frame_control_layout.addWidget(self.max_frames_label)
        frame_control_layout.addWidget(self.frame_label)
        frame_control_layout.addWidget(self.media_label)
        frame_control_layout.addWidget(self.next_frame_button)

        load_buttons_layout = QHBoxLayout()
        load_buttons_layout.addWidget(self.load_video_button)
        load_buttons_layout.addWidget(self.load_images_button)

        drawing_mode_layout = QHBoxLayout()
        drawing_mode_layout.addWidget(self.draw_ellipse_radio)
        drawing_mode_layout.addWidget(self.draw_points_radio)
        drawing_mode_layout.addWidget(self.fit_ellipse_button)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_label)
        alpha_layout.addWidget(self.alpha_slider)

        edge_alpha_layout = QHBoxLayout()
        edge_alpha_layout.addWidget(self.edge_alpha_label)
        edge_alpha_layout.addWidget(self.edge_alpha_slider)

        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(self.gamma_label)
        gamma_layout.addWidget(self.gamma_slider)

        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(self.contrast_label)
        contrast_layout.addWidget(self.contrast_slider)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)
        layout.addLayout(frame_control_layout)
        layout.addLayout(load_buttons_layout)
        layout.addLayout(drawing_mode_layout)
        layout.addLayout(alpha_layout)
        layout.addLayout(edge_alpha_layout)
        layout.addLayout(gamma_layout)
        layout.addLayout(contrast_layout)
        layout.addWidget(self.next_frame_checkbox)
        layout.addWidget(self.save_button)
        layout.addWidget(self.delete_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
