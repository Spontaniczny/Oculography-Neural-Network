import sys
import os
import math
import cv2
import numpy as np

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QSlider, QSpinBox, QFormLayout, QMessageBox,
                             QRadioButton, QButtonGroup, QLineEdit, QProgressBar)
from PyQt5.QtCore import Qt, QDir, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIntValidator

class VideoFrameExtractor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Frame Extractor")

        # Variables for video
        self.video_path = None
        self.total_frames = 0
        self.fps = 0
        self.start_frame = 0
        self.end_frame = 0
        self.current_frame_index = 0
        self.cap = None

        # Variables for playback
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.playback_speed_ms = 30  # ~30 fps simulation

        # UI elements
        self.videoLabel = QLabel("No video loaded.")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("border: 1px solid black; background-color: #222;")
        # Set a fixed size for stable video display area
        self.videoLabel.setFixedSize(640, 360)

        self.loadButton = QPushButton("Load Video")
        self.loadButton.clicked.connect(self.loadVideo)

        self.playButton = QPushButton("Play")
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.playVideo)

        self.pauseButton = QPushButton("Pause")
        self.pauseButton.setEnabled(False)
        self.pauseButton.clicked.connect(self.pauseVideo)

        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.stopVideo)

        # Position slider to navigate through video frames
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setEnabled(False)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # Start and end sliders
        self.startSlider = QSlider(Qt.Horizontal)
        self.endSlider = QSlider(Qt.Horizontal)
        self.startSlider.setEnabled(False)
        self.endSlider.setEnabled(False)
        self.startSlider.valueChanged.connect(self.updateRangeAndExtractionInfo)
        self.endSlider.valueChanged.connect(self.updateRangeAndExtractionInfo)

        # Buttons to set start/end based on current frame
        self.setStartButton = QPushButton("Set Start")
        self.setStartButton.setEnabled(False)
        self.setStartButton.clicked.connect(self.setStartToCurrentFrame)
        self.setEndButton = QPushButton("Set End")
        self.setEndButton.setEnabled(False)
        self.setEndButton.clicked.connect(self.setEndToCurrentFrame)

        # Extraction parameters
        self.frameCountSpinBox = QSpinBox()
        self.frameCountSpinBox.setMinimum(1)
        self.frameCountSpinBox.setValue(10)
        self.frameCountSpinBox.valueChanged.connect(self.updateRangeAndExtractionInfo)

        self.percentageSpinBox = QSpinBox()
        self.percentageSpinBox.setMinimum(1)
        self.percentageSpinBox.setMaximum(100)
        self.percentageSpinBox.setValue(10)
        self.percentageSpinBox.valueChanged.connect(self.updateRangeAndExtractionInfo)

        self.frameWidthEdit = QLineEdit("300")
        self.frameWidthEdit.setValidator(QIntValidator(1, 10000))
        self.frameWidthEdit.textChanged.connect(self.updateRangeAndExtractionInfo)

        self.frameHeightEdit = QLineEdit("300")
        self.frameHeightEdit.setValidator(QIntValidator(1, 10000))
        self.frameHeightEdit.textChanged.connect(self.updateRangeAndExtractionInfo)

        # Radio buttons for extraction mode
        self.evenRadio = QRadioButton("Evenly spaced frames")
        self.normalRadio = QRadioButton("Normal distribution")
        self.evenRadio.setChecked(True)
        self.distributionGroup = QButtonGroup()
        self.distributionGroup.addButton(self.evenRadio)
        self.distributionGroup.addButton(self.normalRadio)
        self.evenRadio.toggled.connect(self.updateRangeAndExtractionInfo)
        self.normalRadio.toggled.connect(self.updateRangeAndExtractionInfo)

        # Radio buttons for selection method (count or percentage)
        self.byCountRadio = QRadioButton("By count")
        self.byCountRadio.setChecked(True)
        self.byPercentageRadio = QRadioButton("By percentage")
        self.selectionGroup = QButtonGroup()
        self.selectionGroup.addButton(self.byCountRadio)
        self.selectionGroup.addButton(self.byPercentageRadio)
        self.byCountRadio.toggled.connect(self.updateRangeAndExtractionInfo)
        self.byPercentageRadio.toggled.connect(self.updateRangeAndExtractionInfo)

        # Info labels
        self.infoLabel = QLabel("No video loaded.")
        self.rangeInfoLabel = QLabel("Start: 0, End: 0, Frames in Range: 0")
        self.extractionInfoLabel = QLabel("Frames to extract: 0")

        # Extract button
        self.extractButton = QPushButton("Extract Frames")
        self.extractButton.setEnabled(False)
        self.extractButton.clicked.connect(self.extractFrames)

        # Extraction progress
        self.extractionProgressBar = QProgressBar()
        self.extractionProgressBar.setValue(0)
        self.extractionProgressBar.setVisible(False)
        self.extractionProgressLabel = QLabel("")
        self.extractionProgressLabel.setVisible(False)

        # Layouts
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.loadButton)
        topLayout.addWidget(self.playButton)
        topLayout.addWidget(self.pauseButton)
        topLayout.addWidget(self.stopButton)

        videoControlLayout = QHBoxLayout()
        videoControlLayout.addWidget(self.positionSlider)

        startEndLayout = QHBoxLayout()
        startEndLayout.addWidget(self.setStartButton)
        startEndLayout.addWidget(self.setEndButton)

        rangeLayout = QFormLayout()
        rangeLayout.addRow("Start Frame:", self.startSlider)
        rangeLayout.addRow("End Frame:", self.endSlider)
        rangeLayout.addRow("", startEndLayout)

        selectionLayout = QHBoxLayout()
        selectionLayout.addWidget(self.byCountRadio)
        selectionLayout.addWidget(self.frameCountSpinBox)
        selectionLayout.addWidget(self.byPercentageRadio)
        selectionLayout.addWidget(self.percentageSpinBox)

        distributionLayout = QHBoxLayout()
        distributionLayout.addWidget(self.evenRadio)
        distributionLayout.addWidget(self.normalRadio)

        sizeLayout = QHBoxLayout()
        sizeLayout.addWidget(QLabel("Width:"))
        sizeLayout.addWidget(self.frameWidthEdit)
        sizeLayout.addWidget(QLabel("Height:"))
        sizeLayout.addWidget(self.frameHeightEdit)

        progressLayout = QHBoxLayout()
        progressLayout.addWidget(self.extractionProgressLabel)
        progressLayout.addWidget(self.extractionProgressBar)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoLabel)
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(videoControlLayout)
        mainLayout.addWidget(self.infoLabel)
        mainLayout.addWidget(self.rangeInfoLabel)
        mainLayout.addLayout(rangeLayout)
        mainLayout.addLayout(selectionLayout)
        mainLayout.addLayout(distributionLayout)
        mainLayout.addLayout(sizeLayout)
        mainLayout.addWidget(self.extractionInfoLabel)
        mainLayout.addWidget(self.extractButton)
        mainLayout.addLayout(progressLayout)

        self.setLayout(mainLayout)

    def loadVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Video", QDir.homePath(),
                                                  "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if fileName:
            self.video_path = fileName
            self.cap = cv2.VideoCapture(fileName)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Unable to open video.")
                self.video_path = None
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0  # fallback if fps is not available

            self.start_frame = 0
            self.end_frame = self.total_frames - 1
            self.startSlider.setMaximum(self.total_frames - 1)
            self.endSlider.setMaximum(self.total_frames - 1)
            self.startSlider.setValue(0)
            self.endSlider.setValue(self.total_frames - 1)
            self.startSlider.setEnabled(True)
            self.endSlider.setEnabled(True)
            self.current_frame_index = 0

            self.positionSlider.setMaximum(self.total_frames - 1)
            self.positionSlider.setEnabled(True)

            self.setStartButton.setEnabled(True)
            self.setEndButton.setEnabled(True)

            base_name = os.path.basename(fileName)
            self.infoLabel.setText(f"Loaded: {base_name}, Total Frames: {self.total_frames}, FPS: {self.fps:.2f}")

            # Enable playback controls
            self.playButton.setEnabled(True)
            self.pauseButton.setEnabled(True)
            self.stopButton.setEnabled(True)
            self.extractButton.setEnabled(True)

            self.updateFrame(show_first_frame=True)
            self.updateRangeAndExtractionInfo()

    def updateFrame(self, show_first_frame=False):
        if not self.cap or not self.cap.isOpened():
            return

        if show_first_frame:
            self.current_frame_index = 0

        if self.current_frame_index < 0:
            self.current_frame_index = 0
        if self.current_frame_index >= self.total_frames:
            self.current_frame_index = self.total_frames - 1
            self.pauseVideo()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.cap.read()
        if ret:
            # Convert to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.videoLabel.setPixmap(pix.scaled(self.videoLabel.width(),
                                                 self.videoLabel.height(),
                                                 Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation))
        else:
            # If no frame was retrieved, pause
            self.pauseVideo()

        # Update position slider
        self.positionSlider.blockSignals(True)
        self.positionSlider.setValue(self.current_frame_index)
        self.positionSlider.blockSignals(False)

        if self.playing:
            self.current_frame_index += 1

    def playVideo(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.playing = True
        self.timer.start(self.playback_speed_ms)

    def pauseVideo(self):
        self.playing = False
        self.timer.stop()

    def stopVideo(self):
        self.playing = False
        self.timer.stop()
        self.current_frame_index = 0
        self.updateFrame()

    def setPosition(self, position):
        self.current_frame_index = position
        self.updateFrame()

    def setStartToCurrentFrame(self):
        self.startSlider.setValue(self.current_frame_index)

    def setEndToCurrentFrame(self):
        self.endSlider.setValue(self.current_frame_index)

    def updateRangeAndExtractionInfo(self):
        if self.video_path is None:
            return

        self.start_frame = self.startSlider.value()
        self.end_frame = self.endSlider.value()
        if self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
            self.endSlider.setValue(self.end_frame)

        frames_in_range = self.end_frame - self.start_frame + 1
        self.rangeInfoLabel.setText(f"Start: {self.start_frame}, End: {self.end_frame}, Frames in Range: {frames_in_range}")

        # Determine how many frames to extract
        if frames_in_range > 0:
            if self.byCountRadio.isChecked():
                num_extract = self.frameCountSpinBox.value()
            else:
                percent = self.percentageSpinBox.value()
                num_extract = max(1, int(math.floor((percent / 100.0) * frames_in_range)))
        else:
            num_extract = 0

        self.extractionInfoLabel.setText(f"Frames to extract: {num_extract}")

    def extractFrames(self):
        if not self.video_path or not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "No video loaded.")
            return

        start_f = self.start_frame
        end_f = self.end_frame
        frames_in_range = end_f - start_f + 1
        if frames_in_range <= 0:
            QMessageBox.warning(self, "Warning", "Invalid start/end range.")
            return

        if self.byCountRadio.isChecked():
            num_extract = self.frameCountSpinBox.value()
        else:
            percent = self.percentageSpinBox.value()
            num_extract = max(1, int(math.floor((percent / 100.0) * frames_in_range)))

        even_dist = self.evenRadio.isChecked()

        # Compute which frames to extract
        if even_dist:
            indices = np.linspace(start_f, end_f, num_extract, dtype=int)
        else:
            mid = (start_f + end_f) / 2.0
            std_dev = (end_f - start_f) / 6.0
            candidate = np.random.normal(loc=mid, scale=std_dev, size=num_extract)
            candidate = np.clip(candidate, start_f, end_f)
            indices = np.rint(candidate).astype(int)

        try:
            w = int(self.frameWidthEdit.text())
            h = int(self.frameHeightEdit.text())
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid frame width/height.")
            return

        ret = QMessageBox.question(self, "Confirm Extraction",
                                   f"Extract {len(indices)} frames out of {frames_in_range} frames in range?",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.No:
            return

        # Create output directory
        out_dir = os.path.join(os.path.dirname(self.video_path), "extracted_frames")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Show progress bar
        self.extractionProgressBar.setVisible(True)
        self.extractionProgressBar.setRange(0, len(indices))
        self.extractionProgressBar.setValue(0)
        self.extractionProgressLabel.setVisible(True)

        unique_indices = np.unique(indices)
        extracted_count = 0

        for idx in unique_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {idx}")
                continue
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(out_dir, f"{base_name}_{idx}.png")
            cv2.imwrite(out_path, frame)
            extracted_count += 1

            # Update progress
            self.extractionProgressBar.setValue(extracted_count)
            self.extractionProgressLabel.setText(f"Extracted {extracted_count}/{len(indices)} frames")
            QApplication.processEvents()

        QMessageBox.information(self, "Done", f"Extracted {extracted_count} frames to {out_dir}")
        self.extractionProgressBar.setVisible(False)
        self.extractionProgressLabel.setVisible(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoFrameExtractor()
    window.resize(800, 700)
    window.show()
    sys.exit(app.exec_())
