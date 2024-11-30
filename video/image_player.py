import os
import cv2
from natsort import natsorted

class ImagePlayer:
    def __init__(self):
        self.media_path = None
        self.media_name = None
        self.image_names = []
        self.frame_count = 0

    def load_images(self, image_dir):
        self.media_path = image_dir
        self.media_name = os.path.basename(image_dir)
        self.image_names = [
            img for img in os.listdir(image_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        self.image_names = natsorted(self.image_names)
        self.frame_count = len(self.image_names)

    def get_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        image_path = os.path.join(self.media_path, self.image_names[frame_idx])
        frame = cv2.imread(image_path)
        return frame
