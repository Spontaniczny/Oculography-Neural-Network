import cv2
import os

class VideoPlayer:
    def __init__(self):
        self.video_path = None
        self.video_name = None
        self.video_cap = None
        self.frame_count = 0

    def load_video(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, frame_idx):
        if not self.video_cap:
            return None
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if ret:
            return frame
        else:
            return None
