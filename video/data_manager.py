import os
import cv2
import csv

class DataManager:
    def __init__(self):
        pass

    def save_frame(self, video_name, frame_idx, current_frame, binary_mask, ellipse_info, data_dir):
        frames_dir = os.path.join(data_dir, 'frames')
        annotations_dir = os.path.join(data_dir, 'annotations')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        frame_filename = os.path.join(frames_dir, f"{video_name}_frame_{frame_idx}.png")
        binary_filename = os.path.join(annotations_dir, f"{video_name}_frame_{frame_idx}.png")
        cv2.imwrite(frame_filename, current_frame)
        cv2.imwrite(binary_filename, binary_mask)
        self.save_to_csv(video_name, frame_idx, ellipse_info, data_dir)

    def save_to_csv(self, video_name, frame_idx, ellipse_info, data_dir):
        csv_filename = os.path.join(data_dir, 'ellipse_info.csv')
        rows = []
        new_entry = [
            f"{video_name}_frame_{frame_idx}",
            ellipse_info['center_x'],
            ellipse_info['center_y'],
            ellipse_info['size_x'],
            ellipse_info['size_y'],
            ellipse_info['angle']
        ]
        if not os.path.exists(csv_filename):
            rows.append(["frame_name", "center_x", "center_y", "size_x", "size_y", "angle"])
        if os.path.exists(csv_filename):
            with open(csv_filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] != new_entry[0]:
                        rows.append(row)
        rows.append(new_entry)
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
