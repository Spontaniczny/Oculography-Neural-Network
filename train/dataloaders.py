import pandas as pd
import os
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


def load_and_prepare_annotations(data_dir_path, pupil_type: str = "2P-fc2"):
    data = os.path.join(data_dir_path, 'annotation', 'annotations.csv')
    data = pd.read_csv(data)
    data = data[data.filename.str.startswith(pupil_type)]
    data['annotation'] = data_dir_path + '/pupil_map/' + data.filename.str.replace(r'jpg', 'png')
    data['frame'] = data_dir_path + '/fullFrames/' + data.filename
    return data


class ImageDataset(Dataset):
    def __init__(self, data_dir_path: str):
        super().__init__()
        self._data = load_and_prepare_annotations(data_dir_path)
        
        self.transform_in = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((334, 334), InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        self.transform_out = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((244, 244), InterpolationMode.NEAREST),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_data = self._data.iloc[idx]
        input_img = self.transform_in(decode_image(img_data.frame))
        output_img = self.transform_out(decode_image(img_data.annotation))
        output_img = output_img[0].unsqueeze(0)
        return input_img, output_img
    

def prepare_dataloaders(
        data_dir: str,
        pupil_type: str,
        split_ratio: list[float],
        batch_size: int = 16, 
        shuffle: bool = False, 
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

    dataset = ImageDataset(data_dir)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, split_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, test_loader
    
