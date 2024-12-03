import pandas as pd
import os
import re
import torch
from PIL import Image
from typing import Type, Optional
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


def find_annotations(data_dir_path: str):
    annotations = os.path.join(data_dir_path, "annotations")
    if os.path.exists(annotations):
        return "annotations"
    
    annotations = os.path.join(data_dir_path, "labels")
    if os.path.exists(annotations):
        return "labels"
    
    raise FileNotFoundError("Could not find annotations folder")

def find_frames(data_dir_path: str):
    frames = os.path.join(data_dir_path, "frames")
    if os.path.exists(frames):
        return "frames"
    
    raise FileNotFoundError("Could not find frames folder")
    

def load_and_prepare_metadata(data_dir_path: str) -> pd.DataFrame:
    data = os.path.join(data_dir_path, 'metadata', 'metadata.csv')
    data = pd.read_csv(data)

    frames = find_frames(data_dir_path)
    annotations = find_annotations(data_dir_path)

    data['annotation'] = data_dir_path + f'/{annotations}/' + data.filename.str.replace(r'jpg', 'png')
    data['frame'] = data_dir_path + f'/{frames}/' + data.filename
    return data



class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        net_input_size: Optional[int] = None
    ) -> None:
        super().__init__()

        self._data = metadata
        self._original_image_width = int(metadata['w'].mean())
        self._original_image_height = int(metadata['h'].mean())

        if net_input_size is None:
            net_input_size = min(self._original_image_width, self._original_image_height)

        if net_input_size % 16:
            net_input_size -= net_input_size % 16

        self.net_input_size = net_input_size

        self.transform_in = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((net_input_size, net_input_size), InterpolationMode.BILINEAR),
        ]) 

        self.transform_out = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((net_input_size, net_input_size), InterpolationMode.NEAREST),
        ])

    def get_example_with_metadata(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, pd.Series]:
        img_data = self._data.iloc[idx]
        return *self.__getitem__(idx), img_data
    

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_data = self._data.iloc[idx]
        input_img = self.transform_in(Image.open(img_data.frame))
        output_img = self.transform_out(Image.open(img_data.annotation))
        return input_img, output_img[0].unsqueeze(0)


def prepare_segmentation_dataset(
        dataset_path: str, 
        input_size: Optional[int] = None,
    ) -> SegmentationDataset:

    metadata = load_and_prepare_metadata(dataset_path)
    metadata = metadata.loc[:, ~metadata.columns.str.contains('^Unnamed')]

    dataset = SegmentationDataset(
        metadata=metadata,
        net_input_size=input_size
    )

    return dataset


def load_multiple_datasets(
        datasets_prefix: str,
        input_size: int,
        suffix_regex: Optional[str] = None,
        suffix_list: Optional[list[str]] = None
) -> ConcatDataset:
    
    if suffix_regex is None and suffix_list is None:
        raise ValueError("At lest one of ['suffix_regex', 'suffix_list'] needs to be specified")
    
    if suffix_regex is not None:
        filenames = os.listdir(datasets_prefix)
        suffix_list = sorted((filter(lambda filename: re.search(suffix_regex, filename), filenames)))
    
    datasets = []
    for suffix in suffix_list:
        dataset_path = f"{datasets_prefix}/{suffix}"
        datasets.append(
            prepare_segmentation_dataset(dataset_path, input_size)
        )

    concatenated = ConcatDataset(datasets)
    return concatenated


