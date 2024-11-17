import pandas as pd
import os
import torch
from PIL import Image
from typing import Type, Optional
from models.segmentation import U_NET
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


def load_and_prepare_annotations(data_dir_path: str) -> pd.DataFrame:
    data = os.path.join(data_dir_path, 'metadata', 'metadata.csv')
    data = pd.read_csv(data)
    data['annotation'] = data_dir_path + '/annotations/' + data.filename.str.replace(r'jpg', 'png')
    data['frame'] = data_dir_path + '/frames/' + data.filename
    return data


# def load_and_prepare_annotations(data_dir_path):
#     data = os.path.join(data_dir_path, 'annotation', 'annotations.csv')
#     data = pd.read_csv(data)
#     data = data[data.filename.str.startswith("2P-fc2")]
#     data['annotation'] = data_dir_path + '/pupil_map/' + data.filename.str.replace(r'jpg', 'png')
#     data['frame'] = data_dir_path + '/fullFrames/' + data.filename
#     return data


class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        input_image_size: int,
        output_image_size: int
    ) -> None:
        super().__init__()

        self._data = metadata
        self.transform_in = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((input_image_size, input_image_size), InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]) 

        self.transform_out = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            # v2.Grayscale(),
            v2.Resize((output_image_size, output_image_size), InterpolationMode.NEAREST),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self._input_image_size = input_image_size
        self._output_image_size = output_image_size

    @property
    def input_image_size(self):
        return self._input_image_size
    
    @property
    def output_image_size(self):
        return self._output_image_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_data = self._data.iloc[idx]
        input_img = self.transform_in(Image.open(img_data.frame))
        output_img = self.transform_out(Image.open(img_data.annotation))
        return input_img, output_img[0].unsqueeze(0)


def _infer_model_input_size(metadata: pd.DataFrame) -> int:
    input_w, input_h = int(metadata['w'].mean()), int(metadata['h'].mean())
    return min(input_w, input_h)


def prepare_segmentation_dataset_and_net(
        dataset_path: str, 
        model_type: Type[U_NET],
        input_size: Optional[int] = None,
    ) -> tuple[SegmentationDataset, U_NET]:

    metadata = load_and_prepare_annotations(dataset_path)
    metadata = metadata.loc[:, ~metadata.columns.str.contains('^Unnamed')]

    model_input_size = input_size if input_size else _infer_model_input_size(metadata)
    net = model_type(model_input_size)
    model_output_size = net.final_size

    dataset = SegmentationDataset(
        metadata=metadata,
        input_image_size=model_input_size,
        output_image_size=model_output_size
    )

    return dataset, net

