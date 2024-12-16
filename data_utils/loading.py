import os
import re
import pandas as pd
from typing import Type, Optional
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
from .datasets import SegmentationDataset, RegressionDataset, SegmentationAugmented, RegressionAugmented


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
    metadata_path = os.path.join(data_dir_path, 'metadata', 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Could not find metadata file")
    
    data = pd.read_csv(metadata_path)
    frames = find_frames(data_dir_path)
    annotations = find_annotations(data_dir_path)

    data['annotation'] = data_dir_path + f'/{annotations}/' + data.filename.str.replace(r'jpg', 'png')
    data['frame'] = data_dir_path + f'/{frames}/' + data.filename
    return data


def prepare_dataset(
        dataset_path: str,
        dataset_type: Type[SegmentationDataset | RegressionDataset],
        input_size: Optional[int] = None,
    ) -> SegmentationDataset:

    metadata = load_and_prepare_metadata(dataset_path)
    metadata = metadata.loc[:, ~metadata.columns.str.contains('^Unnamed')]

    dataset = dataset_type(
        metadata=metadata,
        net_input_size=input_size
    )

    return dataset


def load_multiple_datasets(
        dataset_type: str,
        datasets_prefix: str,
        input_size: int,
        suffix_regex: Optional[str] = None,
        suffix_list: Optional[list[str]] = None
) -> ConcatDataset:
    
    if suffix_regex is None and suffix_list is None:
        raise ValueError("At lest one of ['suffix_regex', 'suffix_list'] needs to be specified")
    
    if dataset_type not in ["segmentation", "regression"]:
        raise ValueError("Parameter dataset_type should be one of ['segmentation', 'regression']")
    
    if suffix_regex is not None:
        filenames = os.listdir(datasets_prefix)
        suffix_list = sorted((filter(lambda filename: re.search(suffix_regex, filename), filenames)))
    
    dataset_class = SegmentationDataset if dataset_type == "segmentation" else RegressionDataset
    datasets = []
    for suffix in suffix_list:
        dataset_path = f"{datasets_prefix}/{suffix}"
        datasets.append(
            prepare_dataset(dataset_path, dataset_class, input_size)
        )

    concatenated = ConcatDataset(datasets)
    return concatenated


def split_path(path: str) -> tuple[str, str]:
    path = os.path.normpath(path)

    path.split(os.sep)

    path_components = os.path.split(path)
    prefix = os.path.join(*path_components[:-1])
    suffix = path_components[-1]

    return prefix, suffix


def load_dataset(dataset_path: str, input_size: int, dataset_type: str = "segmentation"):
    prefix, suffix = split_path(dataset_path)
    return load_multiple_datasets(dataset_type, prefix, input_size, suffix_regex=suffix)


def prepare_dataloaders(
        dataset: Dataset,
        split_ratio: list[float],
        dataset_type: str,
        batch_size: int = 16, 
        augment: bool = False,
        shuffle: bool = True,
        num_workers: int = 6,
        
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_set, val_set, test_set = random_split(dataset, split_ratio)

    if augment:
        if dataset_type == "regression":
            train_set = RegressionAugmented(train_set)
        elif dataset_type == "segmentation":
            train_set = SegmentationAugmented(train_set)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        prefetch_factor=6 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        prefetch_factor=6 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )

    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
