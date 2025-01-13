import torch
import torch.nn as nn
import os
import json
import datetime
from abc import ABC, abstractmethod
from typing import Optional

class BaseNet(nn.Module, ABC):

    def __init__(self):
        super().__init__()

        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")

        self.saving_folder = "tmp"
        self.saved_weights = None
    
    def cache_current_weights(self):
        saving_path = f"{self.saving_folder}/best_weights.pt"
        self.saved_weights = saving_path
        print(f"Caching current weights to {saving_path}")
        torch.save(self.state_dict(), saving_path)

    def load_best_weights(self):
        if self.saved_weights:
            print("Restoring optimal model's weights")
            self.load_state_dict(torch.load(self.saved_weights, weights_only=True))
        else:
            print("No cached weights found")

    def save_model(self, nn_config: dict, directory: str = ".", finetuning: bool = False) -> str:
        self.to("cpu")

        class_name = str.lower(self.__class__.__name__)

        main_dir = f"{directory}/saved_models"
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

        saving_path = f"{main_dir}/{class_name}"
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        experiment_id = nn_config['experiment_id']
        suffix = "_finetuning" if finetuning else "" 
        now = datetime.datetime.now()

        config_path = f"{saving_path}/{now.strftime("%d:%m:%Y-%H:%M:%S")}{suffix}.json"
        with open(config_path, "w") as outfile: 
            json.dump(nn_config, outfile, indent=4)

        torch.save(self.state_dict(), f"{saving_path}/{experiment_id}.pt")

        return config_path


    def freeze_backbone(self):
        """Freezes the backbone weights. Assumes 'backbone' exists in child classes."""
        if hasattr(self, 'backbone') and isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def get_trainable_params(self):
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params
    
    def count_params(self) -> dict[str, int]:
        all_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, 'backbone') and isinstance(self.backbone, nn.Module):
            backbone_count = sum(p.numel() for p in self.backbone.parameters())
            head_params = all_params - backbone_count
            return {
                "backbone_params_count": backbone_count,
                "all_params_count": all_params,
                "head_params_count": head_params
            }
        else:
            return {
                "all_params_count": all_params
            }

        
    @abstractmethod
    def predict_mask(self, batch: torch.Tensor, threshold: Optional[float] = 0.5) -> torch.Tensor:
        pass

    @abstractmethod
    def draw_ellipse(self, params_batch: torch.Tensor) -> torch.Tensor:
        pass
