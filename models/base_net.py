import torch
import torch.nn as nn
import os
import json
import datetime
# from abc import ABC, abstractmethod

class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()

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

    def save_model(self, nn_config: dict):
        self.to("cpu")

        class_name = str.lower(self.__class__.__name__)
        saving_path = f"saved_models/{class_name}"

        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        experiment_id = nn_config['experiment_id']
        now = datetime.datetime.now()
        with open(f"{saving_path}/{now.strftime("%d:%m:%Y-%H:%M:%S")}.json", "w") as outfile: 
            json.dump(nn_config, outfile)

        torch.save(self.state_dict(), f"{saving_path}/{experiment_id}.pt")

