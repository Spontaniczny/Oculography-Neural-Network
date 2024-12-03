import wandb
from dotenv import load_dotenv
import os


env_var = load_dotenv()
wandb_key = os.environ.get("key")
wandb.login(key=wandb_key)