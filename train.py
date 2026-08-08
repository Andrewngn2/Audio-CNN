import modal
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

app = modal.App("audio-cnn")

image = (modal.Image.debian_slim() #creating modal image
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget","unzip", "ffmpeg", "libsndfile1"])
        .run_commands([
            "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
            "cd /tmp && unzip esc50.zip",
            "mkdir -p /opt/esc50-data",
            "cp -r /tmp/ESC-50-master/* /opt/esc50-data",
            "rm -rf /tmp/ESC-50-master /tmp/esc50.zip"
        ])
        .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True) #attaches to /opt/esc50-data in the container
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True) 

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split= "train", transform=None): # one instance for training & one instance for validation
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5] 
        else: 
            self.metadata = self.metadata[self.metadata['fold'] == 5]

            
@app.function(image=image, gpu="A10", volumes={"/data": volume, "/models": model_volume},timeout=60*60*3)
def train():
    print("training")

@app.local_entrypoint()
def main():
    train.remote()