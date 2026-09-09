import modal
import pandas as pd
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from model import AudioCNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#install modal to run serverless gpus

app = modal.App("audio-cnn")

image = (modal.Image.debian_slim() #creating modal image (the blueprint)
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget","unzip", "ffmpeg", "libsndfile1"]) # downloads wget to downlaod files from internet, unzip, ffmpeg to record, convert, and stream audio, and libsndfile1 a c library that allows programs to read and write audio files.
        .run_commands([
            "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip", #installs dataset to /tmp directory -O flag saves the name as esc50.zip
            "cd /tmp && unzip esc50.zip", # unzips the file 
            "mkdir -p /opt/esc50-data", # creates new directory
            "cp -r /tmp/ESC-50-master/* /opt/esc50-data", # copies the esc data from the one directory to the new directory
            "rm -rf /tmp/ESC-50-master /tmp/esc50.zip" # removes the data from the temperorary directory
        ])
        .add_local_python_source("model")) # imports our model.py into container

volume = modal.Volume.from_name("esc50-data", create_if_missing=True) #attaches to /opt/esc50-data in the container. As we close container the dataset doesnt get deleted
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)   # creates a new volume for the model to reference when we run inference

class ESC50Dataset(Dataset):  #inherits from dataset class from pytorch. 
    #meta data is divided in csv into 5 different fold per 400 files
    def __init__(self, data_dir, metadata_file, split= "train", transform=None): # one instance for training & one instance for validation, transform param required to account for validation and training dataset
        super().__init__()
        self.data_dir = Path(data_dir) #imported form pathlib
        self.metadata = pd.read_csv(metadata_file) #imported from pandas library to read csv file and returns data frame
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5] # this filters training and validation data through a boolean series
        else: 
            self.metadata = self.metadata[self.metadata['fold'] == 5] # 20% used for validation

        self.classes = sorted(self.metadata['category'].unique()) #returns a sorted list of  every unique value in categories alphabetically
        self.class_to_idx = {cls: idx   for idx, cls in enumerate(self.classes)} #loops through enumerate classes and then creates a dictionary that flips class as key and idx as value
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)  #adds the new indexes to the metadata in a new column
        #.map from pandas loops through each row and reads the category and matches it to the index in the dictionary class to idx and adds it to label column
    def __len__(self): #retursn number of rows in dataframe
        return len(self.metadata)

                    #idx is the id of the element we are intrested in from the data
    def __getitem__(self, idx): #needed for indexing into the dataset
        row = self.metadata.iloc[idx] #grabs the row from the panda dataframe as a pandas series which is like a dictionary
        audio_path = self.data_dir / "audio" / row['filename'] # points to audio folder then the row and looks for the value that is related to filename.
        waveform, sample_rate = torchaudio.load(audio_path) # torchaudio.load returns a tuple containg a tensor for the waveform and n int for the sample_rate
        #print(waveform.shape) = [channels, timesteps] / [rows,columns]
        if waveform.shape[0] > 1:  
            waveform = torch.mean(waveform, dim=0, keepdim=True) #finds the averages of values in tensor along the columns and reduces into a single row to account for if the wav file has multiple channels
                                    #keepdim =true keeps the shape of the rows and doesn't just leaves columns it acknowledges that there is still one row
        if self.transform:
            spectrogram = self.transform(waveform)
        else: 
            spectrogram = waveform
        return spectrogram, row['label'] #returns the spectrogram and looks in pandas series to look for label which is an index and returns that too
    #data mixing in order to force model to not be overconfident and to extract sounds closely from mixed data
def mixup_data(x, y): #x is features & y is labels
    lam = np.random.beta(0.2,0.2) #favors blending weight close to 0 or 1

    batch_size = x.size(0) #we can see batch size in first dimension of tensor
    index = torch.randperm(batch_size).to(x.device) #move it to gpu with the rest of our data
        #shuffles batch of audio clips
    # (0.7 * audio1 + 0.3* audio2)
    mixed_x = lam * x + (1 - lam) * x[index, :] #this targets only batch dimension so data to that batch gets mixed entirely
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a,y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    #loss calculation for the synthetic data


 # defines image above, references which gpu to use,  attaches volumes created above, timeout is how long function can run
@app.function(image=image, gpu="A10", volumes={"/data": volume, "/models": model_volume},timeout=60*60*3) 
def train():
    from datetime import datetime #import datetime class
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/models/tensorboard_logs/run_{timestamp}" #save tensor board file as timestamp
    writer = SummaryWriter(log_dir) #tensorboard summary writer
    esc50_dir = Path("/opt/esc50-data") #where our downloaded dataset is

    train_transform = nn.Sequential( #torchaudio.MelSpectrogram transforms file to mel spectrogram configurations is a random one online
        T.MelSpectrogram(
            sample_rate=44100, 
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(), #changes amplitude decibel scale that is more fit for human hearing since T.MelSpectrogram only warps frequency axis to match human hearing
        T.FrequencyMasking(freq_mask_param=30), #to prevent overfitting | frequency randomly masks frequency bins in spectrogram. applies a rnadom block of zero across frequency channels in a spectrogram
        T.TimeMasking(time_mask_param=80) #does the same thing but for time range
    )

    val_transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=44100, 
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )
    #creating instances for the datasets
    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file= esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)

    val_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file= esc50_dir / "meta" / "esc50.csv", split="test", transform=val_transform)

    print(f"training samples: {len(train_dataset)}") # returns number of rows in the training dataset
    print(f"validation samples: {len(val_dataset)}") # same for validation data set

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle = True) #tests 32 samples at a time, shuffle randomizes order of samples in each epoch
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #creates device object from pytorch to tell code to run on Nvidia GPU
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device) # puts the model into the memory of  the nvidia gpu

    num_epochs = 100
                # a crossentropyloss function used to calculate loss. distributes margin of 0.1 to other predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #be more humble in its prediciton |distribute numbers to other classes | prevents overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01) #weight decay is a regularization technique that penalizes large weights in the model 
            #adam because it is an adaptive learning rate optimizer which scales learning rate  based on how its gradient changes
    scheduler = OneCycleLR( #learning rate schedule that ramps the learning rate up to a high peak then brings it very low
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader), #how many optimizers steps shouild happen per epoch in this case the same amount as the batch
        pct_start=0.1 #spends 10% of training increasing learning rate and 90% decreasing lr
    )

    best_accuracy = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() #set model to training mode so dropout works and so does batch normalizaiton
        epoch_loss = 0.0
                        #tqdm is a progress bar library
        progress_bar = tqdm(train_dataloader, desc= f"Epoch{epoch+1}/{num_epochs}") #wrap iterable in tqdm
        for data, target in progress_bar: #loops across the dataloader and moves the input and label on to the gpu
            data, target = data.to(device), target.to(device)
                #30% of cases apply the synthetic data
            if np.random.random() > 0.7: 
                data, target_a, target_b, lam = mixup_data(data,target)
                output = model(data) #feeds the data into model and calculates loss
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else: #this is how you call forward pass
                output = model(data)
                loss = criterion(output,target)

            optimizer.zero_grad() #reset gradients from last run
            loss.backward() #launches back propagation process
            optimizer.step() #calculates and updates model parameters
            scheduler.step() #advances scheduler changing lr

            epoch_loss += loss.item() #calculates total loss for epoch
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
       
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch) #LOOKS F for learning rate  hyperparameter

        #validation after each epoch
        model.eval() #validation mode

        correct =0
        total = 0
        val_loss= 0

        with torch.no_grad(): #doesnt touch weights or biases
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output,target)
                val_loss += loss.item()
                    #throw away variable , stores indicies of the maximumm value=  finds maximum value along dimension 1
                _, predicted = torch.max(output.data, 1)
                total += target.size(0) #adds batch to the total amount of samples tested
                correct += (predicted == target).sum().item()
                #gets index tensor and compares .sum counts how many matches 
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_dataloader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        writer.close()
     

        print(f"Epoch {epoch +1} Loss: {avg_epoch_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state_dict": model.state_dict(),
                        "accuracy": accuracy,
                         "epoch": epoch,
                         "classes": train_dataset.classes}
                         , "/models/best_model.pth") #saves training checkpoint to the modal volume and metadata in a dictionary
            print(f"New best model saved: {accuracy:.2f}%")

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")        
        

@app.local_entrypoint()
def main():
    train.remote()