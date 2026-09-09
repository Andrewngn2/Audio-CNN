import base64
import io
from unittest import result

import modal 
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from model import AudioCNN
import soundfile as sf
import librosa
from pydantic import BaseModel
app = modal.App("audio-cnn-inference")
#create and deploy new modal app
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install("libsndfile1")
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")

#transforms audio so it is suitable for input into model
class AudioProcessor:
    def __init__(self):
        self.transform =nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050, 
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        #creates tensor
        waveform = waveform.unsqueeze(0)
        #adds a channel dimension since 

        spectrogram = self.transform(waveform)
        #transforms waveform to spectrogram
        #adds batch dimension because this is what the model expects
        return spectrogram.unsqueeze(0)

class InferenceRequest(BaseModel):#defines schema for modal endpoint schema inherits from pydantic basemodel
    audio_data: str

@app.cls(image=image, gpu="A10" , volumes = {"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("loading models on enter")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load("/models/best_model.pth", map_location=self.device) #loads model checkpoint
        #model metadata was stored so we load those too
        self.classes= checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"]) #creates new model instance and loads weights and biases
        self.model.to(self.device) 
        self.model.eval()
        #creates audioProcessor instance
        self.audio_processor = AudioProcessor()

        print("model loaded on enter")
    @modal.fastapi_endpoint(method="POST") #creating api endpoint
    def inference(self,request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)
            #extracts a string of text from inference request object and turns it back into binary data
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype = "float32")
         #returns numpy array and sample rate  #extracts waveform #use io module to wrap bytes so it behaves like a file 
        if audio_data.ndim > 1: #if it isnt mono make it one channel
            audio_data = np.mean(audio_data, axis =1)

        if sample_rate != 44100: #make file have right sample rate
            audio_data = librosa.resample(y= audio_data, orig_sr= sample_rate, target_sr = 44100)
            #converts audio to spectrogram and loads into gpu memory
        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps = True)
                #converts not a number output to 0
            output = torch.nan_to_num(output) #outputs logits
            probabilities = torch.softmax(output, dim=1) #converts logits to probabilities, dim =1 is the  classes and dim =0 is batch-size
            top3_probs, top3_indicies = torch.topk(probabilities[0],3) # since only one item in batch we select probability distrubition for our only sample
                        #iterates over tuple to print top 3 probabilities
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()} for prob, idx in zip(top3_probs, top3_indicies)]

            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4: #[batch_size, channels, height, width]
                    aggregated_tensor =torch.mean(tensor,dim =1)
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                        }

            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            max_samples = 8000
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data


        response = {"predictions": predictions,
                    "visualization": viz_data,
                    "input_spectrogram":{"shape": list(clean_spectrogram.shape), 
                    "values": clean_spectrogram.tolist()},
                    "waveform": {"values": waveform_data.tolist(), "sample_rate": 22050, "duration": len(audio_data)/ 22050}
                    }

        return response

    @app.local_entrypoint()
    def main():
        audio_data, sample_rate =sf.read("dogbark.wav")

        buffer = io.BytesIO() #creates virtual file in memory
        sf.write(buffer, audio_data, sample_rate, format = "WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")# gets the binary bytes from memory and turns it into b64 binary data and then converts it into python text string
        payload = {"audio_data": audio_b64}

        server = AudioClassifier()
        url = server.inference.get_web_url()
        response = requests.post(url, json=payload)
        response.raise_for_status() #gets my request and returns as json , throws and error if we get a 400 back
        result = response.json()

        waveform_info = result.get("waveform", {})

        if waveform_info: 
            values = waveform_info.get("values", {})
            print(f"First 10 values: {[round(v,4) for v in values[:10]]}...")
            print(f"Duration: {waveform_info.get('duration',0)}")

        print("Top predictions:")
        for pred in result.get("predictions", []):
            print(f" -{pred["class"]} {pred["confidence"]:0.2% } ")