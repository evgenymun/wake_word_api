from fastapi import FastAPI

from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import librosa

import ipywidgets as widgets
from IPython import display as disp
from IPython.display import display, Audio, clear_output
# from google.colab import output
import base64
from pydub import AudioSegment
import io
import tempfile
import numpy as np


import torch
import torchaudio
#sudo apt-get install libportaudio2
# import sounddevice as sd
from scipy.io.wavfile import write
from torch import nn
# from torchsummary import summary

app = FastAPI()

import wave
from dataclasses import dataclass, asdict

import pyaudio


@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Recorder:
    """Recorder uses the blocking I/O facility from pyaudio to record sound
    from mic.

    Attributes:
        - stream_params: StreamParams object with values for pyaudio Stream
            object
    """
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None
        self._wav_file = None

    def record(self, duration: int, save_path: str) -> None:
        """Record sound from mic for a given amount of seconds.

        :param duration: Number of seconds we want to record for
        :param save_path: Where to store recording
        """
        print("Start recording...")
        self._create_recording_resources(save_path)
        self._write_wav_file_reading_from_stream(duration)
        self._close_recording_resources()
        print("Stop recording")

    def _create_recording_resources(self, save_path: str) -> None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        self._create_wav_file(save_path)

    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)

    def _write_wav_file_reading_from_stream(self, duration: int) -> None:
        for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            self._wav_file.writeframes(audio_data)

    def _close_recording_resources(self) -> None:
        self._wav_file.close()
        self._stream.close()
        self._pyaudio.terminate()


WAKE_WORDS = ["hey", "fourth", "brain"]
class CNN(nn.Module):
    def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
        super(CNN, self).__init__()
        conv0 = nn.Conv2d(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
        pool = nn.MaxPool2d(2)
        conv1 = nn.Conv2d(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
        self.num_hidden_input = num_hidden_input
        self.encoder1 = nn.Sequential(conv0,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(num_maps1, affine=True))
        self.encoder2 = nn.Sequential(conv1,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(num_maps2, affine=True))
        self.output = nn.Sequential(nn.Linear(num_hidden_input, hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden_size, num_labels))

    def forward(self, input_data):
        x1 = self.encoder1(input_data)
        x2 = self.encoder2(x1)
        x = x2.view(-1, self.num_hidden_input)
        return self.output(x)

# Little helper function to play the audio
def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

# Before we can test the data on the wake word phrase 
# let's create a function to trim or padd the data
def prepare_Stream(signal, num_samples):

    length_signal = signal[0].shape[0]

    if length_signal > num_samples:
        signal = signal[:, :num_samples]

    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)

        signal = torch.nn.functional.pad(signal, last_dim_padding)

    return signal

# The function will load the file, split inti words and then make a prediction
# If all three words are detected it will produce a note that wake word is detected
def predict_wake_word(recording_path):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    WAKE_WORDS = ["hey", "fourth", "brain"]
    num_labels = len(WAKE_WORDS) + 1 # oov
    num_maps1  = 48
    num_maps2  = 128
    num_hidden_input =  1024
    hidden_size = 128
    SAMPLE_RATE = 16000

    cnn2 = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
    state_dict = torch.load("wakeworddetaction_cnn7.pth", map_location=torch.device('cpu'))
    cnn2.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        # n_fft=512,
        # hop_length=200,
        # n_mels=40
    )
    mel_spectrogram.to(device)

    cnn2.eval()
    
    classes = WAKE_WORDS[:]
    # negative
    classes.append("negative")

    audio_float_size = 32767

    CHUNK = 500
    CHANNELS = 1
    RATE = SAMPLE_RATE
    RECORD_MILLI_SECONDS = 1000


    testFile = recording_path

    waveform, sample_rate = torchaudio.load(testFile)
    print(f"Recording SR: {sample_rate}")
    sounddata = librosa.core.load(testFile, sr=RATE, mono=True)[0]
    print(f"Original audio")
    play_audio(waveform, sample_rate)


    sound = AudioSegment.from_wav(testFile)
    chunks = split_on_silence(sound, 
        # must be silent for at least half a second
        min_silence_len=100,

        # consider it silent if quieter than -16 dBFS
        silence_thresh=-40
    )

    paths = []
    for i, chunk in enumerate(chunks):
        chunk.export("./vab/chunk{0}.wav".format(i), format="wav")
        paths.append("./vab/chunk{0}.wav".format(i))

    inference_track = []
    target_state = 0

    for path in paths: 
    
        waveform = ''
        waveform, sample_rate = torchaudio.load(path)
        print(f"path: {path}")
        play_audio(waveform, sample_rate)

        signal = prepare_Stream(waveform, 16000)
        with torch.no_grad():
            mel_audio_data = mel_spectrogram(signal.to(device)).float()
            predictions = cnn2(mel_audio_data.unsqueeze_(0).to('cpu'))
            print(f"predictions: {predictions}")
            predicted_index = predictions[0].argmax(0)
            print(f"predicted_index: {predicted_index}")
            predicted = classes[predicted_index]
            print(f"predicted: {predicted}")
            print(f"target_state1: {target_state}")
            label = WAKE_WORDS[target_state]
            print(f"label: {label}")

        if predicted == label:
            target_state = target_state + 1 # go to next label
            inference_track.append(predicted)
            
            print(f"target_state2: {target_state}")
            print(f"inference track: {inference_track}")
            if inference_track == WAKE_WORDS:
                print(f"Wake word {' '.join(inference_track)} detected")
                # target_state = 0
                # inference_track = []
                return(f"Wake word {' '.join(inference_track)} detected")
        elif target_state == 2: 
            target_state = 0
    return(f"Wake word is not detected: {' '.join(inference_track)}")

@app.get("/health")
def health():
    return "Service is running."

@app.get("/wav")
def wav():
    # audio_file = open(some_file_path, mode="rb")
    # return StreamingResponse(audio_file, media_type="audio/wav")

    fs = 16000  # Sample rate
    seconds = 4  # Duration of recording

    # myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    stream_params = StreamParams()
    recorder = Recorder(stream_params)
    recorder.record(3, "/vab/output.wav")

    # sd.wait()  # Wait until recording is finished
    # write('./vab/output.wav', fs, myrecording)  # Save as WAV file 

    return (predict_wake_word('vab/output.wav'))
    
# server {
#     listen 80;
#     server_name 34.222.29.197;
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#     }
# }
