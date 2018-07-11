import os
from os.path import exists, join, expanduser

import torch
import numpy as np
import librosa
import librosa.display

# need this for English text processing frontend
import nltk

import pickle

# import dv3.synthesis
# import train
# from deepvoice3_pytorch import frontend
# from train import build_model
# from train import restore_parts, load_checkpoint
from dv3.synthesis import tts as _tts


def tts(model, text, p=0, speaker_id=0, fast=True, figures=True):
  from dv3.synthesis import tts as _tts
  waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
  if figures:
      visualize(alignment, spectrogram)
  IPython.display.display(Audio(waveform, rate=fs))

def visualize(alignment, spectrogram):
  label_fontsize = 16
  figure(figsize=(16,16))

  subplot(2,1,1)
  imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
  xlabel("Decoder timestamp", fontsize=label_fontsize)
  ylabel("Encoder timestamp", fontsize=label_fontsize)
  colorbar()

  subplot(2,1,2)
  librosa.display.specshow(spectrogram.T, sr=fs,
                           hop_length=hop_length, x_axis="time", y_axis="linear")
  xlabel("Time", fontsize=label_fontsize)
  ylabel("Hz", fontsize=label_fontsize)
  tight_layout()
  colorbar()


def generate_cloned_samples(model,cloning_text_path  = None, no_speakers = 108 , fast = True, p =0 ):

    cloning_texts = ["this is the first" , "this is the second"]
    if(cloning_text_path == None):
        cloning_text_path = "./Cloning_Audio/cloning_text.txt"

    # cloning_texts = open("./Cloning_Audio/cloning_text.txt").read().splitlines()
    # no_cloning_texts = len(cloning_texts)

    all_speakers = []

    for speaker_id in range(no_speakers):
        speaker_cloning_mel = []
        # print("The Speaker being cloned speaker-{}".format(speaker_id))
        for text in cloning_texts:
            waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
            speaker_cloning_mel.append(mel)
            #print(np.array(speaker_cloning_mel).shape)
        all_speakers.append(speaker_cloning_mel)
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p", "wb") as fp:   #Pickling
            pickle.dump(all_speakers, fp)
        # print("")

    print("Shape of all speakers:",np.array(all_speakers).shape)
    # print(all_speakers.shape)


    # all speakers[speaker_id][cloned_audio_number]
    # print(all_speakers[0][1].shape)
    return all_speakers
