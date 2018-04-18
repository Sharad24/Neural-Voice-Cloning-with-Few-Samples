import os
from os.path import exists, join, expanduser

import torch
import numpy as np
import librosa
import librosa.display
# import IPython
# from IPython.display import Audio
# need this for English text processing frontend
import nltk
import numpy as np
import pickle


def tts(model, text, p=0, speaker_id=0, fast=True, figures=True):
  from synthesis import tts as _tts
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

def generate_cloned_samples(cloning_texts_location  = None, no_speakers = 108 , fast = True, p =0 ):

    # Clone
    name = "deepvoice3_pytorch"
    # if not exists(name):
    #   print("Clone the repo!!")
    # else:
    #     print("Exists!")

    # Change working directory to the project dir
    os.chdir(join(expanduser("."), name))

    import hparams
    import json
    import synthesis
    import train
    from deepvoice3_pytorch import frontend
    from train import build_model
    from train import restore_parts, load_checkpoint
    from synthesis import tts as _tts



    # get_ipython().system(u' python3 -m nltk.downloader cmudict')

    checkpoint_path = "20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"

    if not exists(checkpoint_path):
        print("Dowload the Pre-Trained Network!!")
    #   !curl -O -L "https://www.dropbox.com/s/uzmtzgcedyu531k/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"



    # Copy preset file (json) from master
    # The preset file describes hyper parameters
    # get_ipython().system(u' git checkout master --quiet')
    preset = "./presets/deepvoice3_vctk.json"
    # get_ipython().system(u' cp -v $preset .')
    # preset = "./deepvoice3_vctk.json"

    # And then git checkout to the working commit
    # This is due to the model was trained a few months ago and it's not compatible
    # with the current master.
    # ! git checkout 0421749 --quiet
    # ! pip install -q -e .




    # print(hparams.hparams.get_model_structure())
    # Newly added params. Need to inject dummy values
    for dummy, v in [("fmin", 0), ("fmax", 0), ("rescaling", False),
                     ("rescaling_max", 0.999),
                     ("allow_clipping_in_normalization", False)]:
      #if hparams.hparams.get(dummy) is None:
        hparams.hparams.add_hparam(dummy, v)

    # Load parameters from preset
    with open(preset) as f:
      hparams.hparams.parse_json(f.read())

    # Tell we are using multi-speaker DeepVoice3
    hparams.hparams.builder = "deepvoice3_multispeaker"

    # Inject frontend text processor

    synthesis._frontend = getattr(frontend, "en")
    train._frontend =  getattr(frontend, "en")

    # alises
    fs = hparams.hparams.sample_rate
    hop_length = hparams.hparams.hop_size



    model = build_model()
    model = load_checkpoint(checkpoint_path, model, None, True)




    # text = "here i am"
    # speaker_id = 0
    # fast = True
    # p = 0
    # waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
    # print(waveform.shape)
    # print(alignment.shape)
    # print(spectrogram.shape)
    # print(mel.shape)
    # print(type(mel))



    cloning_texts = ["this is the first" , "this is the first"]
    if(cloning_texts_location == None)
        cloning_texts_location = "./Cloning_Audio/cloning_text.txt"

    # cloning_texts = open("./Cloning_Audio/cloning_text.txt").splitlines()
    # no_cloning_texts = len(cloning_texts)

    all_speakers = []
    for speaker_id in range(no_speakers):
        speaker_cloning_mel = []
        print("The Speaker being cloned speaker-{}".format(speaker_id))
        for text in cloning_texts:
            waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
            speaker_cloning_mel.append(mel)
            #print(np.array(speaker_cloning_mel).shape)
        all_speakers.append(speaker_cloning_mel)
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p", "wb") as fp:   #Pickling
            pickle.dump(all_speakers, fp)
        print("")

    print(np.array(all_speakers).shape)
    # print(all_speakers.shape)


    # all speakers[speaker_id][cloned_audio_number]
    # print(all_speakers[0][1].shape)
    return all_speakers
