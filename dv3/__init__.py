import torch
import numpy as np
import librosa
import librosa.display
# import IPython
# from IPython.display import Audio
# need this for English text processing frontend
import nltk

# from .dv3.deepvoice3_pytorch import synthesis as syn
# import synthesis
import dv3.train
# print(os.getcwd())
import dv3.hparams
import json

from dv3.train import build_model
from dv3.train import restore_parts, load_checkpoint
from dv3.synthesis import tts as _tts


from dv3.deepvoice3_pytorch import frontend

# print(os.getcwd())

checkpoint_path = "./20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"


# Copy preset file (json) from master
# The preset file describes hyper parameters
# ! git checkout master --quiet
preset = "./dv3/presets/deepvoice3_vctk.json"
# ! cp -v $preset .
# preset = "./deepvoice3_vctk.json"



def build_deepvoice_3():
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
    # Tell we need speaker embedding in 512 dim
    hparams.hparams.speaker_embed_dim = 512

    # Inject frontend text processor
    synthesis._frontend = getattr(frontend, "en")
    train._frontend =  getattr(frontend, "en")
    # alises
    fs = hparams.hparams.sample_rate
    hop_length = hparams.hparams.hop_size

    dv3_model = train.build_model()
    dv3_model = train.load_checkpoint(checkpoint_path, dv3_model, None, True)
    # Change the working directory back
    return dv3_model
# model = build_deepvoice_3()
