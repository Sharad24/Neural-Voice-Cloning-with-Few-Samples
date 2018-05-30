import pickle

import torch
from torch.autograd import Variable
# from deepvoice3_pytorch.hparams import hparams
from utils import generate_cloned_samples
import dv3
# from deepvoice3_pytorch import synthesis
import sys, os
sys.path.append('./deepvoice3_pytorch')
from dv3 import build_deepvoice_3

# print(hparams)

#
# def build_deepvoice_3():
#     # # Clone
#     import os
#     from os.path import exists,join,expanduser
#     name = "deepvoice3_pytorch"
#     if not exists(name):
#       print("Clone the repo!!")
#     else:
#         print("Exists!")
#
#     # Change working directory to the project dir
#     os.chdir(join(expanduser("."), name))
#     import torch
#     import numpy as np
#     import librosa
#     import librosa.display
#     # import IPython
#     # from IPython.display import Audio
#     # need this for English text processing frontend
#     import nltk
#     # ! python3 -m nltk.downloader cmudict
#     checkpoint_path = "20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"
#
#
#     # Copy preset file (json) from master
#     # The preset file describes hyper parameters
#     # ! git checkout master --quiet
#     preset = "./presets/deepvoice3_vctk.json"
#     # ! cp -v $preset .
#     # preset = "./deepvoice3_vctk.json"
#
#     # print(os.getcwd())
#     import hparams
#     import json
#
#     for dummy, v in [("fmin", 0), ("fmax", 0), ("rescaling", False),
#                      ("rescaling_max", 0.999),
#                      ("allow_clipping_in_normalization", False)]:
#       #if hparams.hparams.get(dummy) is None:
#         hparams.hparams.add_hparam(dummy, v)
#     # Load parameters from preset
#     with open(preset) as f:
#       hparams.hparams.parse_json(f.read())
#     # Tell we are using multi-speaker DeepVoice3
#     hparams.hparams.builder = "deepvoice3_multispeaker"
#     # Tell we need speaker embedding in 512 dim
#     hparams.hparams.speaker_embed_dim = 512
#
#     # from .deepvoice3_pytorch import synthesis as syn
#     import synthesis
#     import train
#     # print(os.getcwd())
#     from deepvoice3_pytorch import frontend
#     # Inject frontend text processor
#     synthesis._frontend = getattr(frontend, "en")
#     train._frontend =  getattr(frontend, "en")
#
#     # alises
#     fs = hparams.hparams.sample_rate
#     hop_length = hparams.hparams.hop_size
#
#     from train import build_model
#     from train import restore_parts, load_checkpoint
#     from synthesis import tts as _tts
#
#
#     dv3_model = train.build_model()
#     dv3_model = train.load_checkpoint(checkpoint_path, dv3_model, None, True)
#     # Change the working directory back
#     os.chdir("./..")
#
#     return dv3_model


def get_cloned_voices(no_speakers,no_cloned_texts):
    try:
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p" , "rb") as fp:
            cloned_voices = pickle.load(fp)
    except:
        cloned_voices = generate_cloned_samples()
    if(numpy.array(cloned_voices).shape != (no_speakers , no_cloned_texts)):
        cloned_voices = generate_cloned_samples("./Cloning_Audio/cloning_text.txt" ,no_speakers,True,0)

    return cloned_voices

# Assumes that only deep Voice 3 is given
def get_speaker_embeddings(model):
    '''
        return the peaker embeddings and its shape from deep voice 3
    '''
    embed = model.embed_speakers.weight.data
    # shape = embed.shape
    return embed

def build_encoder():
    encoder = Encoder()
    return  encoder



def train_encoder(encoder, speakers, embeddings, batch_size=[1,1], epochs=1000):

	criterion = nn.L1Loss()
	optimizer = torch.optim.SGD(encoder.parameters(),lr=0.002)

	for i in range(epochs):

		for j in range(batch_size[0]):

			for k in range(batch_size[1]):

				elem = speakers[j][k]
				elem = np.reshape(elem, (1,1,elem.shape[0],elem.shape[1]))

				if(k==0):
					inner_inputs = elem
				else:
					inner_inputs = np.hstack((inner_inputs,elem))

			if(j==0):
				true_inputs = inner_inputs
				embed = embeddings[i]
			else:
				true_inputs = np.vstack((true_inputs,inner_inputs))
				embed = np.vstack((embed,embeddings[i]))


		optimizer.zero_grad()
		input_to_encoder = Variable(torch.from_numpy(true_inputs).type(torch.FloatTensor))
		output_from_encoder = encoder(input_to_encoder)

		embeddings = Variable(torch.from_numpy(embed).type(torch.LongTensor))

		loss = criterion(output_from_encoder,embeddings)
		loss.backward()
		optimizer.step()
		save_checkpoint(encoder,optimizer,"encoder_checkpoint.pth",epoch)

def save_checkpoint(model, optimizer, checkpoint_dir,epoch):

    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_epoch": epoch,
        "epoch":epoch+1,

    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":

    #Load Deep Voice 3
    # Pre Trained Model
    dv3_model = build_deepvoice_3()
    # all_speakers = generate_cloned_samples(dv3_model,
    #                                     "./Cloning_Audio/cloning_text.txt",
    #                                     108,True,0)
    # speaker_ebed = get_speaker_embeddings(dv3_model)
    #
    # encoder = build_encoder()
    #
    # # Training The Encoder
    #
    # try:
    #     train_encoder(encoder, all_speakers, speaker_embed, batch_size=[1,1], epochs=1000)
    # except KeyboardInterrupt:
    #     pass
    #
    # print("Finished")
    # sys.exit(0)
