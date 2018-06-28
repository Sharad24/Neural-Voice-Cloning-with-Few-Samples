import pickle

import torch
from torch.autograd import Variable
from torch.utils import data as data_utils
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, Dataloader
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit


from utils import generate_cloned_samples, Speech_Dataset
import dv3

import sys
import os

# sys.path.append('./deepvoice3_pytorch')
from dv3 import build_deepvoice_3
from SpeechEmbedding import Encoder

# print(hparams)

def get_cloned_voices(no_speakers = 108,no_cloned_texts = 23):
    try:
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p" , "rb") as fp:
            cloned_voices = pickle.load(fp)
    except:
        cloned_voices = generate_cloned_samples()
    if(np.array(cloned_voices).shape != (no_speakers , no_cloned_texts)):
        cloned_voices = generate_cloned_samples("./Cloning_Audio/cloning_text.txt" ,no_speakers,True,0)
    print("Cloned_voices Loaded!")
    return cloned_voices

# Assumes that only Deep Voice 3 is given
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


def save_checkpoint(model, optimizer, checkpoint_dir,epoch):

    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_epoch": epoch,
        "epoch":epoch+1,

    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def train_encoder(encoder, data, epochs=100000, after_epoch_download=1000):

	criterion = nn.L1Loss()
	optimizer = torch.optim.SGD(encoder.parameters(),lr=0.0006)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

	for i in range(epochs):

		for voice, embed in enumerate(data):

		optimizer.zero_grad()
		input_to_encoder = Variable(torch.from_numpy(voice).type(torch.FloatTensor))
		output_from_encoder = encoder(input_to_encoder)

		embeddings = Variable(torch.from_numpy(embed).type(torch.LongTensor))

		loss = criterion(output_from_encoder,embeddings)
		loss.backward()
		optimizer.step()

		if i%100==0:
			save_checkpoint(encoder,optimizer,"encoder_checkpoint.pth",i)
        if i%1000==0:
            download_file("encoder_checkpoint.pth")
        if i%8000=0:
            scheduler.step()

def download_file(file_name=None):
    from google.colab import files
    files.download(file_name)


batch_size=16

if __name__ == "__main__":

    #Load Deep Voice 3
    # Pre Trained Model
    dv3_model = build_deepvoice_3(True)

    all_speakers = get_cloned_voices()
    print("Cloning Texts are produced")

    speaker_embed = get_speaker_embeddings(dv3_model)
    #
    encoder = build_encoder()
    print("Encoder is built!")

    speech_data = Speech_Dataset(all_speakers, speaker_embed)
    data_loader = Dataloader(speech_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # Training The Encoder

    try:
        train_encoder(encoder, data_loader, epochs=100000)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    #
    print("Finished")
    sys.exit(0)
