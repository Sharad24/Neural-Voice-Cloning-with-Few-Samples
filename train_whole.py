import pickle

import torch
from torch.autograd import Variable
from torch.utils import data as data_utils
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit


from utils import generate_cloned_samples
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
    if(numpy.array(cloned_voices).shape != (no_speakers , no_cloned_texts)):
        cloned_voices = generate_cloned_samples("./Cloning_Audio/cloning_text.txt" ,no_speakers,True,0)

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
    dv3_model = build_deepvoice_3(True)

    # all_speakers = get_cloned_voices()
    # print("Cloning Texts are produced")
    
    speaker_ebed = get_speaker_embeddings(dv3_model)
    #
    encoder = build_encoder()
    #
    # # Training The Encoder

    # try:
    #     train_encoder(encoder, all_speakers, speaker_embed, batch_size=[1,1], epochs=1000)
    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt")

    #
    # print("Finished")
    # sys.exit(0)
