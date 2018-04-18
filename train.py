import pickle

import torch
from torch.Autograd import Variable

from utils import generate_cloned_samples


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
    shape = embed.shape
    return embed,shape


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
		save_checkpoint({
            	'epoch': epoch + 1,
            	'arch': args.arch,
            	'state_dict': model.state_dict(),
            	'best_prec1': best_prec1,
            	'optimizer' : optimizer.state_dict(),
        	})


def save_checkpoint(state,filename = "model_checkpoint.pth.tar"):
	torch.save(state, filename)


























					
		
		
	
	
