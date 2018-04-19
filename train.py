import pickle

import torch
from torch.Autograd import Variable

from .utils import generate_cloned_samples

from .deepvoice3_pytorch import hparams
from .deepvoice3_pytorch import json
from .deepvoice3_pytorch import synthesis
from .deepvoice3_pytorch import train as dv3train
from .deepvoice3_pytorch/deepvoice3_pytorch import frontend
from .deepvoice3_pytorch/train import build_model
from .deepvoice3_pytorch/train import restore_parts, load_checkpoint
from .deepvoice3_pytorch/synthesis import tts as _tts
from .SpeechEmbedding import Encoder


def build_deepvoice_3():
    checkpoint_path = "./deepvoice3_pytorch/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"
    preset = "./deepvoice3_pytorch/presets/deepvoice3_vctk.json"
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


    dv3_model = build_model()
    dv3_model = load_checkpoint(checkpoint_path, model, None, True)
    return dv3_model


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
    all_speakers = generate_cloned_samples(dv3dv3_model,
                                        "./Cloning_Audio/cloning_text.txt",
                                        108,True,0)
    speaker_ebed = get_speaker_embeddings(dv3_model)

    encoder = build_encoder()

    # Training The Encoder

    try:
        train_encoder(encoder, all_speakers, speaker_embed, batch_size=[1,1], epochs=1000)
    except KeyboardInterrupt:
        pass

    print("Finished")
    sys.exit(0)
