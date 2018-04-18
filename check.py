import pickle

from .helper import generate_cloned_samples


def get_cloned_voices(no_speakers,no_cloned_texts):
    try:
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p" , "rb") as fp:
            cloned_voices = pickle.load(fp)
    except:
        cloned_voices = generate_cloned_samples()
    if(numpy.array(cloned_voices).shape != (no_speakers , no_cloned_texts))
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
    
