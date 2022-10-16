import torch
import os

def save_checkpoint(out_dir, config, epoch, score, 
                    encoder, decoder, encoder_optimizer, decoder_optimizer, 
                    is_best=False, filename='checkpoint'):
    """
    Save model checkpoint.
    """

    state = {
        'epoch':epoch,
        'score':score,
        'config':config,
        'encoder':encoder,
        'decoder':decoder,
        'encoder_optimizer':encoder_optimizer,
        'decoder_optimizer':decoder_optimizer
    }
    filename = f'{filename}.pth.tar'
    if is_best:
        filename = 'best_'+filename
    torch.save(state, os.path.join(out_dir, filename))


def load_checkpoint(filepath):
    """
    returns: encoder, decoder, encoder_optimizer, decoder_optimizer, 
    """
    # probably should just do this in train.py...
    checkpoint = torch.load(filepath)
    epoch = checkpoint['epoch']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint['decoder_optimizer']
    score = checkpoint['score']
    config = checkpoint['config']
    
