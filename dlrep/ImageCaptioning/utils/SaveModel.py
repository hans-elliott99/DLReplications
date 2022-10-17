import torch
import os
import json
from pathlib import Path
import shutil

def save_checkpoint(out_dir, config, string2int_dict, epoch, score, 
                    encoder_sd, decoder_sd, encoder_optimizer_sd, decoder_optimizer_sd, 
                    is_best=False, filename='checkpoint'):
    """
    Save model checkpoint.
    """
    if is_best:
        filename = 'best_'+filename

    dirpath = Path(os.path.join(out_dir, filename))
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    meta = {
        'epoch':epoch,
        'score':score,
        'config':config,
        'string2int':string2int_dict,
        'encoder_state':str(os.path.join(dirpath, 'encoder_'+filename+'.pth.tar')),
        'decoder_state':str(os.path.join(dirpath, 'decoder_'+filename+'.pth.tar')),
        'encoder_optimizer_state':str(os.path.join(dirpath, 'enc_opt_'+filename+'.pth.tar')),
        'decoder_optimizer_state':str(os.path.join(dirpath, 'dec_opt_'+filename+'.pth.tar'))
    }
    # Save state dicts
    torch.save(encoder_sd, meta['encoder_state'])
    torch.save(decoder_sd, meta['decoder_state'])
    torch.save(encoder_optimizer_sd, meta['encoder_optimizer_state'])
    torch.save(decoder_optimizer_sd, meta['decoder_optimizer_state'])
    # Save meta
    with open(os.path.join(dirpath, 'meta_'+filename+'.json'), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4)

###TODO
def load_checkpoint(filepath):
    """
    returns: encoder, decoder, encoder_optimizer, decoder_optimizer, 
    """
    checkpoint = torch.load(filepath)
    epoch = checkpoint['epoch']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint['decoder_optimizer']
    score = checkpoint['score']
    config = checkpoint['config']
    
