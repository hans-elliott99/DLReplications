import torch
import os
import json

def save_checkpoint(out_dir, config, epoch, score, 
                    encoder_sd, decoder_sd, encoder_optimizer_sd, decoder_optimizer_sd, 
                    is_best=False, filename='checkpoint'):
    """
    Save model checkpoint.
    """
    _filename = filename
    filename = f'{filename}.pth.tar'
    if is_best:
        filename = 'best_'+filename

    meta = {
        'epoch':epoch,
        'score':score,
        'config':config,
        'encoder_state':str(os.path.join(out_dir, 'encoder_'+filename)),
        'decoder_state':str(os.path.join(out_dir, 'decoder_'+filename)),
        'encoder_optimizer_state':str(os.path.join(out_dir, 'enc_opt_'+filename)),
        'decoder_optimizer_state':str(os.path.join(out_dir, 'dec_opt_'+filename))
    }
    # Save state dicts
    torch.save(encoder_sd, meta['encoder_state'])
    torch.save(decoder_sd, meta['decoder_state'])
    torch.save(encoder_optimizer_sd, meta['encoder_optimizer_state'])
    torch.save(decoder_optimizer_sd, meta['decoder_optimizer_state'])
    # Save meta
    with open(os.path.join(out_dir, 'meta_'+_filename+'.json'), 'w') as f:
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
    
