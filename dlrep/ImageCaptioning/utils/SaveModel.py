import torch

import os
import io
import json
import tarfile
import pathlib
import shutil

from dlrep.ImageCaptioning.data.dataload import String2Int, load_meta
from dlrep.ImageCaptioning.models import ShowAttendTell

def save_checkpoint(out_dir, config:dict, string2int_dict:dict, info:dict, 
                    encoder_sd, decoder_sd, encoder_optimizer_sd, decoder_optimizer_sd, 
                    is_best=False, filename='checkpoint'):
    """
    Save model state_dicts to out_dir/filename/. Saves encoder, decoder, their optimizers.
    Also saves the config, string2int_dict, paths to the saved state_dicts, and other information as model 'metadata'.  
    """
    if is_best:
        filename = 'best_'+filename
    if filename.endswith('.tar') == False:
        filename += '.tar'

    temp_dir = pathlib.Path(os.path.join(out_dir, f'TEMP_SAVEMODEL/'))
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    meta = {
        'info':info,
        'config':config,
        'string2int_dict':string2int_dict
    }
    # Save state dicts and meta to temporary directory
    torch.save(encoder_sd, os.path.join(temp_dir, 'encoder_sd.pth.tar'))
    torch.save(decoder_sd, os.path.join(temp_dir, 'decoder_sd.pth.tar'))
    torch.save(encoder_optimizer_sd, os.path.join(temp_dir, 'enc_opt_sd.pth.tar'))
    torch.save(decoder_optimizer_sd, os.path.join(temp_dir, 'dec_opt_sd.pth.tar'))
    with open(os.path.join(temp_dir, 'meta_sd.json'), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4)
    
    # Move all files to model tarfile
    with tarfile.open(os.path.join(out_dir, filename), "w") as tarf:
        for model_file in os.listdir(temp_dir):
            arcname = model_file.split('.')[0]
            tarf.add(os.path.join(temp_dir, model_file), arcname=arcname)
    # Delete temporary directory
    shutil.rmtree(temp_dir)



def load_checkpoint(checkpoint_tar, device=torch.device('cpu')):
    """
    Load a checkpoint as saved by SaveModel.save_checkpoint(). Only specify paths if names within checkpoint_dir/ have been altered.

    checkpoint_tar = path/to/checkpoint/file.tar\n
    device = 'cpu' or 'gpu'. device to map models to.\n
    optimizers = (encoder_optimizer, decoder_optimizer). a tuple(torch.optim, torch.optim), like (torch.optim.Adam, torch.optim.Adam). Set to None to ignore.\n

    returns: (state_dicts:dict containing all the state dicts, the model meta data) 
    """
    # Ensure checkpoint_dir is formatted correctly
    assert tarfile.is_tarfile(checkpoint_tar), "provide .tar file"

    checkpoint_tar = checkpoint_tar.replace('\\', '/')
    filename = checkpoint_tar.split('/')[-1]
    path = '/'.join(checkpoint_tar.split("/")[:-1])+'/'

    with tarfile.open(checkpoint_tar, "r") as tarf:      
        encoder_dict = torch.load(io.BytesIO(tarf.extractfile("encoder_sd").read()), map_location=str(device))    
        decoder_dict = torch.load(io.BytesIO(tarf.extractfile("decoder_sd").read()), map_location=str(device))
        enc_opt_dict = torch.load(io.BytesIO(tarf.extractfile("enc_opt_sd").read()), map_location=str(device))
        dec_opt_dict = torch.load(io.BytesIO(tarf.extractfile("dec_opt_sd").read()), map_location=str(device))
        meta_bytes = io.BytesIO(tarf.extractfile("meta_sd").read())
        meta = json.load(meta_bytes)

    state_dicts = {
        'encoder' : encoder_dict,
        'decoder' : decoder_dict,
        'enc_opt' : enc_opt_dict,
        'dec_opt' : dec_opt_dict
    }
    return state_dicts, meta    