#!/usr/bin/python

from dacite import from_dict
import torch
from nltk.translate.bleu_score import corpus_bleu
import wandb
from dotenv import dotenv_values

import time
import datetime
import random
import string
import os
import sys
import warnings
import traceback 

from .utils.dataload import ImageCaptionDataset, String2Int, load_meta
from .models import ShowAttendTell
from .utils import SaveModel, utilities


#TODO: pretrained word embeddings.

def train_loop(Xy_train:tuple, Xy_valid:tuple, config, device, stoi_dict=None, checkpoint=None,
              log_path=None, modelsave_path=None, batch_print_freq=40, log_wandb=False):
    """Train and evaluate model for the desired epochs. Save metrics and model components as desired.

    Xy_train: tuple of inputs (image paths), labels (image captions) for training.\n
    Xy_train: tuple of inputs (image paths), labels (image captions) for validation.\n
    config: the model and training configuration. See get_config_layout()\n
    device: the torch.device() to train on.\n
    stoi_dict: The string-to-integer dict created by String2Int.save_dict() for converting captions to data. 
                If None, a new mapping is created based on the given labels. Best practice is to provide the stoi_dict based on the full corpus.\n
    checkpoint: A path to a saved model checkpoint. If provided, this model is loaded and training is continued.\n 
    log_path: If not None, metrics are saved to log files in this directory.\n
    modelsave_path: If not None, model checkpoints are saved to this directory.\n
    batch_print_freq: Per epoch, print training updates every 'batch_print_freq' batches.\n
    log_wandb: If True, the model is tracked by Weights&Biases. Requires wandb.login() and wandb.init() to have been called first.\n  
    """
    run_name = ''.join(random.choices(string.ascii_uppercase, k=6))
    if log_path is not None:
        loss_log = utilities.MetricLog(os.path.join(log_path,"losses.txt"), reset=True)
        val_loss_log = utilities.MetricLog(os.path.join(log_path,"val_losses.txt"), reset=True)
        top5acc_log = utilities.MetricLog(os.path.join(log_path,"top5acc.txt"), reset=True)
        val_top5acc_log = utilities.MetricLog(os.path.join(log_path,"val_top5acc.txt"), reset=True)
        bleu4_log = utilities.MetricLog(os.path.join(log_path, "bleu4.txt"), reset=True)

    best_val_loss = 0.0
    best_val_bleu4 = 0.0
    epochs_since_improve = 0
    start_epoch = 0

    # IMPORT DATA ---
    train_paths, train_labels = Xy_train[0], Xy_train[1]
    valid_paths, valid_labels = Xy_valid[0], Xy_valid[1]

    # PREP MODEL ---
    if checkpoint is not None:
        print("Loading model checkpoints.")
        state_dicts, meta = SaveModel.load_checkpoint(checkpoint)
        checkp_config = meta['config']
        start_epoch = meta['info']['epoch'] + 1   ##get the epoch that this model was saved at
        best_val_bleu4 = meta['info']['bleu4']
        stoi_map = String2Int(saved_dict=meta['string2int_dict'])

        ## Ensure that the model dimensions in the current config are that of the checkpoint's config.
        ## Any other config items (for ex learning rates, batch sizes, epochs) can be altered between runs as desired.
        for item in ['attention_dim', 'dec_embed_dim', 'dec_hidden_dim']:
            config[item] = checkp_config[item]
        
        # initialize the encoder and load in the checkpoint weights
        encoder = ShowAttendTell.Encoder(weights='random')
        encoder.load_state_dict(state_dicts['encoder'])
        encoder.finetune(config['enc_finetune'])
        # initialize the decoder and load in the checkpoint weights
        decoder = ShowAttendTell.AttentionDecoder(
            dec_embed_dim=config['dec_embed_dim'], dec_hidden_dim=config['dec_hidden_dim'],
            attention_dim=config['attention_dim'], string2int=stoi_map,
            activ_fn=config['activ_fn'], dropout=config['dropout']
        )
        decoder.load_state_dict(state_dicts['decoder'])

    else:
        # otherwise initialize encoder and decoder with random weights (excluding any pretrained weights/embeddings)
        if stoi_dict:
            stoi_map = String2Int(saved_dict=stoi_dict)
        else:
            ##else make new dict for this run given the labels, might not work if using a checkpoint that saw different data. 
            stoi_map = String2Int(train_labels+valid_labels, start_token=config['start_token'], stop_token=config['stop_token'], 
                                  pad_token=config['pad_token'], remove_punct=config['remove_punct'])

        encoder = ShowAttendTell.Encoder(weights="pretrained")
        encoder.finetune(config['enc_finetune'])

        decoder = ShowAttendTell.AttentionDecoder(
            dec_embed_dim=config['dec_embed_dim'], dec_hidden_dim=config['dec_hidden_dim'],
            attention_dim=config['attention_dim'], string2int=stoi_map,
            activ_fn=config['activ_fn'], dropout=config['dropout']
        )

    config['vocab_size'] = len(stoi_map)
        
    encoder.to(device)
    decoder.to(device)
    assert (next(encoder.parameters()).is_cuda == next(decoder.parameters()).is_cuda), f"ensure models are on {device}"


    print(
        f"Vocab Size = {config['vocab_size'] :,}\t"
        f"Model Params = {encoder.n_params + decoder.n_params :,}\t\t"
        f"Training Samples = {len(train_paths) :,}\t"
        f"Valid Samples = {len(valid_paths) :,}"
    )
    
    # DATASETS+LOADERS ---
    train_dataset = ImageCaptionDataset(X_paths=train_paths,
                                        y_labels=train_labels,
                                        string2int=stoi_map,
                                        transforms=encoder.transforms, ##extracted from the resnet module loaded from torchvision
                                        augmentation=None
                                        )
    valid_dataset = ImageCaptionDataset(X_paths=valid_paths, 
                                        y_labels=valid_labels,
                                        string2int=stoi_map,
                                        transforms=encoder.transforms
                                        )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                                   num_workers=config['workers'], drop_last=True, pin_memory=True
                                                   ) ##pin_memory to speed up data loading: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723 
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=config['workers'], drop_last=True, pin_memory=True
                                                   )

    # LOSS ---
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # OPTIMIZERS ---
    ## Encoder Optimizer - only needed if finetuning the base ConvNet
    if config['enc_finetune']:
        enc_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config['encoder_lr']
        )
        if checkpoint is not None:
            enc_optimizer.load_state_dict(state_dicts['enc_opt'])
    else:
        enc_optimizer = None      ##if not finetuning the encoder, we don't have an optimzer for it
    
    ## Decoder optimizer
    dec_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=config['decoder_lr']
    )
    if checkpoint is not None:
        dec_optimizer.load_state_dict(state_dicts['dec_opt'])


    # wandb 
    if log_wandb:
        try:
            wandb.run.name = run_name
            wandb.watch(encoder, log='all', log_freq=40, idx=1)
            wandb.watch(decoder, log='all', log_freq=40, idx=2)
        except Exception as e:
            print(e, "Ensure wandb.init() has been called prior to training.")
    
    # -*-*--EPOCH LOOP --*-*-
    for epoch in range(start_epoch, config['epochs']+start_epoch):
        
        print(f"EPOCH {epoch+1}/{config['epochs']+start_epoch}")
        start = time.time()

        # Adjust LR
        if epochs_since_improve > 0 and epochs_since_improve % 8 == 0:
            if enc_optimizer:
                new_enclr = utilities.adjust_lr_step(enc_optimizer, 0.5)
            new_declr = utilities.adjust_lr_step(dec_optimizer, 0.5)
            print(f"Decaying LR. EncoderLR: {new_enclr :e}\t DecoderLR: {new_declr :e}")
        
        # Training
        train_loss, train_top5acc = batched_train(
                                    train_dataloader,
                                    encoder, decoder,
                                    enc_optimizer, dec_optimizer,
                                    criterion,
                                    config, device,
                                    epoch, batch_print_freq
                                )

        # Validation
        val_loss, val_top5acc, bleu4 = batched_valid(
                                            valid_dataloader,
                                            encoder, decoder,
                                            criterion, stoi_map,
                                            config, device,
                                            epoch, log_wandb
                                        )
        
        if log_path is not None:
            loss_log.log(epoch, train_loss.mean)
            top5acc_log.log(epoch, train_top5acc.mean)
            val_loss_log.log(epoch, val_loss.mean)
            val_top5acc_log.log(epoch, val_top5acc.mean)
            bleu4_log.log(epoch, bleu4)
        
        if log_wandb:
            wandb.log({
                "loss" : train_loss.mean,
                "val_loss" : val_loss.mean,
                "top5acc" : train_top5acc.mean,
                "val_top5acc" : val_top5acc.mean,
                "val_bleu4" : bleu4
                })

        print(
            f'[Epoch {epoch+1} Final] (et {time.time()-start :.3f}s):\t'
            f'Loss = {train_loss.mean :.5f}\t'
            f'Val_Loss = {val_loss.mean :.5f}\t'
            f'Top5acc = {train_top5acc.mean :.5f}\t'
            f'Val_Top5acc = {val_top5acc.mean :.5f}\t'
            f'Bleu4 = {bleu4 :.5f}\n'
        )

        
        # determine relative performance ...
        is_bleu_best = bleu4 > best_val_bleu4
        is_loss_best = val_loss.mean > best_val_loss
        best_val_bleu4 = max(bleu4, best_val_bleu4)
        best_val_loss = max(val_loss.mean, best_val_loss)
        if not is_bleu_best:
            epochs_since_improve += 1
            print(f"Epochs since improvement: {epochs_since_improve}\n")
        elif modelsave_path is not None:
            sv_st = time.time()
            if enc_optimizer:
                enc_opt_statedict = encoder.state_dict()
            else:
                enc_optimizer = None            
            SaveModel.save_checkpoint(
                out_dir=os.path.abspath(modelsave_path),
                config=config, 
                string2int_dict=stoi_map.save_dict(),
                info={'epoch':epoch, 'bleu4':bleu4, 'val_loss':val_loss.mean},     #provide whatever info
                encoder_sd=encoder.state_dict(), 
                decoder_sd=decoder.state_dict(),
                encoder_optimizer_sd=enc_opt_statedict, 
                decoder_optimizer_sd=dec_optimizer.state_dict(),
                is_best=is_bleu_best,
                filename=run_name
            )
            epochs_since_improve = 0 ##reset to 0 if we have an improvement
            print(f"Saving new best. BLEU4 = {bleu4 :.5E}. ModelSaveTime={time.time()-sv_st :.3f}s\n")
        else:
            epochs_since_improve = 0 ##just reset if not saving   

def batched_train(train_dataloader, encoder, decoder, enc_optim, dec_optim, criterion, config, device, epoch, batch_print_freq):
    encoder.train()
    decoder.train()

    avg_loss = utilities.RunningMean()
    avg_top5acc = utilities.RunningMean()
    avg_batchtime = utilities.RunningMean()
    avg_datatime = utilities.RunningMean()

    start = time.time() ##start here before we load in data
    for i, (x, y, _) in enumerate(train_dataloader):

        x = x.to(device)
        y = y.to(device)
        avg_datatime.update(time.time()-start) ##track how long it takes to load in data

        # FORWARD
        encoded_imgs = encoder(x)
        logits, alphas, decode_lengths, sort_idx = decoder(encoded_imgs, y)

        # Loss
        ## sort target captions as they were sorted by the decoder
        ## ignore the first token in the caption since it is the start token, which we fed the model
        targets = y[sort_idx][:, 1:]
        ## use 'pack_padded_sequences' to remove 'timesteps' which were either padded or not decoded - https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        ## we already determined the decode lengths in the decoder^
        ## do this for the logits and targets
        logits = torch.nn.utils.rnn.pack_padded_sequence(logits, lengths=decode_lengths, batch_first=True)[0].to(device)
        targets = torch.nn.utils.rnn.pack_padded_sequence(targets, lengths=decode_lengths, batch_first=True)[0].to(device)
        
        # LOSS
        loss = criterion(logits, targets)

        # Add doubly stochastic attention regularization (Show Attend Tell section 4.2.1)
        loss += config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # BACKWARD
        if enc_optim:
            enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()

        # Clip gradients
        if config['grad_clip'] is not None:
            utilities.clip_grad_values(dec_optim, config['grad_clip'])
            if enc_optim:
                utilities.clip_grad_values(enc_optim, config['grad_clip'])

        # Updated params
        if enc_optim:
            enc_optim.step()
        dec_optim.step()

        # Metrics
        top5acc = utilities.topkaccuracy(logits, targets, topk=5)
        avg_batchtime.update(time.time() - start)
        avg_top5acc.update(top5acc, n=sum(decode_lengths))
        avg_loss.update(loss.item(), n=sum(decode_lengths))

        if ((i % batch_print_freq == 0) or (i==0)):
                print(
                    f'[{i}/{len(train_dataloader)}]' 
                    f'(BatchTime: {avg_batchtime.value :.3f}s | DataLoadTime: {avg_datatime.value :.3f}s)\t'
                    f'Batch_Loss: {loss.item() :.5f}\t'  ##^ .value gives most recent value not .mean
                    f'Batch_Top5Acc: {top5acc :.5f}'
                )
                sys.stdout.flush()

        start = time.time() ##reset start time

    # Epoch-wise metrics
    return avg_loss, avg_top5acc

@torch.no_grad()
def batched_valid(valid_dataloader, encoder, decoder, criterion, stoi_map, config, device, epoch, log_wandb):
    encoder.eval()
    decoder.eval()

    avg_loss = utilities.RunningMean()
    avg_top5acc = utilities.RunningMean()
    avg_batchtime = utilities.RunningMean()
    avg_datatime = utilities.RunningMean()

    # for blue-4 score
    references = list()
    hypotheses = list()

    # BATCHED INFERENCE
    start = time.time()
    for i, (x, y, allcaps) in enumerate(valid_dataloader):
        x = x.to(device)
        y = y.to(device)
        avg_datatime.update(time.time()-start)

        # FORWARD
        encoded_imgs = encoder(x)
        logits, alphas, decode_lengths, sort_idx = decoder(encoded_imgs, y)

        # Loss
        ## sort target captions as they were sorted by the decoder
        ## ignore the first token in the caption since it is the start token, which we fed the model
        targets = y[sort_idx][:, 1:]
        ## use 'pack_padded_sequences' remove 'timesteps' which were either padded or not decoded - https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        ## we already determined the decode lengths in the decoder^
        ## do this for the logits and targets
        logits_copy = logits.clone()
        logits = torch.nn.utils.rnn.pack_padded_sequence(logits, lengths=decode_lengths, batch_first=True)[0].to(device)
        targets = torch.nn.utils.rnn.pack_padded_sequence(targets, lengths=decode_lengths, batch_first=True)[0].to(device)
        ## calc loss
        loss = criterion(logits, targets)
        
        # Track Metrics
        top5acc = utilities.topkaccuracy(logits, targets, topk=5)

        avg_batchtime.update(time.time() - start)
        avg_top5acc.update(top5acc, n=sum(decode_lengths))
        avg_loss.update(loss.item(), n=sum(decode_lengths))

        start = time.time() ##reset start time

        # BLEU-4 Score
        ## extract references
        allcaps = allcaps[sort_idx]
        if len(allcaps.size()) == 2:       ##if no captions dimension, add 1 (for ex, if just one cap per image)
            allcaps = allcaps.unsqueeze(1) ##(batch_size, caps_per_image, caption_length)
        for j in range(allcaps.size(0)):
            ref = allcaps[j].tolist()
            ref = list(
                map(lambda cap: [stoi_map(w) for w in cap if 
                                    w not in {decoder.start_tok_idx, decoder.stop_tok_idx, decoder.pad_tok_idx}],
                    ref)
            )
            references.append(ref)

        ## determine hypothesis
        _, preds = torch.max(logits_copy, dim=2)
        preds = preds.tolist()
        unpadded_preds = list()
        for i, p in enumerate(preds):
            unpadded_preds.append(
                preds[i][:decode_lengths[i]] ##remove padding on prediction i
            )
        hypotheses.extend([[stoi_map(w) for w in pred] for pred in unpadded_preds])
        assert len(references) == len(hypotheses), f"{len(references), len(hypotheses)}"
    
    # Calculate bleu4 score
    with warnings.catch_warnings(): ##silence UserWarning
        warnings.simplefilter("ignore")
        bleu4 = corpus_bleu(references, hypotheses)

    return avg_loss, avg_top5acc, bleu4



if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict(
        # Model Params
        dec_embed_dim = 666,  ##dimension of caption word embeddings
        dec_hidden_dim = 512, ##dim of hidden states for decoded RNN
        attention_dim = 512,  ##dim of attention network (number of neurons)
        activ_fn = 'relu',      ##activation function used in attention network, relu or tanh for now
        dropout = 0.65,         ##dropout prob., applied to hidden state before network's classifier 
        enc_finetune = True, 

        # Training Params
        workers = 1,      ##cpu workers for data loading
        epochs = 1,
        batch_size = 12,
        encoder_lr = 3e-4,
        decoder_lr = 3e-4,
        alpha_c = 1.,     ##regularization parameter for 'doubly stochastic attention', as in the paper sec. 4.2.1 equation 14
        grad_clip = 5.,   ##clamp gradients at [-5, 5]
        
        # Data Params
        remove_punct = '<>"',
        start_token = '<sos>',
        stop_token = '<eos>',
        pad_token = '<pad>',
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = None       #"./ImageCaptioning/checkpoints/best_mod_100eps.tar",    

    SAMPLES = None           ##set to an int to use just SAMPLES training/valid examples
    REQUIRE_CUDA = True    ##sometimes my pytorch doesn't find the gpu, do i still want to run the script?
    LOG_CONSOLE = True     ##send console output to a log file instead of the console?
    LOG_PATH = "./ImageCaptioning/logs"
    LOG_WANDB = False      ##whether to initialize run tracking with weights&biases
    # SAVE_BEST, SAVE_EVERY...

    if torch.cuda.is_available() != REQUIRE_CUDA:
        sys.exit(f"gpu not found. torch.cuda.is_available() = {torch.cuda.is_available()}")
    if LOG_CONSOLE:
        sys.stdout = open(os.path.abspath(os.path.join(LOG_PATH, 'console-output.txt')), 'w')
    if LOG_WANDB:
        keys = dotenv_values(".env")
        wandb.login(key=keys['WANDB_API_KEY'])
        wandb.init(
            project = 'ConceptualCaps',
            config = config,
            group = 'ShowAttendTell',
            anonymous = 'allow'
        )

    # LOAD IN DATA AND PREP PATHS
    train_meta = load_meta("./ImageCaptioning/data/metadata/train_meta.json")
    train_paths = ['./ImageCaptioning/data'+path for path in train_meta['paths']][:SAMPLES]
    train_labels = train_meta['labels'][:SAMPLES]

    valid_meta = load_meta('./ImageCaptioning/data/metadata/valid_meta.json')
    valid_paths = ['./ImageCaptioning/data'+path for path in valid_meta['paths']][:SAMPLES]
    valid_labels = valid_meta['labels'][:SAMPLES]

    # string to integer mapping
    stoi_dict = load_meta("./ImageCaptioning/data/metadata/stoi_all_loaded_caps.json")

    # START TRANING LOOP
    print(f"Date Time: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"device = {DEVICE}\n")
    start = time.time()
    try:
        train_loop(
            Xy_train=(train_paths, train_labels),
            Xy_valid=(valid_paths, valid_labels),
            config=config,
            device=DEVICE,
            stoi_dict=stoi_dict,
            checkpoint=CHECKPOINT,           ##model checkpoint to load for continued training
            log_path=LOG_PATH,              ##path to save log files to 
            modelsave_path="./ImageCaptioning/checkpoints", ##path to save model checkpoints to
            batch_print_freq=40,
            log_wandb=LOG_WANDB
        )
             
    except:
        print('\n',traceback.format_exc()) ##explicitly print error w/ traceback so it is saved to console log 

    print(f"Total Runtime: {time.time() - start :.3f}\n")
    for k,v in list(config.items())[:11]:
        print(f"{k}:{v}")

    if LOG_CONSOLE:
        sys.stdout.close()