#!/usr/bin/python

import torch
from nltk.translate.bleu_score import corpus_bleu

import time, datetime, os, sys
import warnings, traceback 
import GPUtil

from image_captioning.data.LoadData import ImageCaptionDataset, String2Int, load_meta
from image_captioning.models import ShowAttendTell
from image_captioning.utils import metrics, SaveModel

def train_loop(Xy_train:tuple, Xy_valid:tuple, config, device,
              samples=None, log_path=None, modelsave_path=None, log_console=False, batch_print_freq=40):

    if log_path is not None:
        loss_log = metrics.MetricLog(os.path.join(log_path,"losses.txt"), reset=True)
        val_loss_log = metrics.MetricLog(os.path.join(log_path,"val_losses.txt"), reset=True)
        top5acc_log = metrics.MetricLog(os.path.join(log_path,"top5acc.txt"), reset=True)
        val_top5acc_log = metrics.MetricLog(os.path.join(log_path,"val_top5acc.txt"), reset=True)
        bleu4_log = metrics.MetricLog(os.path.join(log_path, "bleu4.txt"), reset=True)

    best_val_loss = 0.0
    best_val_bleu4 = 0.0
    epochs_since_improve = 0

    # IMPORT DATA ---
    train_paths, train_labels = Xy_train[0], Xy_train[1]
    valid_paths, valid_labels = Xy_train[0], Xy_train[1]
    stoi_map = String2Int(train_labels+valid_labels,
                            start_token=config['start_token'], 
                            stop_token=config['stop_token'], 
                            pad_token=config['pad_token'],
                            remove_punct=config['remove_punct']
    )
    config['vocab_size'] = len(stoi_map)
    config['string2int'] = stoi_map.save_dict()

    # PREP MODEL ---
    encoder = ShowAttendTell.Encoder(weights="pretrained")
    encoder.finetune(config['enc_finetune'])

    decoder = ShowAttendTell.AttentionDecoder(
        dec_embed_dim=config['dec_embed_dim'], dec_hidden_dim=config['dec_hidden_dim'],
        attention_dim=config['attention_dim'], string2int=stoi_map,
        activ_fn=config['activ_fn'], dropout=config['dropout']
    )
    encoder.to(device)
    decoder.to(device)
    assert (next(encoder.parameters()).is_cuda), f"ensure encoder is on {device}"
    assert (next(decoder.parameters()).is_cuda), f"ensure decoder is on {device}"

    print(
        f"Vocab Size = {config['vocab_size'] :,}\t"
        f"Model Params = {encoder.n_params + decoder.n_params :,}\t\t"
        f"Training Samples = {len(train_paths[:samples]) :,}\t"
        f"Valid Samples = {len(valid_paths[:samples]) :,}"
    )

    # DATASETS+LOADERS ---
    train_dataset = ImageCaptionDataset(X_paths=train_paths[:samples],
                                        y_labels=train_labels[:samples],
                                        string2int=stoi_map,
                                        transforms=encoder.transforms, ##extracted from the resnet module loaded from torchvision
                                        augmentation=None
                                        )
    valid_dataset = ImageCaptionDataset(X_paths=valid_paths[:samples], 
                                        y_labels=valid_labels[:samples],
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
    enc_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=config['encoder_lr']
    )

    dec_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=config['decoder_lr']
    )

    gpu = GPUtil.getGPUs()[0]
    # EPOCH LOOP ---
    for epoch in range(config['epochs']):
        
        print(f"EPOCH {epoch+1}/{config['epochs']}")
        start = time.time()
        # adjust lr...

        # Training
        train_loss, train_top5acc = batched_train(
                                    train_dataloader,
                                    encoder, decoder,
                                    enc_optimizer, dec_optimizer,
                                    criterion,
                                    config, device, gpu,
                                    epoch, batch_print_freq
                                )


        # Validation
        val_loss, val_top5acc, bleu4 = batched_valid(
                                            valid_dataloader,
                                            encoder, decoder,
                                            criterion, stoi_map,
                                            config, device,
                                            epoch
                                        )
        
        if log_path is not None:
            loss_log.log(epoch, train_loss.mean)
            top5acc_log.log(epoch, train_top5acc.mean)
            val_loss_log.log(epoch, val_loss.mean)
            val_top5acc_log.log(epoch, val_top5acc.mean)
            bleu4_log.log(epoch, bleu4)

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
            SaveModel.save_checkpoint(
                os.path.abspath(modelsave_path),
                config,
                epoch, bleu4,
                encoder, decoder,
                enc_optimizer, dec_optimizer,
                is_best=is_bleu_best,
                filename=f"mod_{config['epochs']}ep"
            )
            epochs_since_improve = 0 ##reset to 0 if we have an improvement
            print(f"Saving new best. BLEU4 = {bleu4 :.5E}. ModelSaveTime={time.time()-sv_st :.3f}s\n")
    # close all loggers
    # for l in [loss_log, val_loss_log, top5acc_log, val_top5acc_log, bleu4_log]:
    #     l.close()

def batched_train(train_dataloader, encoder, decoder, enc_optim, dec_optim, criterion, config, device, gpu_obj, epoch, batch_print_freq):
    encoder.train()
    decoder.train()

    avg_loss = metrics.RunningMean()
    avg_top5acc = metrics.RunningMean()
    avg_batchtime = metrics.RunningMean()
    avg_datatime = metrics.RunningMean()

    start = time.time() ##start here before we load in data
    for i, (x, y, _) in enumerate(train_dataloader):

        x = x.to(device)
        y = y.to(device)
        assert (x.is_cuda == True) & (y.is_cuda == True), f"put data on {device}"
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
        logits = torch.nn.utils.rnn.pack_padded_sequence(logits, lengths=decode_lengths, batch_first=True)[0].to(device)
        targets = torch.nn.utils.rnn.pack_padded_sequence(targets, lengths=decode_lengths, batch_first=True)[0].to(device)
        ## calc loss
        loss = criterion(logits, targets)

        # BACKWARD
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()

        # Clip grads ?

        # Updated params
        enc_optim.step()
        dec_optim.step()

        # Metrics
        top5acc = metrics.accuracy(logits, targets, topk=5)
        avg_batchtime.update(time.time() - start)
        avg_top5acc.update(top5acc, n=sum(decode_lengths))
        avg_loss.update(loss.item(), n=sum(decode_lengths))

        if ((i % batch_print_freq == 0) or (i==0)):
                print(
                    f'[{i}/{len(train_dataloader)}]' 
                    f'(BatchTime: {avg_batchtime.value :.3f}s | DataLoadTime: {avg_datatime.value :.3f}s | GPUTemp: {gpu_obj.temperature}c)\t'
                    f'Batch_Loss: {loss.item() :.5f}\t'  ##^ .value gives most recent value not .mean
                    f'Batch_Top5Acc: {top5acc :.5f}'
                )
                sys.stdout.flush()

        start = time.time() ##reset start time

    # Epoch-wise metric logging
    return avg_loss, avg_top5acc

@torch.no_grad()
def batched_valid(valid_dataloader, encoder, decoder, criterion, stoi_map, config, device, epoch):
    encoder.eval()
    decoder.eval()

    avg_loss = metrics.RunningMean()
    avg_top5acc = metrics.RunningMean()
    avg_batchtime = metrics.RunningMean()
    avg_datatime = metrics.RunningMean()

    # for blue-4 score
    references = list()
    hypotheses = list()

    # BATCHED INFERENCE
    start = time.time()
    for i, (x, y, allcaps) in enumerate(valid_dataloader):
        x = x.to(device)
        y = y.to(device)
        assert (x.is_cuda == True) & (y.is_cuda == True), f"put data on {device}"
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
        top5acc = metrics.accuracy(logits, targets, topk=5)

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

def get_config_layout():
        return dict(
        # Model Params
        dec_embed_dim = "int, dimension of caption word embeddings",
        dec_hidden_dim = "int, dim of hidden states for decoded RNN",
        attention_dim = "int,  dim of attention network (number of neurons)",
        activ_fn = "like torch.nn.ReLU, activation function used in attention network",
        dropout = "float, dropout prob. applied to hidden state before network's classifier",
        enc_finetune = "bool, whether to finetune encoder", ##NOTE: if false, enc optimizer gets empty list of params which raises an error so need to implement ifelse

        # Training Params
        workers = "int, suggest 1, cpu workers for data loading",
        epochs = "int, training epochs",
        batch_size = "int, batch size for training and validation",
        encoder_lr = "float",
        decoder_lr = "float",

        # Data Params
        remove_punct = "str, like '<>-;'",
        start_token = 'str, like <sos>',
        stop_token = 'str, like <eos>',
        pad_token = 'str, like <pad>',
    )


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict(
        # Model Params
        dec_embed_dim = 512,  ##dimension of caption word embeddings
        dec_hidden_dim = 512, ##dim of hidden states for decoded RNN
        attention_dim = 512,  ##dim of attention network (number of neurons)
        activ_fn = torch.nn.ReLU, ##activation function used in attention network
        dropout = 0.5,         ##dropout prob., applied to hidden state before network's classifier 
        enc_finetune = True, ##TODO: if false, enc optimizer gets empty list of params which raises an error so need to implement ifelse

        # Training Params
        workers = 1,      ##cpu workers for data loading
        epochs = 100,
        batch_size = 12,
        encoder_lr = 2e-4,
        decoder_lr = 4e-4,

        # Data Params
        remove_punct = '<>',
        start_token = '<sos>',
        stop_token = '<eos>',
        pad_token = '<pad>',
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = DEVICE
    SAMPLES = None         ##set to an int to use just SAMPLES training/valid examples (to speed up times)
    BATCH_PRINT_FREQ = 40  ##each epoch, print every n batches
    REQUIRE_CUDA = True    ##sometimes my pytorch doesn't find the gpu, do i still want to run the script?
    LOG_CONSOLE = True     ##send console output to a log file instead of the console?
    LOG_PATH = "./logs"
    MODEL_PATH = "./checkpoints"
    # SAVE_BEST, SAVE_EVERY...

    if torch.cuda.is_available() != REQUIRE_CUDA:
        sys.exit(f"gpu not found - torch.cuda.is_available() = {torch.cuda.is_available()}")
    if LOG_CONSOLE:
        sys.stdout = open(os.path.abspath('./logs/console-output.txt'), 'w')

    train_meta = load_meta("./data/metadata/train_meta.json")
    valid_meta = load_meta('./data/metadata/valid_meta.json')
    train_paths = ['./data'+path for path in train_meta['paths']]
    train_labels = train_meta['labels']
    valid_paths = ['./data'+path for path in valid_meta['paths']]
    valid_labels = train_meta['labels']

    print(f"Date Time: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"device = {DEVICE}\n")
    start = time.time()
    try:
        train_loop(
            Xy_train=(train_paths, train_labels),
            Xy_valid=(valid_paths, valid_labels),
            config=config,
            device=DEVICE,
            samples=SAMPLES,  
            log_path=LOG_PATH,
            modelsave_path=MODEL_PATH,
            log_console=LOG_CONSOLE,
            batch_print_freq=BATCH_PRINT_FREQ
        )
             
    except:
        print('\n',traceback.format_exc()) ##explicitly print error w/ traceback so it is saved to console log 

    print(f"Total Runtime: {time.time() - start :.3f}\n")
    for k,v in list(config.items())[:11]:
        print(f"{k}:{v}")

    if LOG_CONSOLE: sys.stdout.close()