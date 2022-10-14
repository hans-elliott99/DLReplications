import torch

import time, os, sys, warnings
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

from data.DownloadData import load_meta
from data.LoadData import ImageCaptionDataset, String2Int
from models import ShowAttendTell
from utils import metrics, SaveModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.cuda.empty_cache()


SAMPLES = None         ##set to an int to use just SAMPLES training/valid examples (to speed up times)
BATCH_PRINT_FREQ = 40  ##

class config:
    # Model Params
    dec_embed_dim = 33  ##dimension of caption word embeddings
    dec_hidden_dim = 64 ##dim of hidden states for decoded RNN
    attention_dim = 64
    activ_fn = torch.nn.ReLU
    dropout = 0.0
    enc_finetune = True ##NOTE: if false, enc optimizer gets empty list of params which raises an error so need to implement ifelse

    # Training Params
    device = device
    workers = 1
    epochs = 3
    batch_size = 16
    encoder_lr = 1e-4
    decoder_lr = 3e-4

    # Data Params
    remove_punct = '<>'
    start_token = '<sos>'
    stop_token = '<eos>'
    pad_token = '<pad>'


log_path = os.path.abspath("./logs")
loss_log = metrics.MetricLog(os.path.join(log_path,"losses.txt"), reset=True)
val_loss_log = metrics.MetricLog(os.path.join(log_path,"val_losses.txt"), reset=True)
top5acc_log = metrics.MetricLog(os.path.join(log_path,"top5acc.txt"), reset=True)
val_top5acc_log = metrics.MetricLog(os.path.join(log_path,"val_top5acc.txt"), reset=True)
bleu4_log = metrics.MetricLog(os.path.join(log_path, "bleu4.txt"), reset=True)

def main():
    best_val_loss = 0.0
    best_val_bleu4 = 0.0
    epochs_since_improve = 0

    # IMPORT DATA ---
    train_meta = load_meta("./data/metadata/train_meta.json")
    valid_meta = load_meta('./data/metadata/valid_meta.json')
    train_paths = ['./data'+path for path in train_meta['paths']]
    valid_paths = ['./data'+path for path in valid_meta['paths']]
    stoi_map = String2Int(train_meta['labels']+valid_meta['labels'],
                            start_token=config.start_token, 
                            stop_token=config.stop_token, 
                            pad_token=config.pad_token,
                            remove_punct=config.remove_punct
    )
    config.vocab_size = len(stoi_map)
    # longest_caption_length = len(max([line.split() for line in train_meta['labels']], key=len))
    # print(f"longest caption length = {longest_caption_length}")

    
    # PREP MODEL ---
    encoder = ShowAttendTell.Encoder(device=device)
    encoder.finetune(config.enc_finetune)

    decoder = ShowAttendTell.AttentionDecoder(
        dec_embed_dim=config.dec_embed_dim, dec_hidden_dim=config.dec_hidden_dim,
        attention_dim=config.attention_dim, string2int=stoi_map,
        activ_fn=config.activ_fn, dropout=config.dropout,
        device=device
    )
    assert (encoder.device == device) & (decoder.device == device), f"ensure encoder and decoder are on {device}"

    print(
        f"Vocab Size = {config.vocab_size}\t"
        f"Model Params = {encoder.n_params + decoder.n_params :,}\t"
        f"Training Samples = {len(train_paths[:SAMPLES])}"
    )

    encoder.to(device)
    decoder.to(device)
    # DATASETS+LOADERS ---
    train_dataset = ImageCaptionDataset(X_paths=train_paths[:SAMPLES],
                                        y_labels=train_meta['labels'][:SAMPLES],
                                        split="train",
                                        string2int=stoi_map,
                                        transforms=encoder.encoder_transforms, ##extracted from the resnet module loaded from torchvision
                                        augmentation=None
                                        )
    valid_dataset = ImageCaptionDataset(X_paths=valid_paths, 
                                        y_labels=valid_meta['labels'],
                                        split="valid",
                                        string2int=stoi_map,
                                        transforms=encoder.encoder_transforms
                                        )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                                   num_workers=config.workers, drop_last=True, pin_memory=True
                                                   ) ##pin_memory to speed up data loading: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723 
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=config.workers, drop_last=True, pin_memory=True
                                                   )

    # LOSS ---
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # OPTIMIZERS ---
    enc_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=config.encoder_lr
    )

    dec_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=config.decoder_lr
    )

    
    # EPOCH LOOP ---
    for epoch in range(config.epochs):
        # adjust lr...

        # Training
        train_loss, train_top5acc = batched_train(
                                    train_dataloader,
                                    encoder, decoder,
                                    enc_optimizer, dec_optimizer,
                                    criterion,
                                    epoch=epoch
                                )
        loss_log.log(epoch, train_loss.mean)
        top5acc_log.log(epoch, train_top5acc.mean)


        # Validation
        val_loss, val_top5acc, bleu4 = batched_valid(
                                            valid_dataloader,
                                            encoder, decoder,
                                            criterion,
                                            stoi_map=stoi_map,
                                            epoch=epoch
                                        )
        val_loss_log.log(epoch, val_loss.mean)
        val_top5acc_log.log(epoch, val_top5acc.mean)
        bleu4_log.log(epoch, bleu4)


        print(
            f'EPOCH [{epoch}/{config.epochs}]:\t'
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
            print(f"Epochs since loss improvement: {epochs_since_improve}\n")
        else:
            SaveModel.save_checkpoint(
                os.path.abspath("./checkpoints"),
                config,
                epoch, bleu4,
                encoder, decoder,
                enc_optimizer, dec_optimizer,
                is_best=is_bleu_best,
                filename='best_mod'
            )
            epochs_since_improve = 0 ##reset to 0 if we have an improvement
            print("\n")
    
    # close all loggers
    for l in [loss_log, val_loss_log, top5acc_log, val_top5acc_log, bleu4_log]:
        l.close()

def batched_train(train_dataloader, encoder, decoder, enc_optim, dec_optim, criterion, epoch):
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
        assert (x.device == device) & (y.device == device), f"put data on {device}"
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

        if ((i % BATCH_PRINT_FREQ == 0) or (i==0)):
                print(
                    f'[ep {epoch}][{i}/{len(train_dataloader)}] '
                    f'(BatchTime: {avg_batchtime.value :.3f}s | DataLoadTime: {avg_datatime.value :.3f}s)\t'
                    f'Batch_Loss: {loss.item() :.5f}\t'
                    f'Batch_Top5Acc: {top5acc :.5f}\n'
                )
                sys.stdout.flush()

        start = time.time() ##reset start time

    # Epoch-wise metric logging
    return avg_loss, avg_top5acc

@torch.no_grad()
def batched_valid(valid_dataloader, encoder, decoder, criterion, stoi_map, epoch):
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
        assert (x.device == device) & (y.device == device), f"put data on {device}"
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
            allcaps = allcaps.unsqueeze(1) 
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
    print(f"device = {device}\n")
    main()

    losses = metrics.LoadMetricLog("./logs/losses.txt")
    val_losses = metrics.LoadMetricLog("./logs/val_losses.txt")
    top5acc = metrics.LoadMetricLog("./logs/top5acc.txt")
    val_top5acc = metrics.LoadMetricLog("./logs/val_top5acc.txt")
    bleu4 = metrics.LoadMetricLog("./logs/bleu4.txt")
    
    plt.figure(figsize=(20,10))
    
    plt.subplot(221)
    plt.title("Loss")
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="valid")
    plt.legend()

    plt.subplot(222)
    plt.title("Accuracy")
    plt.plot(top5acc, label="train")
    plt.plot(val_top5acc, label="valid")
    plt.legend()

    plt.subplot(223)
    plt.title("BLEU 4 Score")
    plt.plot(bleu4, label="valid")
    plt.legend()

    plt.show(block=False)