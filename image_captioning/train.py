import torch
import time
import os
import matplotlib.pyplot as plt

from data.DownloadData import load_meta
from data.LoadData import ImageCaptionDataset, String2Int
from models import ShowAttendTell
from utils import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.cuda.empty_cache()

print(f"device = {device}")
SAMPLES = 100
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
    epochs = 1
    batch_size = 16
    encoder_lr = 1e-4
    decoder_lr = 3e-4
    print_freq = 4

    # Data Params
    remove_punct = '<>'
    start_token = '<sos>'
    stop_token = '<eos>'
    pad_token = '<pad>'


avg_loss = metrics.RunningMean()
avg_top5acc = metrics.RunningMean()
avg_batchtime = metrics.RunningMean()
avg_datatime = metrics.RunningMean()

log_path = os.path.abspath("./logs")
losses = metrics.MetricLog(os.path.join(log_path,"losses.txt"), reset=True)
val_losses = metrics.MetricLog(os.path.join(log_path,"val_losses.txt"), reset=True)
top5accuracies = metrics.MetricLog(os.path.join(log_path,"top5acc.txt"), reset=True)

def main():
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


    longest_caption_length = len(max([line.split() for line in train_meta['labels']], key=len))
    print(f"vocab size = {config.vocab_size}")
    print(f"longest caption length = {longest_caption_length}")

    
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
    print(f"Total Params: {encoder.n_params + decoder.n_params :,}")

    encoder.to(device)
    decoder.to(device)
    # DATASETS+LOADERS ---
    train_dataset = ImageCaptionDataset(X_paths=train_paths[:SAMPLES],
                                        y_labels=train_meta['labels'][:SAMPLES],
                                        string2int=stoi_map,
                                        transforms=encoder.encoder_transforms ##extracted from the resnet module loaded from torchvision
                                        )
    valid_dataset = ImageCaptionDataset(X_paths=valid_paths, 
                                        y_labels=valid_meta['labels'],
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

        # Batched Training
        batched_train(
            train_dataloader,
            encoder, decoder,
            enc_optimizer, dec_optimizer,
            criterion,
            epoch=epoch
        )

        # Validation
        #...



def batched_train(train_dataloader, encoder, decoder, enc_optim, dec_optim, criterion, epoch):
    start = time.time()
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        assert (x.device == device) & (y.device == device), f"put data on {device}"

        # FORWARD
        encoded_imgs = encoder(x)
        logits, alphas, decode_lengths, sorted_caps = decoder(encoded_imgs, y)

        # Loss
        ## ignore the first token in the caption since it is the start token, which we fed the model
        targets = sorted_caps[:, 1:]
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

        losses.update(epoch=epoch, value=loss.item())
        # val_losses.update(epoch, )
        top5accuracies.update(epoch, top5acc)

        if i % config.print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(train_dataloader)}]\n'
                f'BatchTime: {avg_batchtime.mean :.3f}\t'
                f'DataLoadTime: {avg_datatime.mean :.3f}\t'
                f'Loss: {avg_loss.mean :.5f}\t'
                f'Top5Acc: {avg_top5acc.mean :.5f}'
            )


if __name__ == '__main__':
    main()

    losses = metrics.LoadMetricLog("./logs/losses.txt")
    val_losses = metrics.LoadMetricLog("./logs/val_losses.txt")
    top5acc = metrics.LoadMetricLog("./logs/top5acc.txt")
    
    plt.title("Loss")
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="valid")
    plt.legend()
    plt.show()

    plt.title("Accuracy")
    plt.plot(top5acc, label="train")
    plt.legend()
    plt.show()