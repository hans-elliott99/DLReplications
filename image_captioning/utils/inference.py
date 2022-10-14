from pickletools import string1
import torch
import torchvision
from nltk.translate.bleu_score import corpus_bleu

import PIL.Image
import os, warnings

import matplotlib.pyplot as plt


@torch.no_grad()
def beam_caption(image, encoder, decoder, string2int, 
                 captions=None,
                 beam_size=1):
    """
    Use beam search to perform inference and caption an image. If a true caption(s) is provided, calculate BLEU-4 Score

    image: a single image tensor of shape (1, img_size, img_size, 3)\n
    captions: optional. tensor containing all ground-truth captions associated with the single image (some datasets provide multiple caps per image).\n
    beam_size: the number of candidate words to search over at each step of the decoding.\n
    Device inferred from encoder/decoder
    """
    if len(image.size()) == 3:
        image = image.unsqueeze(0) ##add batch dimension
    elif len(image.size()) == 4:
        assert image.size()[0] == 1, "Provide a single image"
    
    device = encoder.device
    assert device==image.device

    # ENCODE
    encoder_out = encoder(image) ##(1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1) ##14
    encoder_dim = encoder_out.size(3) ##2048

    # Flatten encoded representation
    encoder_out = encoder_out.view(1, -1, encoder_dim) ##(1, num_pixels (14x14), encoder_dim)
    num_pixels = encoder_out.size(1) ##enc_image_size x enc_image_size 

    # Treat as having batch size of 'beam_size' - so copy encoder output beam_size times so that we have beam_size encoder_outputs
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim) ##(beam_size, num_pixels, encoder_dim)

    # Storing tensors
    ## Top k previous words at each step - fill with <sos>
    k_prev_words = torch.LongTensor([[decoder.start_tok_idx]] * beam_size).to(device) ##(beam_size, 1)
    ## Top k sequences - fill with <sos>
    seqs = k_prev_words
    ## Top k sequence scores - fill with 0s
    top_k_scores = torch.zeros(beam_size, 1).to(device) ##(beam_size, 1)

    ## Store completed sequences and score
    complete_seqs = list()
    complete_seq_scores = list()

    ## Store alphas (attention scores) to visualize attention
    alphas = list() ##could be torch.tensor of <pad>, remove pads later

    # DECODE 
    step = 1
    hidden, cell = decoder.init_hidden(encoder_out)

    # s <= beam_size (starts as beam_size, decreases as captions finish)
    while True:
        # Produce beam_size (k) embeddings based on the previous word
        embeds = decoder.output_embed(k_prev_words).squeeze(1) ##(s, dec_embed_dim)

        # Calculate the context vector and gate it
        context_vec, alpha = decoder.attention(encoder_out, hidden) ##(s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(hidden))
        context_vec = context_vec * gate

        # Update RNN hidden and cell states and produce logits
        hidden, cell = decoder.rnn_decode(
            torch.cat((embeds, context_vec), dim=1),
            (hidden, cell)
        ) ##(s, dec_hidden_dim)
        logits = decoder.fc_head(hidden) ##(s, vocab_size)

        # Log Softmax the logits to produce scores
        scores = torch.nn.functional.log_softmax(logits, dim=1) ##(s, vocab_size)
        # Add scores to top_k_scores to produce an additive scores
        scores = top_k_scores.expand_as(scores) + scores

        # For step 1, all k points will have the same score since k_prev_words are all <start>
        # Otherwise, need to unroll the scores and find their unrolled top scores + indices
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k=beam_size, dim=0, largest=True, sorted=True) ##(s),(s)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k=beam_size, dim=0, largest=True, sorted=True)  ##(s),(s)

        # Convert unrolled indices to actual indices by dividing by the vocab size
        prev_word_inds = (top_k_words / decoder.vocab_size).long() ##(s) [0, beam_size]
        next_word_inds = top_k_words % decoder.vocab_size ##(s) [0, vocab_size]

        # Add new words to sequence (which is a LongTensor), storing the (beam_size) previous words with highest scores and the new words 
        seqs = torch.cat(
            (seqs[prev_word_inds], next_word_inds.unsqueeze(1)), dim=1
        ) ##(s, step+1)
        
        # Determine which sequences have not reached the end token
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != decoder.stop_tok_idx]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) ##convert to sets for speed, then back to list

        # set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seq_scores.extend(top_k_scores[complete_inds])

        # store alphas for completed sequence steps
        if step == 1: ##first step, alphas are same for all beam_size dimensions since the decoder saw the same <start> token for each beam
            alphas.append(alpha[0].view(-1))
        else: ##otherwise, save the alpha corresponding to the word that has been decided on 
            alpha = alpha[prev_word_inds[complete_inds]]
            #for i in alpha.size(0):
            alphas.append(alpha.view(-1))

        # reduce beam size
        beam_size -= len(complete_inds)

        # proceed with incomplete sequences unless all are complete
        if beam_size==0:
            break
        seqs = seqs[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        hidden = hidden[prev_word_inds[incomplete_inds]]
        cell = cell[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]

        # break if decoding takes too long
        if step > 50:
            break
        step += 1

    # determine the indices of the best scores and use to determine the final sequence
    best_i = complete_seq_scores.index(max(complete_seq_scores))
    final_seq = complete_seqs[best_i]
    
    # BLEU-4 Score
    references = list() ##see NLTK bleu-4 docs - references need to be list like [[cap1word1, cap1word2, cap1word3], [cap2word1, cap2word2]] for *each* image
    hypotheses = list()  ##see NLTK bleu-4 docs - hypothesis needs to be list [hyp-word1, hyp-word2, ...] for each image
    # references
    if captions is not None: 
        if len(captions.size())==2: ##add dimension so size = (# of captions for this image, 1, padded length of captions)
            captions = captions.unsqueeze(1) ##(1, caps_per_image, caption_length)
        for j in range(captions.size(0)): ##for each caption (or once if just one caption)
            ref = captions[j].tolist()
            ref = list(
                map(lambda cap: [string2int(w) for w in cap if 
                                    w not in {decoder.start_tok_idx, decoder.pad_tok_idx}],
                    ref)
            )
            references.append(ref[0])
    
    # hypothesis
    hypotheses.extend([[string2int(w) for w in final_seq if w not in {decoder.start_tok_idx, decoder.pad_tok_idx}]])
    # return references, hypotheses
    assert len(references) == len(hypotheses)

    bleu4 = None
    with warnings.catch_warnings(): ##silence UserWarning
        warnings.simplefilter("ignore")
        bleu4 = corpus_bleu(hypotheses, references)

    caption = ' '.join([w for w in hypotheses[0] if w not in {string2int.stop_token}])
    return caption, final_seq, bleu4, alphas




def load_prep_image(image_path, transforms=None):
    """
    Load image from filepath and apply preprocessing steps.  
    """    
    # Read image and process
    img = PIL.Image.open(image_path)
    return _preprocess_image(img, transforms=transforms)

def _preprocess_image(img, transforms=None):
    """Apply preprocessing to an image to prepare for model inference."""
    img = img.resize((256, 256), resample=PIL.Image.Resampling.BICUBIC)
    img = torchvision.transforms.ToTensor()(img).unsqueeze(0) ##add batch dimension
    img = img.float()
    # img *= 1.0/255 ##ToTensor scales to [0.,1.] so don't rescale manually
    if transforms is not None:
        img = transforms(img)
    return img

if __name__=='__main__':
    image_path = os.path.abspath("./data/train_images/0.jpg")
    img = beam_caption(image_path)
    
    print(torch.max(img))
    print(img.shape)
    plt.imshow(img.squeeze(0).permute(1, 2, 0))
    plt.show()