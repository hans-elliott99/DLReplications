from typing import Type
import torch
import torchvision

class Encoder(torch.nn.Module):
    def __init__(self, encoded_image_size=14, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.device = device
        self._init_resnet()  ##load resnet model and prep
        self.finetune(False) ##init model as non-trainable

        # Resize image to fixed size to allow input images of variable sizes
        ## (encoded 'images' will be of size 14x14)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)).to(self.device)
        self.n_params = sum(p.nelement() for p in self.parameters())

    def _init_resnet(self): ##add similair method to load any different torchvision model if desired
        # Load weights
        res_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        # Save preprocessing info
        self.encoder_transforms = res_weights.transforms()
        # Load model
        resnet = torchvision.models.resnet50(weights=res_weights)
        # Remove model head
        layers = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*layers).to(self.device)
    
    def forward(self, images):
        out = self.resnet(images)   ##(batch, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        return out.permute(0, 2, 3, 1) ##(batch, image_size/32, image_size/32, 2048)

    def finetune(self, fine_tune=True):
        """
        Put ResNet into finetune mode, so we don't retrain early convolution layers
        """        
        for p in self.resnet.parameters(): ##set all layers to nontrainable
            p.requires_grad=False

        for chd in list(self.resnet.children())[5:]: ##only train conv blocks from the second-on
            for p in chd.parameters():
                p.requires_grad = fine_tune



class SoftAttention(torch.nn.Module):
    """
    Inspired by Bahdanau et al (2015) - see pg. 12 for key details
    """
    def __init__(self, enc_output_dim, dec_hidden_dim, attention_dim, activ_fn=torch.nn.ReLU, device=torch.device("cpu")):
        super(SoftAttention, self).__init__()
        self.device = device
        self.enc2att = torch.nn.Linear(enc_output_dim, attention_dim, bias=False, device=self.device)    #map from encoded images to attention    
        self.hidden2att = torch.nn.Linear(dec_hidden_dim, attention_dim, bias=False, device=self.device) #map from decoder hidden state to attention
        self.att = torch.nn.Linear(attention_dim, 1, device=self.device)

        self.activ_fn = activ_fn() ##paper uses tanh
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_output, dec_hidden):
        att1 = self.enc2att(encoder_output)                ##(batch_size, num_pixels, attention_dim)
        att2 = self.hidden2att(dec_hidden).unsqueeze(1)    ##(batch_size, 1, attention_dim)

        att = self.att(self.activ_fn(att1 + att2)).squeeze(2) ##(batch_size, num_pixels)
        alpha = self.softmax(att) ##(batch_size, num_pixels)  - add 3rd dimension to broadcast alpha across all encoded feature maps
        context = (encoder_output * alpha.unsqueeze(2)).sum(dim=1) ##(batch_size, encoder_dim)
        return context, alpha


        
class AttentionDecoder(torch.nn.Module):
    def __init__(self, dec_embed_dim, dec_hidden_dim, attention_dim, string2int:Type[object],
                enc_output_dim=2048, activ_fn=torch.nn.ReLU, dropout=0.0,
                device=torch.device("cpu")
    ) -> None:
        """
        
        enc_output_dim: the number of feature maps produced by the encoder (2048 if ResNet)\n
        """
        super(AttentionDecoder, self).__init__()
        self.device = device
        #params
        self.embed_dim = dec_embed_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attention_dim = attention_dim
        self.enc_output_dim = enc_output_dim

        #data specs
        self.vocab_size = len(string2int)
        self.start_tok_idx = string2int.stoi[string2int.start_token]
        self.pad_tok_idx = string2int.stoi[string2int.pad_token]
        self.stop_tok_idx = string2int.stoi[string2int.stop_token]
        self.n_params = sum(p.nelement() for p in self.parameters())

        #core model
        self.attention = SoftAttention(enc_output_dim=enc_output_dim, dec_hidden_dim=dec_hidden_dim,
                                        attention_dim=attention_dim, activ_fn=activ_fn, device=self.device)
        self.output_embed = torch.nn.Embedding(self.vocab_size, self.embed_dim, device=self.device) #word embeddings
        self.dropout = torch.nn.Dropout(p=dropout)
        self.rnn_decode = torch.nn.LSTMCell(self.embed_dim+enc_output_dim, dec_hidden_dim, bias=True, device=self.device)
        self.f_beta = torch.nn.Linear(dec_hidden_dim, enc_output_dim, device=self.device) #layer for sigmoid gate post-rnn
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_head = torch.nn.Linear(dec_hidden_dim, self.vocab_size, device=self.device) #layer to compute probability dist over vocab

        #hidden state + weight init
        self.init_h = torch.nn.Linear(enc_output_dim, dec_hidden_dim, device=self.device) ##learnable layers for initializing hidden states
        self.init_c = torch.nn.Linear(enc_output_dim, dec_hidden_dim, device=self.device)
        self._init_weights()


    def _init_weights(self):
        """
        Prep specific params with better weight initialization for better convergence.
        """
        for child in self.children():
            if isinstance(child, (torch.nn.Linear)):
                torch.nn.init.xavier_normal_(child.weight)
        self.output_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc_head.bias.data.fill_(0)
        self.fc_head.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, encoder_output):
        """
        Initalize first hidden states with the means of the encoded images.  
        Returns hidden, cell
        """
        mean_encoded = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoded) ##(batch_size, num_pixels, dec_hidden_dim)
        c = self.init_c(mean_encoded) ##(same)
        return h, c


    def forward(self, encoder_output, y_encodings):
        self.device = torch.device("cpu")
        batch_size = encoder_output.size(0)   
        encoder_dim = encoder_output.size(-1) ##number of feature maps
        caption_dim = y_encodings.size(1)

        # Flatten encoded images
        encoder_output = encoder_output.view(batch_size, -1, encoder_dim) ##(batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_output.size(1)

        # Sort input data by decreasing length to avoid using pad tokens (we can then use a simple index method to continue decode the longest captions)
        caption_lengths = torch.tensor(
            [len([w.item() for w in caption if w != self.pad_tok_idx]) for caption in y_encodings],
            device=self.device
            )
        caption_lengths, sort_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_output = encoder_output[sort_idx]
        sorted_caps = y_encodings[sort_idx]

        # Also calculate decode lengths - the proper length of decoded output for each caption 
        decode_lengths = (caption_lengths - 1).tolist() ##- 1 since we don't run the 'end' token through the decoder 
        
        # Output Word Embeddings
        embed = self.output_embed(sorted_caps) ##(batch_size, vocab_size, embed_dim)
        
        # Tensors to hold word prediction scores, alphas
        predictions = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=self.device)
        alphas = torch.zeros((batch_size, max(decode_lengths), num_pixels), device=self.device)

        # Initialize hidden states
        hidden, cell = self.init_hidden(encoder_output=encoder_output) ##(batch_size, num_pixels, dec_hidden_dim)

        # Attention + Decoder Recurrent Net
        for t in range(max(decode_lengths)):
            # Determine effective batch size for time step t (to start it will be batch_size)
            batch_size_t = sum([dec_len > t for dec_len in decode_lengths]) ##calculate 'effective batch size' for time step t
            # Forward pass the attention network to calculate the context vector (ie, the attention weighted image encodings)
            context_vec, alpha = self.attention(encoder_output[:batch_size_t],       ##context_vec (batch_size, num_pixels, encoder_dim)
                                                dec_hidden=hidden[:batch_size_t])    ##
            
            # Calculate and apply a sigmoid gate (paper sec. 4.2.1) which may help the model attend
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))
            context_vec = gate * context_vec
            
            # Perform a decoding step with the recurrent net to update hidden states
            ## Pass in the Input: a concat of the embeddings (for the effective batch) with the contect vector
            ## and a tuple, (hidden state, cell state)
            hidden, cell = self.rnn_decode(
                torch.cat((embed[:batch_size_t, t, :], context_vec), dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t])
            ) ## (batch_size_t, vocab_size)

            # Pass hidden state through a FC layer to calculate output logits
            logits = self.fc_head(self.dropout(hidden)) ##(batch_size_t, vocab_size)

            # Add predictions and alphas to save-tensors
            predictions[:batch_size_t, t, :] = logits
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, alphas, decode_lengths, sort_idx



@torch.no_grad()
def beam_evaluate(image, encoder, decoder, 
                 captions=None,
                 beam_size=1):
    """
    Use beam search to perform inference and caption an image. If a true caption is provided, calculate BLEU-4 Score

    image: a single image tensor of shape (1, img_size, img_size, 3)\n
    captions: optional. tensor containing all ground-truth captions associated with the single image (some datasets provide multiple caps per image).\n
    beam_size: the number of candidate words to search over at each step of the decoding.\n
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

    # DECODE 
    step = 1
    hidden, cell = decoder.init_hidden(encoder_out)

    # s <= beam_size (starts as beam_size, decreases as captions finish)
    while True:
        # Produce beam_size (k) embeddings based on the previous word
        embeds = decoder.output_embed(k_prev_words).squeeze(1) ##(s, dec_embed_dim)
        # Calculate the context vector and gate it
        context_vec, _ = decoder.attention(encoder_out, hidden) ##(s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(hidden))
        context_vec = context_vec * gate
        # Update RNN hidden and cell states and produce logits
        hidden, cell = decoder.rnn_decode(
            torch.cat((embeds, context_vec), dim=1),
            (hidden, cell)
        ) ##(s, dec_hidden_dim)
        logits = decoder.fc_head(hidden) ##(s, vocab_size)
        # Log Softmax the logits
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
        prev_word_inds = (top_k_words / decoder.vocab_size).long() ##(s)
        next_word_inds = top_k_words % decoder.vocab_size ##(s)

        # Add new words to sequence (which is a LongTensor)
        seqs = torch.cat(
            (seqs[prev_word_inds], next_word_inds.unsqueeze(1)), dim=1
        ) ##(s, step+1)
        
        # Determine which sequences have not reached the end token
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != decoder.stop_tok_idx]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) ##convert to sets for speed, then back to list
        complete_inds=incomplete_inds ##!
        # set aside complete sequences
        # if len(complete_seqs) > 0: ##!
        complete_seqs.extend(seqs[complete_inds].tolist())
        complete_seq_scores.extend(top_k_scores[complete_inds])
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
    ## references
    reference=None
    if captions is not None: 
        reference = captions.tolist()
        reference = list(
            map(lambda cap: [w for w in cap if w not in {decoder.start_tok_idx, decoder.stop_tok_idx, decoder.pad_tok_idx}],
                reference)
        )
    
    ## hypothesis
    hypothesis = [w for w in final_seq if w not in {decoder.start_tok_idx, decoder.stop_tok_idx, decoder.pad_tok_idx}]

    return hypothesis, reference


        
# from nltk.translate.bleu_score import corpus_bleu
# bleu4 = corpus_bleu(references, hypotheses)
    





