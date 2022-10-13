import torch
import torchvision
from data.LoadData import String2Int

class Encoder(torch.nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        
        self._init_resnet()

        # Resize image to fixed size to allow input images of variable size
        ## (encoded 'images' will be of size 14x14)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))


    def _init_resnet(self): ##add similair method to load any different torchvision model if desired
        # Load weights
        res_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        # Save preprocessing info
        self.encoder_transforms = res_weights.transforms()
        # Load model
        resnet = torchvision.models.resnet50(weights=res_weights)
        # Remove model head
        layers = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*layers)
    
    def forward(self, images):
        out = self.resnet(images)   ##(batch, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        return out.permute(0, 2, 3, 1) ##(batch, image_size/32, image_size/32, 2048)

    # def finetune()
 


class SoftAttention(torch.nn.Module):
    """
    Inspired by Bahdanau et al (2015) - see pg. 12 for key details
    """
    def __init__(self, enc_output_dim, dec_hidden_dim, attention_dim, activ_fn=torch.nn.ReLU):
        super(SoftAttention, self).__init__()

        self.enc2att = torch.nn.Linear(enc_output_dim, attention_dim, bias=False)    #map from encoded images to attention    
        self.hidden2att = torch.nn.Linear(dec_hidden_dim, attention_dim, bias=False) #map from decoder hidden state to attention
        self.att = torch.nn.Linear(attention_dim, 1)

        self.activ_fn = activ_fn() ##paper uses tanh
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_output, dec_hidden):
        att1 = self.enc2att(encoder_output)                ##(batch_size, num_pixels, attention_dim)
        att2 = self.hidden2att(dec_hidden).unsqueeze(1)    ##(batch_size, 1, attention_dim)

        att = self.att(self.activ_fn(att1 + att2)).squeeze(2) ##(batch_size, num_pixels)
        alpha = self.softmax(att) ##(batch_size, num_pixels)  - add 3rd dimension to broadcast alpha across all encoded feature maps
        context = (encoder_output * alpha.unsqueeze(2)).sum(dim=1) ##(batch_size, encoder_dim)
        return context, alpha


        
class Decoder(torch.nn.Module):
    def __init__(self, embed_dim, dec_hidden_dim, attention_dim, string2int:String2Int,
                enc_output_dim=2048, activ_fn=torch.nn.ReLU, dropout=0.0
    ) -> None:
        """
        
        enc_output_dim: the number of feature maps produced by the encoder (2048 if ResNet)\n
        """
        super(Decoder, self).__init__()

        self.device = torch.device("cpu")
        #params
        self.embed_dim = embed_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attention_dim = attention_dim
        self.enc_output_dim = enc_output_dim

        #data specs
        self.vocab_size = len(string2int)
        self.pad_tok_idx = string2int.stoi[string2int.pad_token]
        
        #core model
        self.attention = SoftAttention(enc_output_dim=enc_output_dim, dec_hidden_dim=dec_hidden_dim,
                                        attention_dim=attention_dim, activ_fn=activ_fn)
        self.output_embed = torch.nn.Embedding(self.vocab_size, embed_dim) #word embeddings
        self.dropout = torch.nn.Dropout(p=dropout)
        self.rnn_decode = torch.nn.LSTMCell(embed_dim+enc_output_dim, dec_hidden_dim, bias=True)
        self.f_beta = torch.nn.Linear(dec_hidden_dim, enc_output_dim) #layer for sigmoid gate post-rnn
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_head = torch.nn.Linear(dec_hidden_dim, self.vocab_size) #layer to compute probability dist over vocab

        #hidden state + weight init
        self.init_h = torch.nn.Linear(enc_output_dim, dec_hidden_dim) ##learnable layers for initializing hidden states
        self.init_c = torch.nn.Linear(enc_output_dim, dec_hidden_dim)
        # self._init_weights()

    def _init_weights(self):
        """
        Prep specific params with better weight initialization for better convergence.
        """
        self.output_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc_head.bias.data.fill_(0)
        self.fc_head.weight.data.uniform_(-0.1, 0.1)

    def _init_hidden(self, encoder_output):
        """
        Initalize first hidden states with the means of the encoded images
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
        y_encodings = y_encodings[sort_idx]

        # Also calculate decode lengths - the proper length of decoded output for each caption 
        decode_lengths = (caption_lengths - 1).tolist() ##- 1 since we don't run the 'end' token through the decoder 
        
        # Output Word Embeddings
        embed = self.output_embed(y_encodings) ##(batch_size, vocab_size, embed_dim)

        # Initialize hidden states
        hidden, cell = self._init_hidden(encoder_output=encoder_output) ##(batch_size, num_pixels, dec_hidden_dim)
        
        # Tensors to hold word prediction scores, alphas
        predictions = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=self.device)
        alphas = torch.zeros((batch_size, max(decode_lengths), num_pixels), device=self.device)


        # Decoder Recurrent Net
        for t in range(max(decode_lengths)):
            # Determine effective batch size for time step t (to start it will be batch_size)
            batch_size_t = sum([dec_len > t for dec_len in decode_lengths]) ##calculate 'effective batch size' for time step t
            # Forward pass the attention network to calculate the context vector (ie, the attention weighted image encodings)
            context_vec, alpha = self.attention.forward(encoder_output[:batch_size_t],       ##context_vec (batch_size, num_pixels, encoder_dim)
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
        
        return predictions, alphas, decode_lengths