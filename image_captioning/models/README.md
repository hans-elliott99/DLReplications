# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
Xu et al. 2016 - http://proceedings.mlr.press/v37/xuc15.pdf

A ConvNet is used to encode images. In the paper an Oxford VGGNet is used, and here I used a ResNet.  
Details:
    - An adaptivel average pooling layer is used to resize input images so that their encoded representations are of a consistent size, even if input images are of variable size.   

The encoded images are passed to a soft attention model and an RNN decoder (an LSTM is used as in the paper).  

The attention model receives the encoded images and the decoder's previous hidden state and computes attention scores. The attention scores are applied like weights to the encoded images (creating the 'context vector'), effectively telling the decoder where to look for useful information as it predicts the next word in the caption.    
Details:
    - Hidden states are initialized with the mean of the encoded image.  
    - In section 4.2.1 of Xu et al., they recommend applying a gate to the context vector to further improve the model's ability to focus on important image regions. This can be achieved using a sigmoid layer which is fed the previous hidden state.  

Finally, the context vector is passed to the RNN decoder along with caption word embeddings and with the previous hidden state (and cell state since using LSTM).  
The word embeddings are concatenated with the context vector.  
The RNN produces an updated hidden state (and cell state), which is passed through a fully connected linear layer to produce V (per image) logits/scores where V is the size of the vocabulary (ie, the 'log probability' dist. over the next word in the caption).  

Training:  
    - In training, for any given batch the inputs are sorted by the word length of their captions, in decreasing order. This is so we can efficiently avoid decoding 'pad' tokens, which we only add to the captions so that we can stack them in a single tensor.  
    - We also track 'decode_length' for each image which is the number of non-pad/non-stop characters - this is the number of decoding steps required for the given image.
    - Thus for each timestep in the decoding, we determine which images need to be decoded (their decode_length is less than the current time step) and use this as our 'effective batch' to pass through the decoder.  
    - By keeping the images & captions sorted in descending order, we can continuously track logits and attention scores for each timestep (if a given image does not need decoding at the current timestep, its value in the prediction tensor remains 0...)  
    - To compute loss, we compare the logits tensor (which contains predicted log probs. over words for each step in a given images deocding) to the target word tokens. First we use pytorch's 'pack_padded_sequences' to remove the timesteps from the logit and target tensors that were not actually decoded (ie, where pad tokens were).  
     
