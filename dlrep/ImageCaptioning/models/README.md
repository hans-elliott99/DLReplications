# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
Paper: Xu et al. 2016 - http://proceedings.mlr.press/v37/xuc15.pdf  

Code Inspiration: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/  

## Encoder
A ConvNet is used to **encode** images. In the paper an Oxford VGGNet is used, and here I used a ResNet.  
Implementation Details:  
- An adaptive average pooling layer is used to resize input images so that their encoded representations are of a consistent size, even if input images are of variable size. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/models/ShowAttendTell.py#L139)   
- The pretrained-model's top/classification layer is removed, so that the model's output are feature vectors corresponding to an encoded representation of the input image. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/models/ShowAttendTell.py#L30)  

Next, The encoded images are passed to a soft attention model and an RNN decoder (an LSTM is used as in the paper).  

## Attention
The **soft attention model** receives the encoded images and the decoder's previous hidden state and computes attention scores. The attention scores are applied like weights to the encoded images (creating the 'context vector'), effectively telling the decoder where to look for useful information as it predicts the next word in the caption.    
Implementation Details:  
- Hidden states are initialized with the mean of the encoded image. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/models/ShowAttendTell.py#L139)    
- In section 4.2.1 of Xu et al., they recommend applying a gate to the context vector to further improve the model's ability to focus on important image regions. This can be achieved using a sigmoid layer which is fed the previous hidden state. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/models/ShowAttendTell.py#L189)

## Decoder
Finally, the context vector is passed to the RNN **decoder** along with caption word embeddings and with the previous hidden state (and cell state since using LSTM).  
The word embeddings are concatenated with the context vector.  
The RNN produces an updated hidden state (and cell state), which is passed through a fully connected linear layer to produce V (per image) logits/scores where V is the size of the vocabulary (ie, the distribution of 'log odds' over the next word in the caption).  

## Training:    
- In training, for any given batch the inputs are sorted by the word length of their captions, in decreasing order. This is so we can efficiently avoid decoding 'pad' tokens, which we only add to the captions so that we can stack them in a single tensor. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/models/ShowAttendTell.py#L159)  
- We also track 'decode_length' for each image which is the number of non-pad/non-stop characters - this is the number of decoding steps required for the given image.
- Thus for each timestep in the decoding, we determine which images need to be decoded (their decode_length is less than the current time step) and use this as our 'effective batch' to pass through the decoder.  
- By keeping the images & captions sorted in descending order, we can continuously track logits and attention scores for each timestep (if a given image does not need decoding at the current timestep, its value in the prediction tensor remains 0...)  
- To compute loss, we compare the logits tensor (which contains predicted log odds over words for each step in a given images deocding) to the target word tokens. First we use pytorch's 'pack_padded_sequences' to remove the timesteps from the logit and target tensors that were not actually decoded (ie, where pad tokens were). [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/train.py#L277)  
- We also add "**doubly stochastic attention regularization**" which is addressed in section 4.2.1 of the paper. This regularization method penalizes the alphas so that they tend to sum towards 1 *across* the time-steps. Intuitively, this should help the model to pay equal attention to every part of the image over the course of decoding.  
- **BLEU Score** - the BiLingual Evaluation Understudy score is a metric used in machine translation, that is also applied in other NLP domains. The metric measures the similarity between "hypotheses" (predicted captions in this case) and "references" (true/human captions) based on N-grams. The score is between 0 and 1, with 1 being the best score (but unlikely to be obtained). [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/train.py#L377)  

## Inference:
- In general, we can generate new captions simply by passing an image through the trained encoder, and then passing its encoding through the decoder to produce a sequence of words.  

Beam Search.
- However, consider that each step of the decoder effectively predicts a probability distribution over the next word in the caption. We could use the highest scoring word as the input for the decoder's next step, but what if the second or third word would lead to an overall better caption?
- Using the beam search algorithm, we can simultaneously track the decoder's top K predictions and generate K parallel captions, before settling on the best-scoring combination of words. [(Code.)](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/utils/inference.py#L9)   
- This helps prevent ours captions from being too dependent on the first predicted word.  

Visualizing Attention
- One interesting aspect of the attention model is that, for an example image, we can visualize each of the attention scores for the decoder's time-steps and match them up with the input image to visualize what part of the image the model is attending to while making a word prediction.  
- See [post-train.ipynb](https://github.com/hans-elliott99/DLReplications/blob/main/dlrep/ImageCaptioning/post-train.ipynb) for examples.


     
