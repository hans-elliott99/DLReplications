# DLReplications
Replications & offshoots of deep learning papers

`image_captioning/`  
- `models/`  
    - `ShowAttendTell` - Xu et al. (2016) use a deep ConvNet to encode images and use a recurrent net to decode them into a sequence of text. An attention model uses the image encoding and the recurrent net's hidden state to produce attention scores, which 'tell' the decoder which part of the image encoding it should attend to during each timestep. 
