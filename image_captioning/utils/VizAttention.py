import torch

import cv2
import skimage.transform

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from data.LoadData import String2Int


def plot(image, label, alphas, string2int:String2Int, smooth=True, im_size=14):
    """
    Adapted from: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image.permute(1,2,0).numpy()
    image = cv2.resize(image, (im_size*24, im_size*24), interpolation=cv2.INTER_LANCZOS4)

    caption = [string2int.itos[i.item()] for i in label]

    # for t in range(len(caption)):
    while True:
        for t in range(len(caption)):
            if caption[t] == string2int.pad_token:
                return plt.show()

            plt.subplot(int(np.ceil(len(caption) / 5.0)), 5, t+1)

            plt.text(x=0, y=1, s=caption[t], color='black', backgroundcolor='white', fontsize=12)
            plt.imshow((image*255).astype('uint8'))

            
            if smooth:
                alpha = skimage.transform.pyramid_expand(alphas[t, :].numpy().reshape(14,14), upscale=24, sigma=8)
            else:
                alpha = skimage.transform.resize(alphas[t, :].numpy(), [im_size*24, im_size*24])
            
            if t==0:
                plt.imshow(alpha, alpha=0.0) ##show none of the alpha for first image
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')