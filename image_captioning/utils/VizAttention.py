

import torch

import cv2
import skimage.transform

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from data.LoadData import String2Int

def plot_training(image, label, alphas, string2int:String2Int, smooth=True, im_size=14):
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
                alpha = skimage.transform.pyramid_expand(alphas[t, :].numpy().reshape(im_size,im_size), upscale=24, sigma=8)
            else:
                alpha = skimage.transform.resize(alphas[t, :].numpy(), [im_size*24, im_size*24])
            
            if t==0:
                plt.imshow(alpha, alpha=0.0) ##show none of the alpha for first image
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')


def plot_predicted(image, pred_seq, alphas, string2int, smooth=True, im_size=14):
    """Visualize attention of model when predicting a new caption."""

    if torch.is_tensor(image):
        image = image.cpu()
        if len(image.size()) > 3:
            assert image.size(0)==1, "only provide one image"
            image = image.squeeze(0)

        if torch.max(image) < 255:
            image = (image*255).long()

        image = image.permute(1, 2, 0).numpy().astype('uint8')
        image = cv2.resize(image, (im_size*24, im_size*24), interpolation=cv2.INTER_LANCZOS4)
    else:
        image = np.array(image) ##convert from PIL to np, or if np do nothing

    for t in range(len(pred_seq)):
        plt.subplot(int(np.ceil(len(pred_seq) / 5.0)), 5, t+1)
        plt.text(x=0, y=1, s=string2int(pred_seq[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        if t > 0:
            if smooth:
                alpha = skimage.transform.pyramid_expand(alphas[t-1].cpu().numpy().reshape(im_size,im_size), upscale=24, sigma=8)
            else:
                alpha = skimage.transform.resize(alphas[t-1].cpu().numpy(), [im_size*24, im_size*24])
            opacity=0.7
        else:
            alpha = np.ones((im_size*24, im_size*24, 1))
            opacity=0.0
        plt.imshow(alpha, alpha=opacity)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

