import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from electricmayhem._augment import augment_image, compose

def load_to_tensor(i):
    """
    Input a path to an image (or a PIL.Image object) and convert
    to a normalized tensor in channel-first format
    
    Similar to the pil_to_tensor in torchvision, but does normalization
    and strips alpha channel if there is one
    """
    if isinstance(i, str):
        i = Image.open(i)
        
    # convert to a normalized float numpy array
    i = np.array(i).astype(np.float32)/255
    # convert to tensor in channel-first format
    t = torch.Tensor(i).permute(2,0,1)
    # return first three channels at most
    return t[:3,:,:]


def plot(i, augs=None, detect_func=None, mask=None, perturbation=None):
    """
    Matplotlib macro for plotting a channel-first image tensor
    
    :i: torch.Tensor containing normalized image in channel-first format
    :augs: if a list of augmentation parameters is passed, will sample 9
        and return a grid plot of augmented images
    :detect_func: if a detection function is passed, output will be
        used as an image title
    """
    if augs is not None:
        chosen_augs = np.random.choice(augs, size=9)
        for e, a in enumerate(chosen_augs):
            plt.subplot(3,3,e+1)
            plot(augment_image(i, **a, mask=mask, perturbation=perturbation),
                 detect_func=detect_func)
    else:
        if (perturbation is not None)&(mask is not None):
            i = compose(i, mask, perturbation)
        plt.imshow(i.permute(1,2,0))
        plt.axis(False)
        if detect_func is not None:
            plt.title(detect_func(i))
