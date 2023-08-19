import numpy as np
from PIL import Image
import torch


def _img_to_tensor(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    if len(x.shape) == 2:
        x = np.stack([x,x,x], -1)
    
    x = torch.tensor(x.astype(np.float32)/255)[:,:,:3].permute(2,0,1)
    return x

def _dict_of_tensors_to_dict_of_arrays(d):
    """
    Turn a dictionary of tensors into a dictionary of arrays.
    """
    outdict = {}
    for k in d:
        if isinstance(d[k], torch.Tensor):
            outdict[k] = d[k].cpu().detach().numpy()
        else:
            outdict[k] = d[k]
    return outdict

def _concat_dicts_of_arrays(*d):
    """
    Combine multiple dicts of arrays into one dict
    of concatenated arrays. Assumes all dicts have the
    same keys, they're all numpy arrays, and they all
    have shapes we can concatenate along axis 0.
    """
    outdict = {}
    for k in d[0]:
        outdict[k] = np.concatenate([x[k] for x in d], 0)
    return outdict