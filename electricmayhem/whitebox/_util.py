import numpy as np
from PIL import Image
import torch


def _img_to_tensor(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    
    x = torch.tensor(x.astype(np.float32)/255)[:,:,:3].permute(2,0,1)
    return x