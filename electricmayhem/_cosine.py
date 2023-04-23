import numpy as np
import scipy.fft
from PIL import Image

from electricmayhem._perlin import normalize, _get_patch_outer_box_from_mask, BayesianPerlinNoisePatchTrainer


def _inverse_cosine_transform(z, latent_shape=None, patch_shape=None):
    """
    :z:
    :latent_shape: tuple (H',W'); shape to resize z to before taking IDCT
    :patch_shape: tuple (H,W); shape to resize transformed patch to
    """
    if latent_shape is not None:
        z = z.reshape(latent_shape)
        
    x = scipy.fft.idctn(z)
    
    if patch_shape is not None:
        H, W = patch_shape
        x = np.array(Image.fromarray(x).resize((W,H)))
        
    x = np.expand_dims(x, 0)
    return normalize(x)

