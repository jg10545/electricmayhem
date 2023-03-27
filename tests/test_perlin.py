import numpy as np
import torch

from electricmayhem._perlin import perlin, _get_patch_outer_box_from_mask


def test_perlin():
    H = 237
    W = 119
    noise = perlin(H, W, 0.5, 0.5, 2, 0.5, 2)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (H,W,1)
    
    
    
def test_get_patch_outer_box_from_mask():
    H = 237
    W = 119
    C = 3
    left = 31
    top = 73
    x = 29
    y = 91
    
    mask = np.zeros((C,H,W))
    nonzeropart = np.random.choice([0,1], size=(C,y,x))
    mask[:,top:top+y,left:left+x] += nonzeropart
    mask = torch.Tensor(mask)
    
    box = _get_patch_outer_box_from_mask(mask)
    assert isinstance(box, dict)
    assert box["top"] == top
    assert box["left"] == left
    assert box["height"] == y
    assert box["width"] == x