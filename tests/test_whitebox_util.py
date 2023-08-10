import numpy as np
import torch
from PIL import Image


from electricmayhem.whitebox import _util



def test_img_to_tensor_from_PIL_image():
    H,W,C = 17, 23, 4
    x = Image.fromarray(np.random.randint(0,255, size=(H,W,C)).astype(np.uint8))
    y = _util._img_to_tensor(x)
    
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,H,W)
    
def test_img_to_tensor_from_numpy_array():
    H,W,C = 17, 23, 4
    x = np.random.randint(0,255, size=(H,W,C)).astype(np.uint8)
    y = _util._img_to_tensor(x)
    
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,H,W)