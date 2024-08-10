

import numpy as np
import torch

from electricmayhem.blackbox._augment import *



def test_generate_aug_params():
    params = generate_aug_params()
    assert isinstance(params, dict)
    for k in ["warp", "gamma", "scale", "blur"]:
        assert k in params
        
        
        
def test_augment_image():
    shape = (3,31,43)
    img = torch.Tensor(np.random.uniform(0, 1, shape))
    params = generate_aug_params()
    
    augmented = augment_image(img, **params)
    assert isinstance(augmented, torch.Tensor)
    

def test_compose():
    shape = (3,31,43)
    img = torch.Tensor(np.random.uniform(0, 1, shape))
    pert = torch.Tensor(np.random.uniform(0, 1, shape))
    mask = torch.Tensor(np.random.choice([0,1], size=shape))
    
    img_w_pert = compose(img, mask, pert)
    assert isinstance(img_w_pert, torch.Tensor)
    assert img_w_pert.shape == shape