# -*- coding: utf-8 -*-

import numpy as np
import torch

from electricmayhem._augment import *



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
    
    
    
def test_wiggle_mask_and_perturbations():
    shape = (3,31,43)
    pert = torch.Tensor(np.random.uniform(0, 1, shape))
    mask = torch.Tensor(np.random.choice([0,1], size=shape))
    
    shifted_mask, shifted_pert = wiggle_mask_and_perturbation(mask, pert, 1, 5)
    assert isinstance(shifted_mask, torch.Tensor)
    assert isinstance(shifted_pert, torch.Tensor)
    assert pert.shape == shifted_pert.shape
    assert mask.shape == shifted_mask.shape
    
    

def test_compose():
    shape = (3,31,43)
    img = torch.Tensor(np.random.uniform(0, 1, shape))
    pert = torch.Tensor(np.random.uniform(0, 1, shape))
    mask = torch.Tensor(np.random.choice([0,1], size=shape))
    
    img_w_pert = compose(img, mask, pert)
    assert isinstance(img_w_pert, torch.Tensor)
    assert img_w_pert.shape == shape