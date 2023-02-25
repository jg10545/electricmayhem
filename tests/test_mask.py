# -*- coding: utf-8 -*-
import numpy as np
import torch

from electricmayhem._mask import *


def test_generate_rectangular_frame_mask():
    # generate init/final masks
    W = 101
    H = 107
    box = [31, 42, 41, 52]
    fw = 10
    num_channels = 3
    
    init_mask, final_mask = generate_rectangular_frame_mask(W, H, *box, frame_width=fw, num_channels=num_channels)
    # outer area is product of box dimensions plus the mask width
    # on either side
    outer_area = (10+2*fw)*(10+2*fw)
    # inner area is just the box dimensions
    inner_area = 10*10
    # initial mask should include the whole area
    assert init_mask.sum() == outer_area * num_channels
    # inner mask should include the outer area but with inner cut out
    assert final_mask.sum() == (outer_area - inner_area)*num_channels
    
    

def test_generate_rectangular_frame_mask_return_torch():
    # generate init/final masks
    W = 101
    H = 107
    box = [31, 42, 41, 52]
    fw = 10
    num_channels = 3
    
    init_mask, final_mask = generate_rectangular_frame_mask(W, H, *box, frame_width=fw, num_channels=num_channels, return_torch=True)
    
    assert isinstance(init_mask, torch.Tensor)
    assert isinstance(final_mask, torch.Tensor)
    
    # outer area is product of box dimensions plus the mask width
    # on either side
    outer_area = (10+2*fw)*(10+2*fw)
    # inner area is just the box dimensions
    inner_area = 10*10
    # initial mask should include the whole area
    assert init_mask.numpy().sum() == outer_area * num_channels
    # inner mask should include the outer area but with inner cut out
    assert final_mask.numpy().sum() == (outer_area - inner_area)*num_channels


def test_generate_priority_mask():
    # generate init/final masks
    W = 101
    H = 107
    box = [31, 42, 41, 52]
    fw = 10
    num_channels = 3
    
    init_mask, final_mask = generate_rectangular_frame_mask(W, H, *box, frame_width=fw, num_channels=num_channels, return_torch=True)
    
    priority_mask = generate_priority_mask(init_mask, final_mask)
    assert priority_mask.numpy().sum() < init_mask.numpy().sum()
    assert priority_mask.numpy().sum() > final_mask.numpy().sum()