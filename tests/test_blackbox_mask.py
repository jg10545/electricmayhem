import numpy as np
import torch

from electricmayhem.blackbox.mask import *


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
    
def test_random_subset_mask():
    H = 107
    W = 101
    C = 3
    mask = np.zeros((C,H,W))
    mask[:,13:56,68:90] = 1
    mask = torch.tensor(mask)
    #mask = torch.tensor(np.random.choice([0,1], size=(C,H,W)))
    newmask = random_subset_mask(mask, 0.5)
    # did it return the right data type?
    assert isinstance(newmask, torch.Tensor)
    # did it return the right shape?
    assert newmask.shape == mask.shape
    # does the patch have nonzero elements
    assert newmask.numpy().sum() > 0
    # does the new patch overlap the old one?
    assert (newmask*mask).numpy().sum() > 0
    # does the new patch unmask anything it shouldn't?
    assert (newmask*(1-mask)).numpy().sum() == 0