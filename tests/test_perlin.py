import numpy as np 
import torch

from electricmayhem._perlin import (perlin, 
                                    _get_patch_outer_box_from_mask,
                                    BayesianPerlinNoisePatchTrainer)
from electricmayhem import _augment, mask


def detect_func(x, return_raw=False):
    output = np.random.choice([-1,0,1])
    if return_raw:
        return output, "foobar"
    else:
        return output
    
    
def eval_func(writer, step, img, **kwargs):
    assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)
    assert isinstance(step, int)
    assert isinstance(img, torch.Tensor)
    
    

num_augs = 10
augs = [_augment.generate_aug_params() 
        for _ in range(num_augs)]




def test_perlin():
    H = 237
    W = 119
    noise = perlin(H, W, 0.5, 0.5, 2, 0.5, 2)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (1,H,W)
    
    
    
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
    
    
    
def test_BayesianPerlinNoisePatchTrainer(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BayesianPerlinNoisePatchTrainer(img, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2)
    trainer.fit(epochs=1)
    
    
def test_BayesianPerlinNoisePatchTrainer_with_lacunarity(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BayesianPerlinNoisePatchTrainer(img, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2,
                                   tune_lacunarity=True)
    trainer.fit(epochs=1)
    
def test_BayesianPerlinNoisePatchTrainer_with_GPKG(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BayesianPerlinNoisePatchTrainer(img, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2,
                                   gpkg=True)
    trainer.fit(epochs=1)
    