import numpy as np
import torch
import json

from electricmayhem.whitebox import _aug



def test_korniaaugmentationpipeline_call():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)))
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    y = aug(testim)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testim.shape
    assert not (y.numpy() == testim.numpy()).all()

    
    
    
def test_korniaaugmentationpipeline_reproducibility():
    augdict = {"ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1},
          "RandomAffine":{"scale":[0.1,1.5], "degrees":45, "p":1}}
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    repro = aug.check_reproducibility()
    assert repro == 0
    
def test_korniaaugmentationpipeline_get_last_sample_as_dict():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    batch_size = 7
    testbatch = torch.tensor(np.random.uniform(0, 1, (batch_size,3, 29, 37)))
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    y = aug(testbatch)
    
    sampdict = aug.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    for k in sampdict:
        assert len(sampdict[k]) == batch_size
    
    