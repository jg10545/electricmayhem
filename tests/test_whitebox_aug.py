import numpy as np
import torch
import json

from electricmayhem.whitebox import _aug



def test_korniaaugmentationpipeline_call():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)))
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    y, _ = aug(testim)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testim.shape
    assert not (y.numpy() == testim.numpy()).all()

    

def test_korniaaugmentationpipeline_call_on_dictionary():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)))
    key = "mypatch"
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    y, _ = aug({key:testim})
    assert isinstance(y[key], torch.Tensor)
    assert y[key].shape == testim.shape
    assert not (y[key].numpy() == testim.numpy()).all()



def test_korniaaugmentationpipeline_call_apply_on_train_False():
    augdict = {"ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)).astype(np.float32))
    
    aug = _aug.KorniaAugmentationPipeline(augdict, apply_on_train=False)
    # shouldn't apply on train
    y, _ = aug(testim)
    assert (y.numpy() == testim.numpy()).all()
    # but should on eval
    y, _ = aug(testim, evaluate=True)
    assert not (y.numpy() == testim.numpy()).all()



def test_korniaaugmentationpipeline_call_apply_on_eval_False():
    augdict = {"ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)).astype(np.float32))
    
    aug = _aug.KorniaAugmentationPipeline(augdict, apply_on_eval=False)
    # shouldn't apply on train
    y, _ = aug(testim, evaluate=True)
    assert (y.numpy() == testim.numpy()).all()
    # but should on eval
    y, _ = aug(testim)
    assert not (y.numpy() == testim.numpy()).all()


    
def test_korniaaugmentationpipeline_reproducibility():
    augdict = {"ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1},
          "RandomAffine":{"scale":[0.1,1.5], "degrees":45, "p":1}}
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    repro = aug.check_reproducibility()
    assert repro == 0


def test_korniaaugmentationpipeline_validate():
    augdict = {"ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1},
          "RandomAffine":{"scale":[0.1,1.5], "degrees":45, "p":1}}
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    assert aug.validate(torch.tensor(np.random.uniform(0, 1, (1, 3, 21, 37)).astype(np.float32)))
    

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
    
    