import numpy as np
import torch

from electricmayhem.whitebox import _pipeline

def test_pipelinebase_apply():
    base = _pipeline.PipelineBase(a="foo", b="bar")
    testinput = torch.zeros((3,13,17))
    y = base(testinput)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testinput.shape
    
def test_pipelinebase_to_yaml():
    base = _pipeline.PipelineBase(a="foo", b="bar")
    yml = base.to_yaml()
    assert isinstance(yml, str)
    assert "a: foo" in yml
    assert "b: bar" in yml

def test_korniaaugmentationpipeline_call():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (3, 29, 37)))
    
    aug = _pipeline.KorniaAugmentationPipeline(augdict)
    y = aug(testim)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testim.shape
    assert not (y.numpy() == testim.numpy()).all()
    
def test_korniaaugmentationpipeline_apply():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1, 3, 29, 37)))
    
    aug = _pipeline.KorniaAugmentationPipeline(augdict)
    y = aug.apply(testim)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testim.shape
    assert not (y.numpy() == testim.numpy()).all()
    