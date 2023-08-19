import numpy as np
import torch
import json

from electricmayhem.whitebox import _pipeline, _aug

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

    
def test_modelwrapper():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 3, 5)

        def forward(self, x):
            return self.conv1(x)
        
    mod = Model().eval()
    wrapper = _pipeline.ModelWrapper(mod)
    assert isinstance(wrapper.to_yaml(), str)
    
    
def test_pipeline_manual_creation():
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":0.25}}
    augdict2 = {"ColorJiggle":{"contrast":0.1, "p":0.25}}
    
    aug1 = _aug.KorniaAugmentationPipeline(augdict1)
    aug2 = _aug.KorniaAugmentationPipeline(augdict2)
    
    pipe = _pipeline.Pipeline(aug1, aug2)
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt = pipe(inpt)
    assert inpt.shape == outpt.shape
    
def test_pipeline_dunderscore_creation():
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":0.25}}
    augdict2 = {"ColorJiggle":{"contrast":0.1, "p":0.25}}
    
    aug1 = _aug.KorniaAugmentationPipeline(augdict1)
    aug2 = _aug.KorniaAugmentationPipeline(augdict2)
    
    pipe = aug1 + aug2
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt = pipe(inpt)
    assert inpt.shape == outpt.shape
    
def test_pipeline_creation_with_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 3, 5)

        def forward(self, x):
            return self.conv1(x)
    mod = Model().eval()
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":0.25}}
    augdict2 = {"ColorJiggle":{"contrast":0.1, "p":0.25}}
    
    aug1 = _aug.KorniaAugmentationPipeline(augdict1)
    aug2 = _aug.KorniaAugmentationPipeline(augdict2)
    
    pipe = aug1 + aug2 + mod
    assert len(pipe.steps) == 3
    assert isinstance(pipe.to_yaml(), str)
    
    

def test_korniaaugmentationpipeline_get_last_sample_as_dict():
    augdict = {"RandomPlasmaShadow":{"roughness":(0.4,0.5), "p":1.},
          "ColorJiggle":{"contrast":0.2, "hue":0.2, "p":1.}}
    testim = torch.tensor(np.random.uniform(0, 1, (1,3, 29, 37)))
    
    aug = _aug.KorniaAugmentationPipeline(augdict)
    y = aug(testim)
    sampdict = aug.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)

    