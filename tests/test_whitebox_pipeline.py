import numpy as np
import torch
import json
import pytest

from electricmayhem.whitebox import _pipeline, _aug, _create

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
    

    
def test_pipeline_initialize_patch_params_with_size():
    shape = (1,3,5)
    pipeline = _pipeline.Pipeline()
    pipeline.initialize_patch_params(patch_shape=shape)
    assert isinstance(pipeline.patch_params, torch.Tensor)
    assert pipeline.patch_params.shape == shape
    
    
def test_pipeline_initialize_patch_params_with_pre_initialized_patch():
    shape = (1,3,5)
    startpatch = torch.zeros(shape)
    pipeline = _pipeline.Pipeline()
    pipeline.initialize_patch_params(patch = startpatch)
    assert isinstance(pipeline.patch_params, torch.Tensor)
    assert pipeline.patch_params.shape == shape
    assert pipeline.patch_params.numpy().max() == 0
    
    
def test_pipeline_set_loss():
    def myloss(outputs, patchparam):
        outputs = outputs.reshape(outputs.shape[0],-1)
        outdict = {}
        trueclass = 3
        labels = trueclass*torch.ones(outputs.shape[0], dtype=torch.long)
        outdict["crossent"] = -1* torch.nn.CrossEntropyLoss(reduce=False)(outputs, labels)
        outdict["acc"] = (torch.argmax(outputs,1) == labels).type(torch.float32)
        return outdict     

    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.set_loss(myloss)
    
    
def test_pipeline_set_loss_throws_error_for_bad_outputs():
    def myloss(outputs, patchparam):
        
        return outputs
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    with pytest.raises(AssertionError) as err: 
        pipeline.set_loss(myloss)
        
    assert "dictionary" in str(err.value)
    
    
def test_pipeline_set_loss_throws_error_for_bad_output_shapes():
    def myloss(outputs, patchparam):
        return {"foo":torch.zeros((5,7,2))}
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    with pytest.raises(AssertionError) as err: 
        pipeline.set_loss(myloss)
        
    assert "correct shape" in str(err.value)
    