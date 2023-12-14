import numpy as np
import torch
import json
import pytest
import os

from electricmayhem.whitebox import _pipeline, _aug, _create

augdict1 = {"ColorJiggle":{"contrast":0.2, "p":1}}
augdict2 = {"ColorJiggle":{"contrast":0.1, "p":1}}

aug1 = _aug.KorniaAugmentationPipeline(augdict1)
aug2 = _aug.KorniaAugmentationPipeline(augdict2)

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
    pipe = _pipeline.Pipeline(aug1, aug2)
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt = pipe(inpt)
    assert inpt.shape == outpt.shape
    
def test_pipeline_dunderscore_creation():    
    pipe = aug1 + aug2
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt = pipe(inpt)
    assert inpt.shape == outpt.shape
    
def test_pipeline_len():    
    pipe = aug1 + aug2
    assert len(pipe) == 2
    

def test_pipeline_getitem():    
    pipe = aug1 + aug2
    assert isinstance(pipe[1], _aug.KorniaAugmentationPipeline)
    
def test_pipeline_creation_with_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 3, 5)

        def forward(self, x):
            return self.conv1(x)
    mod = Model().eval()
    
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
    
    
    
  
def test_pipeline_training_loop_runs():
    """
    This is a pretty minimal test just to see
    if it runs without crashing for a trivial case
    """
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1)
    assert out.shape == shape
    
    
def test_pipeline_training_loop_runs_mifgsm_optimizer():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)

    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1,
                         optimizer="mifgsm")
    assert out.shape == shape
    
def test_pipeline_training_loop_runs_adam_optimizer():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1,
                         optimizer="adam")
    assert out.shape == shape
    
def test_pipeline_training_loop_runs_lr_decay_exponential():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1,
                         lr_decay="exponential")
    assert out.shape == shape
    
def test_pipeline_training_loop_runs_no_lr_decay():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1,
                         lr_decay="none")
    assert out.shape == shape
    
def test_pipeline_training_loop_runs_progress_bar_disabled():
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, patchparam):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1,
                         progressbar=False)
    assert out.shape == shape
    
    
def test_pipeline_get_last_sample_as_dict():
    
    pipe = aug1 + aug2
    batch_size = 7
    inpt = torch.tensor(np.random.uniform(0, 1, (batch_size,3,32,32)).astype(np.float32))
    # test apply
    outpt = pipe(inpt)
    lastsample = pipe.get_last_sample_as_dict()
    assert isinstance(lastsample, dict)
    keys = list(lastsample.keys())
    assert len(lastsample[keys[0]]) == batch_size
    
    
def test_pipeline_save_yaml(tmp_path_factory):
    fn = str(tmp_path_factory.mktemp("logs"))
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":1}}
    
    stack = _create.PatchStacker()
    aug = _aug.KorniaAugmentationPipeline(augdict1)
    
    pipe = stack + aug
    pipe.set_logging(logdir=fn)
    pipe.save_yaml()
    # now let's open it back up
    ymltxt = open(os.path.join(pipe.logdir,"config.yml")).read()
    assert "ColorJiggle" in ymltxt
    
    
def test_pipeline_get_profiler(tmp_path_factory):
    fn = str(tmp_path_factory.mktemp("logs"))
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":1}}
    
    stack = _create.PatchStacker()
    aug = _aug.KorniaAugmentationPipeline(augdict1)
    
    pipe = stack + aug
    pipe.set_logging(logdir=fn)
    prof = pipe._get_profiler()
    assert isinstance(prof, torch.profiler.profiler.profile)