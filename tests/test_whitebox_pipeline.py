import numpy as np
import torch
import json
import pytest
import os

from electricmayhem.whitebox import _pipeline, _aug, _create
from .modelgenerators import *

augdict1 = {"ColorJiggle":{"contrast":0.2, "p":1}}
augdict2 = {"ColorJiggle":{"contrast":0.1, "p":1}}

aug1 = _aug.KorniaAugmentationPipeline(augdict1)
aug2 = _aug.KorniaAugmentationPipeline(augdict2)

def test_pipelinebase_apply():
    base = _pipeline.PipelineBase(a="foo", b="bar")
    testinput = torch.zeros((3,13,17))
    y, kwargs = base(testinput)
    assert isinstance(y, torch.Tensor)
    assert y.shape == testinput.shape
    assert isinstance(kwargs, dict)
    
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
    assert isinstance(wrapper.get_last_sample_as_dict(), dict)
    
    
def test_modelwrapper_multiple_models():
    train_model = DummyConvNet(3)
    eval_model = DummyConvNet(5)
    
    wrapper = _pipeline.ModelWrapper(train_model, eval_model=eval_model)
    x = torch.tensor(np.random.uniform(0, 1, (1,3,28,28)).astype(np.float32))
    
    y_train, _ = wrapper(x)
    y_eval, _ = wrapper(x, evaluate=True)
    
    assert y_train.shape == (1,3)
    assert y_eval.shape == (1,5)
    
   
def test_modelwrapper_multiple_models_in_dictionary():
    train_model = DummyConvNet(3)
    eval_model = DummyConvNet(5)
    
    wrapper = _pipeline.ModelWrapper(train_model, eval_model={"trainmodel":train_model,
                                                              "evalmodel":eval_model})
    x = torch.tensor(np.random.uniform(0, 1, (1,3,28,28)).astype(np.float32))
    
    y_train, _ = wrapper(x)
    y_eval, _ = wrapper(x, evaluate=True)
    
    assert y_train.shape == (1,3)
    assert isinstance(y_eval, dict)
    assert y_eval["trainmodel"].shape == (1,3)
    assert y_eval["evalmodel"].shape == (1,5)
    
def test_pipeline_manual_creation():
    pipe = _pipeline.Pipeline(aug1, aug2)
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt, kwargs = pipe(inpt)
    assert inpt.shape == outpt.shape
    assert isinstance(kwargs, dict)
    
def test_pipeline_dunderscore_creation():    
    pipe = aug1 + aug2
    inpt = torch.tensor(np.random.uniform(0, 1, (1,3,32,32)).astype(np.float32))
    # test apply
    outpt, _ = pipe(inpt)
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
    assert isinstance(pipeline.patch_params.patch, torch.Tensor)
    assert pipeline.patch_params.patch.shape == shape
    
    
def test_pipeline_initialize_patch_params_with_pre_initialized_patch():
    shape = (1,3,5)
    startpatch = torch.zeros(shape)
    pipeline = _pipeline.Pipeline()
    pipeline.initialize_patch_params(patch = startpatch)
    assert isinstance(pipeline.patch_params.patch, torch.Tensor)
    assert pipeline.patch_params.patch.shape == shape
    assert pipeline.patch_params.patch.detach().numpy().max() == 0
    
    
def test_pipeline_set_loss():
    def myloss(outputs, **kwargs):
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
    def myloss(outputs, **kwargs):
        
        return outputs
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    with pytest.raises(AssertionError) as err: 
        pipeline.set_loss(myloss)
        
    assert "dictionary" in str(err.value)
    
    
def test_pipeline_set_loss_throws_error_for_bad_output_shapes():
    def myloss(outputs, **kwargs):
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
    def myloss(outputs, **kwargs):
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
    assert isinstance(pipeline._get_learning_rate(), float)
    assert out.shape == shape
    

  
def test_pipeline_training_loop_with_logging(tmp_path_factory):
    logdir = str(tmp_path_factory.mktemp("pipeline_logs"))
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 4
    num_eval_steps = 1
    def myloss(outputs, **kwargs):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    pipeline.set_logging(logdir)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1)
    pipeline.evaluate(batch_size, num_steps)
    assert out.shape == shape
    
    
def test_pipeline_training_loop_with_profiling(tmp_path_factory):
    logdir = str(tmp_path_factory.mktemp("pipeline_logs_with_profile"))
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 4
    num_eval_steps = 1
    def myloss(outputs,  **kwargs):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    pipeline.set_logging(logdir)
    out = pipeline.train_patch(batch_size, num_steps, step_size, 
                         eval_every, num_eval_steps, mainloss=1, profile=2)
    pipeline.evaluate(batch_size, num_steps)
    assert out.shape == shape    
    
    
def test_pipeline_training_loop_runs_mifgsm_optimizer():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, **kwargs):
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
                         optimizer="mifgsm", lr_decay="cosine")
    assert out.shape == shape
    assert isinstance(pipeline._get_learning_rate(), float)
    
def test_pipeline_training_loop_runs_adam_optimizer():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, **kwargs):
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
                         optimizer="adam", lr_decay="plateau")
    assert out.shape == shape
    
def test_pipeline_training_loop_runs_lr_decay_exponential():
    #This is a pretty minimal test just to see
    #if it runs without crashing for a trivial case
    
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    def myloss(outputs, **kwargs):
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
    def myloss(outputs, **kwargs):
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
    def myloss(outputs, **kwargs):
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
    outpt, _ = pipe(inpt)
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
    
    
    
def test_pipeline_optimize_runs_without_crashing(tmp_path_factory):
    # a low bar but let's try to clear it
    logdir = str(tmp_path_factory.mktemp("pipeline_optimize_logs"))
    batch_size = 2
    num_steps = 5
    num_eval_steps = 1
    N = 6
    def myloss(outputs, **kwargs):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.set_loss(myloss)
    pipeline.optimize("mainloss", logdir, shape, N, num_steps,
                      batch_size,
                      num_eval_steps=num_eval_steps, 
                      lr=(0.001, 0.01, "log"),
                      accumulate=(1, 2, "int"),
                      mainloss=(0.1,1.),
                      lr_decay=["cosine", "exponential"],
                      optimizer=["bim", "adam", "mifgsm"])
    

def test_pipeline_passes_validate():
    augdict1 = {"ColorJiggle":{"contrast":0.2, "p":1}}
    
    stack = _create.PatchStacker()
    aug = _aug.KorniaAugmentationPipeline(augdict1)
    
    pipe = stack + aug
    pipe.initialize_patch_params(patch_shape=(1,5,7))
    assert pipe.validate()


def test_update_patch_gradients():
    batch_size = 2
    accumulate = 1
    def myloss(outputs, **kwargs):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        for key in ["input", "global_step", "evaluate", "control"]:
            assert key in kwargs
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)

    lossdict, loss = _pipeline._update_patch_gradients(pipeline, batch_size, 
                                       {"mainloss":1}, accumulate=accumulate,
                                       rho=0)
    assert isinstance(lossdict, dict)
    assert pipeline.patch_params.patch.grad is not None


def test_update_patch_gradients_with_sharpness_aware_minimization():
    batch_size = 2
    accumulate = 1
    def myloss(outputs, **kwargs):
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        for key in ["input", "global_step", "evaluate", "control"]:
            assert key in kwargs
        return outdict    
    

    shape = (3,5,7)
    step = _create.PatchResizer((11,13))
    pipeline = _pipeline.Pipeline(step)
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)

    lossdict, loss = _pipeline._update_patch_gradients(pipeline, batch_size, 
                                       {"mainloss":1}, accumulate=accumulate,
                                       rho=0.5)
    assert isinstance(lossdict, dict)
    assert pipeline.patch_params.patch.grad is not None