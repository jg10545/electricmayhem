import numpy as np
import torch
import torch.utils.tensorboard
import dask

from electricmayhem import _augment, mask
from electricmayhem._graphite import *

dask.config.set(scheduler='threads')

"""

Some utilities:
    
    detect_func(): dummy function pretending to check whether an image
       is correctly detected, missed, or throws an error
    
    augs: list of augmentation parameters


"""



def detect_func(x):
    return np.random.choice([-1,0,1])

num_augs = 10
augs = [_augment.generate_aug_params() 
        for _ in range(num_augs)]



def test_estimate_transform_robustness():
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    
    results = estimate_transform_robustness(detect_func, augs, img)
    assert isinstance(results, dict)
    for c in ["crash_frac", "detect_frac", "tr"]:
        assert c in results
        assert results[c] <= 1
        assert results[c] >= 0
        
    assert "sem" in results
        
    
def test_estimate_transform_robustness_with_error_as_positive():
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    
    results = estimate_transform_robustness(detect_func, augs, img,
                                            include_error_as_positive=True)
    assert isinstance(results, dict)
    for c in ["crash_frac", "detect_frac", "tr"]:
        assert c in results
        assert results[c] <= 1
        assert results[c] >= 0
        
    assert "sem" in results
        
def test_estimate_transform_robustness_return_outcomes():
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    
    results, outcomes = estimate_transform_robustness(detect_func, augs, img, 
                                            return_outcomes=True)
    assert isinstance(results, dict)
    for c in ["crash_frac", "detect_frac", "tr"]:
        assert c in results
        assert results[c] <= 1
        assert results[c] >= 0
        
    assert isinstance(outcomes, tuple)
    assert len(outcomes) == len(augs)
    
    
def test_reduce_mask():
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20, 20, 30, 30,
                                          frame_width=5, return_torch=True)
    priority_mask = mask.generate_priority_mask(init_mask, final_mask)
    
    n = 5
    a, results = reduce_mask(img, priority_mask, pert, detect_func, augs,
                             n=n)

    assert a >= 0
    assert a <= 1
    assert isinstance(results, list)
    assert len(results) == n
    
    
def test_estimate_gradient():
    H = 101
    W = 107
    C = 3
    tr_estimate = 0.5
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20, 20, 30, 30,
                                          frame_width=5, return_torch=True)
    grad = estimate_gradient(img, final_mask, pert, augs, 
                             detect_func, tr_estimate)
    

    assert isinstance(grad, torch.Tensor)
    assert grad.shape == img.shape
    
    
def test_update_perturbation():
    H = 101
    W = 107
    C = 3
    tr_estimate = 0.5
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    grad = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20, 20, 30, 30,
                                          frame_width=5, return_torch=True)
    
    newpert, lr = update_perturbation(img, final_mask, pert, augs,
                                      detect_func, grad)
    
    assert isinstance(newpert, torch.Tensor)
    assert newpert.shape == img.shape
    assert newpert.numpy().max() <= 1
    assert newpert.numpy().min() >= 0
    assert isinstance(lr, dict)
    assert 'lr' in lr
    
    
    
def test_BlackBoxPatchTrainer(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    tr_estimate = 0.5
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BlackBoxPatchTrainer(img, init_mask, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2, 
                                   q=5,
                                   reduce_steps=2)
    trainer.fit(epochs=1)
    
    
  
def test_BlackBoxPatchTrainer_using_query_budget(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    tr_estimate = 0.5
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BlackBoxPatchTrainer(img, init_mask, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2, 
                                   q=5,
                                   reduce_steps=2)
    trainer.fit(budget=10)
    
    
def test_BlackBoxPatchTrainer_with_fixed_augs(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    augs = [_augment.generate_aug_params() for _ in range(100)]
    
    H = 101
    W = 107
    C = 3
    tr_estimate = 0.5
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BlackBoxPatchTrainer(img, init_mask, 
                                   final_mask, detect_func, logdir,
                                   num_augments=2, 
                                   q=5,
                                   reduce_steps=2,
                                   fixed_augs=augs)
    trainer.fit(budget=10)
    
    
    
def test_BlackBoxPatchTrainer_with_different_pert_size(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(C,int(H/2),int(W/3))))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BlackBoxPatchTrainer(img, init_mask, 
                                   final_mask, detect_func, logdir,
                                   perturbation=pert,
                                   num_augments=2, 
                                   q=5,
                                   reduce_steps=2)
    trainer.fit(epochs=1)
    
    

    
def test_BlackBoxPatchTrainer_with_gray_perturbation(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(1,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    trainer = BlackBoxPatchTrainer(img, init_mask, 
                                   final_mask, detect_func, logdir,
                                   perturbation=pert,
                                   num_augments=2, 
                                   q=5,
                                   reduce_steps=2)
    trainer.fit(epochs=1)
    
    