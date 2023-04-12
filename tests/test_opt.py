import numpy as np 
import torch
import torch.utils.tensorboard
import dask

from electricmayhem import _augment, mask
from electricmayhem._opt import BlackBoxOptimizer

dask.config.set(scheduler='threads')

dask.config.set(scheduler='threads')

"""

Some utilities:
    
    detect_func(): dummy function pretending to check whether an image
       is correctly detected, missed, or throws an error
    
    augs: list of augmentation parameters


"""



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



def test_BlackBoxPatchOptimizer_with_gray_perturbation(tmp_path_factory):
    # SAVE IT TO LOG DIR
    logdir = str(tmp_path_factory.mktemp("logs"))
    
    H = 51
    W = 57
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    pert = torch.Tensor(np.random.uniform(0, 1, size=(1,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                        20, 30, 30,
                                        frame_width=5, 
                                        return_torch=True)
    
    opt = BlackBoxOptimizer(img, init_mask, final_mask, 
                            detect_func, logdir,
                            budget=200, 
                            num_augments=[2,5],
                            q=[2,5],
                            beta=[0.5,1.5],
                            downsample=[5,10],
                            eval_augments=5,
                            num_channels=1,
                            eval_func=eval_func)
    opt.fit(2)