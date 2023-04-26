import numpy as np
import torch

from electricmayhem._cosine import _inverse_cosine_transform, BayesianCosinePatchTrainer
from electricmayhem import mask, _augment


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



def test_inverse_cosine_transform():
    Hprime, Wprime = 13, 15
    latent_shape = (3,5)
    z = np.random.normal(0, 1, size=latent_shape[0]*latent_shape[1])
    x = _inverse_cosine_transform(z, latent_shape, (Hprime, Wprime))
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (1,Hprime, Wprime)
    
    

def test_BayesianCosinePatchTrainer(tmp_path_factory):
    logdir = str(tmp_path_factory.mktemp("logs"))
        
    H = 101
    W = 107
    C = 3
    img = torch.Tensor(np.random.uniform(0, 1, size=(C,H,W)))
    init_mask, final_mask = mask.generate_rectangular_frame_mask(W, H, 20,
                                            20, 30, 30,
                                            frame_width=5, 
                                            return_torch=True)
        
    trainer = BayesianCosinePatchTrainer(img, 
                                       final_mask, detect_func, logdir,
                                       num_augments=2)
    trainer.fit(epochs=1)
        