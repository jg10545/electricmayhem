import numpy as np
import torch

from electricmayhem.whitebox.optim import BIM, MIFGSM, _get_optimizer_and_scheduler
from electricmayhem.whitebox.optim import _OPTIMIZER_DICT


def test_bim():
    # make a random tensor that requires a gradient
    random_matrix = np.random.normal(0, 1, size=(5,5)).astype(np.float32)
    foo = torch.tensor(random_matrix).requires_grad_(True)
    # define an optimizer
    lr = 0.1
    opt = BIM([foo], lr)
    # compute squared-error loss
    loss = torch.sum(foo**2)
    loss.backward()
    opt.step()
    # how large was the gradient update?
    change = np.abs(foo.detach().numpy() - random_matrix)
    # each element should be changed by the learning rate in one direction or the other
    assert np.allclose(change, lr, rtol=1e-5)
    
    
    
    
def test_mifgsm():
    # make a random tensor that requires a gradient
    random_matrix = np.random.normal(0, 1, size=(5,5)).astype(np.float32)
    foo = torch.tensor(random_matrix).requires_grad_(True)
    # define an optimizer
    lr = 0.1
    opt = MIFGSM([foo], lr)
    # compute squared-error loss
    loss = torch.sum(foo**2)
    loss.backward()
    opt.step()
    # how large was the gradient update?
    change = np.abs(foo.detach().numpy() - random_matrix)
    # each element should be changed by the learning rate in one direction or the other
    assert np.allclose(change, lr, rtol=1e-5)
    # and there should be a momentum buffer saved in the optimizer
    assert 'momentum_buffer' in list(opt.state.values())[0]
    
def test_get_optimizer_and_scheduler():
    # let's do all combinations in one fell swoop
    foo = torch.tensor(np.random.normal(0, 1,
                                        size=(2,2)).astype(np.float32)).requires_grad_(True)
    
    for optimizer in _OPTIMIZER_DICT:
        for decay in ["none", "cosine", "exponential"]:
            o,s = _get_optimizer_and_scheduler(optimizer, [foo], 1.0, decay, steps=10)
            o.step()
            s.step()
            if decay == "none":
                assert s.get_last_lr()[0] == 1.0
            else:
                assert s.get_last_lr()[0] < 1.0
                