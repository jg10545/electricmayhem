import numpy as np
import torch

from electricmayhem.whitebox import loss

"""
def test_printability_loss():
    # define a printability calculator
    calc = loss.NPSCalculator((48,24))
    pa = calc.get_printability_array(None, (48,24))
    
    test_tensor = np.ones((5, 3, 48, 24))    
    test_tensor[:,0,:,:] *= 0.71765
    test_tensor[:,1,:,:] *= 0.32941
    test_tensor[:,2,:,:] *= 0.40784
    test_tensor = torch.tensor(test_tensor).requires_grad_(True)
    
    elementwise_loss = calc(test_tensor)
    assert len(elementwise_loss) == 5
    # these specific values should be close to zero
    assert elementwise_loss.detach().numpy().max() < 0.01
    # make sure this doesn't throw an error
    torch.sum(elementwise_loss).backward()
    assert isinstance(pa, torch.Tensor)
    
def test_total_variation_loss_zero_case():
    test_tensor_batch = torch.zeros((1,3,23,17), dtype=torch.float32)
    tvloss = loss.total_variation_loss(test_tensor_batch)
    assert isinstance(tvloss, torch.Tensor)
    assert tvloss.shape == ()
    assert tvloss.numpy().max() == 0
    
def test_total_variation_loss_nonzero_case():
    test_tensor_batch = torch.tensor(np.random.uniform(0, 1, size=(1,3,23,17)))
    tvloss = loss.total_variation_loss(test_tensor_batch)
    assert isinstance(tvloss, torch.Tensor)
    assert tvloss.shape == ()
    assert tvloss.numpy().max() > 0
"""
def test_saliency_loss_returns_correct_shape():
    test_tensor_batch = torch.tensor(np.random.uniform(0, 1, size=(1,3,23,17)))
    sal_loss = loss.saliency_loss(test_tensor_batch)
    assert isinstance(sal_loss, torch.Tensor)
    assert sal_loss.shape == ()
