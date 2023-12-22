import numpy as np
import torch

from electricmayhem.whitebox import loss


def test_printability_loss():
    # define a printability calculator
    calc = loss.NPSCalculator((48,24))
    
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
    
def test_total_variation_loss():
    test_tensor_batch = torch.zeros((1,3,23,17), dtype=torch.float32)
    tvloss = loss.compute_total_variation_loss(test_tensor_batch)
    assert isinstance(tvloss, torch.Tensor)
    assert tvloss.shape == ()
    assert tvloss.numpy().max() == 0