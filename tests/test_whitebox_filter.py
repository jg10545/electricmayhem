import numpy as np
import torch

from electricmayhem.whitebox._filter import highpass, HighPassFilter


def test_highpass():
    test_batch = torch.tensor(np.random.uniform(0, 1, size=(1,3,21,37)))
    
    for limit in [5,7,21]:
        output = highpass(test_batch, limit, limit)
        assert output.shape == test_batch.shape
        

def test_HighPassFilter():
    test_batch = torch.tensor(np.random.uniform(0, 1, size=(1,3,21,37)))
    
    for limit in [5,7,21]:
        hp = HighPassFilter(limit)
        output, _ = hp(test_batch)
        assert output.shape == test_batch.shape
        