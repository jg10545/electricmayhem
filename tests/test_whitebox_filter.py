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
        


def test_HighPassFilter_multiple_patches():
    test_batch = {"foo":torch.tensor(np.random.uniform(0, 1, size=(1,3,21,37))),
                  "bar":torch.tensor(np.random.uniform(0, 1, size=(1,1,17,13)))}
    
    for limit in [5,7,21]:
        hp = HighPassFilter(limit)
        output, _ = hp(test_batch)
        assert isinstance(output, dict)
        assert output["foo"].shape == (1,3,21,37)
        assert output["bar"].shape == (1,1,17,13)
        

def test_HighPassFilter_multiple_patches_but_leave_one_out():
    test_batch = {"foo":torch.tensor(np.random.uniform(0, 1, size=(1,3,21,37))),
                  "bar":torch.tensor(np.random.uniform(0, 1, size=(1,1,17,13)))}
    
    for limit in [5,7,21]:
        hp = HighPassFilter(limit, keys=["foo"])
        output, _ = hp(test_batch)
        assert isinstance(output, dict)
        assert output["foo"].shape == (1,3,21,37)
        assert output["bar"].shape == (1,1,17,13)
        assert np.max(np.abs(output["bar"].numpy() - test_batch["bar"].numpy())) < 1e-5
        assert np.max(np.abs(output["foo"].numpy() - test_batch["foo"].numpy())) > 1e-5
        