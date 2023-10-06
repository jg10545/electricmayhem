import numpy as np
import torch

from electricmayhem.whitebox import _create



def test_patchresizer():
    resizer = _create.PatchResizer((31, 29))
    
    assert resizer(torch.zeros((1,3,17,19), dtype=torch.float32)).shape == (1,3,31,29)
    assert resizer(torch.zeros((2,1,23,19), dtype=torch.float32)).shape == (2,1,31,29)
    assert "31, 29" in resizer.get_description()


def test_patchstacker():
        resizer = _create.PatchStacker(num_channels=3)
        
        patch_params = torch.zeros((1,1,32,32)).type(torch.float32)
        output = resizer(patch_params)
        assert output.shape == (1,3,32,32)
        assert "3 channels" in resizer.get_description()
        
        
def test_patchsaver():
        resizer = _create.PatchSaver()
        
        patch_params = torch.zeros((1,1,32,32)).type(torch.float32)
        output = resizer(patch_params)
        assert output.shape == (1,1,32,32)
        
        
    

def test_patchtiler():
    resizer = _create.PatchTiler((31, 29))
    
    assert resizer(torch.zeros((1,3,17,19), dtype=torch.float32)).shape == (1,3,31,29)
    assert resizer(torch.zeros((2,1,23,19), dtype=torch.float32)).shape == (2,1,31,29)
    assert "31, 29" in resizer.get_description()
