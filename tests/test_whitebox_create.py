import numpy as np
import torch

from electricmayhem.whitebox import _create



def test_patchresizer():
    resizer = _create.PatchResizer((31, 29))
    
    assert resizer(torch.zeros((1,3,17,19), dtype=torch.float32)).shape == (1,3,31,29)
    assert resizer(torch.zeros((2,1,23,19), dtype=torch.float32)).shape == (2,1,31,29)