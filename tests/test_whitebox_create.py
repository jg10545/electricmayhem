import numpy as np
import torch

from electricmayhem.whitebox import _create



def test_patchresizer():
    resizer = _create.PatchResizer((31, 29))
    
    assert resizer(torch.zeros((1,3,17,19), dtype=torch.float32))[0].shape == (1,3,31,29)
    assert resizer(torch.zeros((2,1,23,19), dtype=torch.float32))[0].shape == (2,1,31,29)
    assert "31, 29" in resizer.get_description()


def test_patchresizer_with_multiple_patches():
    patch_params = {"foo":torch.zeros((1,3,32,32)).type(torch.float32),
                        "bar":torch.zeros((1,1,13,17)).type(torch.float32)}
    sizes = {"foo":(31,29), "bar":(17,23)}
    resizer = _create.PatchResizer(sizes)

    output, _ = resizer(patch_params)
    assert output["foo"].shape == (1,3,31,29)
    assert output["bar"].shape == (1,1,17,23)
    assert "31, 29" in resizer.get_description()
    assert "17, 23" in resizer.get_description()


def test_patchstacker():
        resizer = _create.PatchStacker(num_channels=3)
        
        patch_params = torch.zeros((1,1,32,32)).type(torch.float32)
        output, _ = resizer(patch_params)
        assert output.shape == (1,3,32,32)
        assert "3 channels" in resizer.get_description()
        
        
def test_patchsaver():
        resizer = _create.PatchSaver()
        
        patch_params = torch.zeros((1,1,32,32)).type(torch.float32)
        output, _ = resizer(patch_params)
        assert output.shape == (1,1,32,32)
        
        

def test_patchsaver_with_multiple_patches():
        resizer = _create.PatchSaver()
        
        patch_params = {"foo":torch.zeros((1,1,32,32)).type(torch.float32),
                        "bar":torch.zeros((1,1,13,17)).type(torch.float32)}
        output, _ = resizer(patch_params)
        assert isinstance(output, dict)
        assert output["foo"].shape == (1,1,32,32)
        assert output["bar"].shape == (1,1,13,17)
        
        
    

def test_patchtiler():
    resizer = _create.PatchTiler((31, 29))
    
    assert resizer(torch.zeros((1,3,17,19), dtype=torch.float32))[0].shape == (1,3,31,29)
    assert resizer(torch.zeros((2,1,23,19), dtype=torch.float32))[0].shape == (2,1,31,29)
    assert "31, 29" in resizer.get_description()


def test_scroll_single_image():
    x = torch.tensor(np.random.uniform(0, 1, size=(3,23,37)).astype(np.float32))
    for offset_x, offset_y in [(0,0), (10,5), (1,36), (20,1)]:
        shifted = _create.scroll_single_image(x, offset_x, offset_y)
        # output should be the same shape
        assert shifted.shape == x.shape
        # if we add up the values they should be the same
        assert (torch.sum(x**2).numpy() - torch.sum(shifted**2).numpy())**2 < 1e-5

def test_patchscroller():
    B = 5
    scroll = _create.PatchScroller()
    test_img = torch.tensor(np.random.uniform(0, 1, size=(3,23,31)).astype(np.float32))
    test_batch = torch.stack([test_img for _ in range(B)], 0)
    scrolled_batch, _ = scroll(test_batch)
    
    assert scrolled_batch.shape == test_batch.shape
    for i in range(B):
        assert (scrolled_batch[i].detach().numpy().sum() - test_img.numpy().sum())**2 < 1e-5
        