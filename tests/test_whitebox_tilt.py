import numpy as np
import torch
import kornia.geometry

from electricmayhem.whitebox._tilt import sample_perspective_transforms

def test_sample_perspective_transforms():
    N = 5
    H = 67
    W = 103

    # compute the transforms
    warps, scales = sample_perspective_transforms(N,H,W)

    assert isinstance(warps, torch.Tensor)
    assert warps.shape == (N,3,3)
    assert isinstance(scales, np.ndarray)
    assert scales.shape == (N,)

    # are they viable transformations?
    test_image_batch = torch.tensor(np.random.uniform(0,1, size=(N,3,H,W)).astype(np.float32))
    warped = kornia.geometry.warp_perspective(test_image_batch, warps, (H,W))
    assert isinstance(warped, torch.Tensor)
    assert warped.shape == (N,3,H,W)