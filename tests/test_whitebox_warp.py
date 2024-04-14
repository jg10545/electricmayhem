import numpy as np
import torch

from electricmayhem.whitebox._warp_implant import warp_and_implant_batch


def test_warp_and_implant_batch_gives_correct_output_shape():
    batch_size = 2

    target_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,21,31)).astype(np.float32))
    patch_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,5,7)).astype(np.float32))
    corners = torch.tensor([[1,2], [5,2], [5,7], [1,7]]).type(torch.float32)
    corners = torch.stack([corners,corners],0)

    implanted = warp_and_implant_batch(patch_batch, target_batch, corners)
    # output should have the shape of the target batch
    assert implanted.shape == target_batch.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target_batch == implanted) > 0



def test_warp_and_implant_batch_gives_correct_output_shape_with_mask():
    batch_size = 2

    target_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,21,31)).astype(np.float32))
    patch_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,5,7)).astype(np.float32))
    corners = torch.tensor([[1,2], [5,2], [5,7], [1,7]]).type(torch.float32)
    corners = torch.stack([corners,corners],0)
    mask = torch.tensor(np.random.uniform(0,1, size=(batch_size, 1, 5,7)).astype(np.float32))

    implanted = warp_and_implant_batch(patch_batch, target_batch, corners, mask=mask)
    # output should have the shape of the target batch
    assert implanted.shape == target_batch.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target_batch == implanted) > 0