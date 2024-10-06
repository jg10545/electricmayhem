import numpy as np
import torch
import kornia.geometry

from electricmayhem.whitebox._warp_implant import warp_and_implant_batch, WarpPatchImplanter, get_mask, warp_and_implant_single

H = 40
W = 27
B = 50
C = 3
imagedict = {"foo":np.random.randint(0,255, (H,W,C)),
            "bar":np.random.randint(0,255, (H,W,C))}
boxdict = {"foo":[[[10,10],[10,35],[20,35],[20,10]], [[0,10],[0,35],[23,35],[20,10]]],
          "bar":[[[10,10],[10,38],[20,35],[20,10]], [[5,4],[10,35],[20,35],[20,10]]]}

boxdict_bad = {"foo":[[[10,10],[10,35],[20,35],[20]], [[0,10],[23,35],[20,10]]],
          "bar":[[[10,10],[10,38],[20,35],[20,10]], [[5,4],[10,35],[20,35],[20,10]]]}

patch_batch = torch.tensor(np.random.uniform(0,1, (B,C,7,11)).astype(np.float32))
mask = torch.tensor(np.random.uniform(0, 1, size=(H,W)).astype(np.float32))

patch_border = torch.tensor([[0.,0.],
                                 [patch_batch.shape[3], 0], 
                                 [patch_batch.shape[3], patch_batch.shape[2]], 
                                 [0., patch_batch.shape[2]]]).unsqueeze(0) # (1,4,2)
corners = torch.tensor([[1,2], [5,2], [5,7], [1,7]]).type(torch.float32).unsqueeze(0)
tfm = kornia.geometry.transform.get_perspective_transform(patch_border, corners)
target = torch.tensor(np.random.uniform(0,1,size=(3,H,W)).astype(np.float32))
chromakey = kornia.geometry.transform.warp_perspective(patch_batch[0].unsqueeze(0), tfm,
                                                      (target.shape[1], target.shape[2]),
                                                      padding_mode="fill", 
                                                      fill_value=torch.tensor([0,1,0])) # (B,C,H,W)
# use the green background to create a mask for deciding where to overwrite the target image
# with the patch
warpmask = ((chromakey[:,0,:,:] == 0)&(chromakey[:,1,:,:] == 1)&(chromakey[:,2,:,:] == 0)).type(torch.float32) # (B,H,W)
warpmask = warpmask.unsqueeze(1) # (1,1,H,W)


def test_warp_and_implant_single_gives_correct_output_shape():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


def test_warp_and_implant_single_gives_correct_output_shape_with_brightness_scaling():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


def test_warp_and_implant_single_gives_correct_output_shape_with_tensor_mask():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=mask)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


def test_warp_and_implant_single_gives_correct_output_shape_with_scalar_mask():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=0.5)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


def test_warp_and_implant_single_gives_correct_output_shape_with_tensor_mask_and_brightness_scaling():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=mask,
                                        scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


def test_warp_and_implant_single_gives_correct_output_shape_with_scalar_mask_and_brightness_scaling():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=0.5,
                                        scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0


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



def test_warp_and_implant_batch_gives_correct_output_shape_with_brightness_scaling():
    batch_size = 2

    target_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,21,31)).astype(np.float32))
    patch_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,5,7)).astype(np.float32))
    corners = torch.tensor([[1,2], [5,2], [5,7], [1,7]]).type(torch.float32)
    corners = torch.stack([corners,corners],0)

    implanted = warp_and_implant_batch(patch_batch, target_batch, corners, scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target_batch.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target_batch == implanted) > 0



def test_warp_and_implant_batch_gives_correct_output_shape_with_tensor_mask():
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

def test_warp_and_implant_batch_gives_correct_output_shape_with_scalar_mask():
    batch_size = 2

    target_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,21,31)).astype(np.float32))
    patch_batch = torch.tensor(np.random.uniform(0,1,size=(batch_size,3,5,7)).astype(np.float32))
    corners = torch.tensor([[1,2], [5,2], [5,7], [1,7]]).type(torch.float32)
    corners = torch.stack([corners,corners],0)
    mask = 0.5

    implanted = warp_and_implant_batch(patch_batch, target_batch, corners, mask=mask)
    # output should have the shape of the target batch
    assert implanted.shape == target_batch.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target_batch == implanted) > 0

def test_warppatchimplanter():
    # simple checks- output shape and reproducibility
    warp = WarpPatchImplanter(imagedict, boxdict)
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6
    assert warp.validate(patch_batch)

def test_bad_warppatchimplanter_fails_validation():
    # simple checks- output shape and reproducibility
    warp = WarpPatchImplanter(imagedict, boxdict_bad)
    
    assert not warp.validate(patch_batch)
    
def test_warppatchimplanter_with_scalar_mask():
    # simple checks- output shape and reproducibility
    warp = WarpPatchImplanter(imagedict, boxdict, mask=0.5)
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6
    
def test_warppatchimplanter_with_2D_mask():
    # simple checks- output shape and reproducibility
    warp = WarpPatchImplanter(imagedict, boxdict, mask=mask)
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6


def test_get_mask_returns_correct_shape():
    shape = (3,640,640)
    coords = [[299.0, 196.0], [477.0, 205.0], [477.0, 394.0], [309.0, 373.0]]
    mask = get_mask(shape, coords)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == shape