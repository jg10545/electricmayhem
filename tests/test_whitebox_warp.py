import numpy as np
import pandas as pd
import torch
import kornia.geometry
from PIL import Image

from electricmayhem.whitebox._warp_implant import warp_and_implant_batch, WarpPatchImplanter, warp_and_implant_single
from electricmayhem.whitebox._warp_implant import get_warpmask_and_tfm, unpack_warp_dataframe

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
mask = torch.tensor(np.random.uniform(0, 1, size=(7,11)).astype(np.float32))

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

def test_get_warpmask_and_tfm_gives_correct_output_shapes():
    patch_shape = (3,11,13)
    w, t = get_warpmask_and_tfm(target.shape, patch_shape, corners)
    assert w.shape[1:] == target.shape

def test_get_warpmask_and_tfm_gives_correct_output_shapes_scalar_mask():
    patch_shape = (3,11,13)
    w, t = get_warpmask_and_tfm(target.shape, patch_shape, corners, mask=0.5)
    assert w.shape[1:] == target.shape
    assert torch.min(w) == 0.5

def test_get_warpmask_and_tfm_gives_correct_output_shapes_tensor_mask():
    patch_shape = (3,11,13)
    mask = 0.5*torch.ones(patch_shape)
    w, t = get_warpmask_and_tfm(target.shape, patch_shape, corners, mask=mask)
    assert w.shape[1:] == target.shape
    assert torch.min(w) == 0.5

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

"""
def test_warp_and_implant_single_gives_correct_output_shape_with_tensor_mask():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=mask)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0"""

"""
def test_warp_and_implant_single_gives_correct_output_shape_with_scalar_mask():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=0.5)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0"""


"""def test_warp_and_implant_single_gives_correct_output_shape_with_tensor_mask_and_brightness_scaling():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=mask,
                                        scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0"""

"""
def test_warp_and_implant_single_gives_correct_output_shape_with_scalar_mask_and_brightness_scaling():
    implanted = warp_and_implant_single(patch_batch[0], target, tfm[0].unsqueeze(0), warpmask, mask=0.5,
                                        scale_brightness=True)
    # output should have the shape of the target batch
    assert implanted.shape == target.shape
    # some pixels in the target batch should be unchanged
    assert torch.sum(target == implanted) > 0"""


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


def test_unpack_warp_dataframe(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes)
    for k in im_filename_list:
        assert k in [test_png_1, test_png_2]
    assert tfms[test_png_1][0].shape == (1,3,3)
    img = np.array(Image.open(test_png_1))
    assert warpmasks[test_png_1][0].shape[1:] == img.shape[:2]
    for k in patchnames:
        for p in patchnames[k]:
            assert p in ["foo", "bar"]
    assert target_images[test_png_1].permute(1,2,0).shape == img.shape


def test_unpack_warp_dataframe_with_scalar_mask(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes, mask=0.5)
    for k in im_filename_list:
        assert k in [test_png_1, test_png_2]
    assert tfms[test_png_1][0].shape == (1,3,3)
    img = np.array(Image.open(test_png_1))
    assert warpmasks[test_png_1][0].shape[1:] == img.shape[:2]
    for k in patchnames:
        for p in patchnames[k]:
            assert p in ["foo", "bar"]
    assert target_images[test_png_1].permute(1,2,0).shape == img.shape
    assert torch.min(warpmasks[test_png_1][0]) == 0.5


def test_unpack_warp_dataframe_with_tensor_mask(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    mask = {k:0.5*torch.ones(patch_shapes[k]) for k in patch_shapes}
    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes, mask=mask)
    for k in im_filename_list:
        assert k in [test_png_1, test_png_2]
    assert tfms[test_png_1][0].shape == (1,3,3)
    img = np.array(Image.open(test_png_1))
    assert warpmasks[test_png_1][0].shape[1:] == img.shape[:2]
    for k in patchnames:
        for p in patchnames[k]:
            assert p in ["foo", "bar"]
    assert target_images[test_png_1].permute(1,2,0).shape == img.shape
    assert torch.min(warpmasks[test_png_1][0]) == 0.5



def test_unpack_warp_dataframe_with_tensor_and_scalar_masks(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    mask = {k:0.5*torch.ones(patch_shapes[k]) for k in patch_shapes}
    mask["bar"] = 0.75
    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes, mask=mask)
    for k in im_filename_list:
        assert k in [test_png_1, test_png_2]
    assert tfms[test_png_1][0].shape == (1,3,3)
    img = np.array(Image.open(test_png_1))
    assert warpmasks[test_png_1][0].shape[1:] == img.shape[:2]
    for k in patchnames:
        for p in patchnames[k]:
            assert p in ["foo", "bar"]
    assert target_images[test_png_1].permute(1,2,0).shape == img.shape
    assert torch.min(warpmasks[test_png_1][0]) == 0.5


def test_unpack_warp_dataframe_with_tensor_and_2D_scalar_mask(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    mask = {k:0.5*torch.ones(patch_shapes[k][1:]) for k in patch_shapes}
    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes, mask=mask)
    for k in im_filename_list:
        assert k in [test_png_1, test_png_2]
    assert tfms[test_png_1][0].shape == (1,3,3)
    img = np.array(Image.open(test_png_1))
    assert warpmasks[test_png_1][0].shape[1:] == img.shape[:2]
    for k in patchnames:
        for p in patchnames[k]:
            assert p in ["foo", "bar"]
    assert target_images[test_png_1].permute(1,2,0).shape == img.shape
    assert torch.min(warpmasks[test_png_1][0]) == 0.5

def test_warppatchimplanter(test_png_1, test_png_2):
    # simple checks- output shape and reproducibility
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    patch_batch = {k:torch.tensor(np.random.uniform(0, 1, patch_shapes[k]).astype(np.float32)).unsqueeze(0)
                   for k in patch_shapes}
    

    im_filename_list, tfms, warpmasks, patchnames, target_images = unpack_warp_dataframe(df, patch_shapes)
    implanted = warp_and_implant_single(patch_batch["foo"][0], target_images[test_png_1], tfms[test_png_1][0], 
                                        warpmasks[test_png_1][0].unsqueeze(0))
    assert implanted.shape == (3,100,100)
    
    warp = WarpPatchImplanter(df, patch_shapes)
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    img = np.array(Image.open(test_png_1))
    B = 1
    C = img.shape[2]
    H = img.shape[0]
    W = img.shape[1]
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6

#def test_bad_warppatchimplanter_fails_validation():
#    # simple checks- output shape and reproducibility
#    warp = WarpPatchImplanter(imagedict, boxdict_bad)
#    
#    assert not warp.validate(patch_batch)
    
def test_warppatchimplanter_with_scalar_mask(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    patch_batch = {k:torch.tensor(np.random.uniform(0, 1, patch_shapes[k]).astype(np.float32)).unsqueeze(0)
                   for k in patch_shapes}
    img = np.array(Image.open(test_png_1))
    B = 1
    C = img.shape[2]
    H = img.shape[0]
    W = img.shape[1]
    
    warp = WarpPatchImplanter(df, patch_shapes, mask = 0.5)
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6
    
def test_warppatchimplanter_with_2D_mask(test_png_1, test_png_2):
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2, test_png_2],
                         "ulx":[10, 60, 13, 50],
                         "uly":[5, 70, 15, 62],
                         "urx":[20, 72, 26, 81],
                         "ury":[10, 68, 20, 72],
                         "lrx":[22, 70, 20, 79],
                         "lry":[25, 90, 30, 80],
                         "llx":[9, 55, 20, 55],
                         "lly":[30, 92, 28, 78],
                         "patch":["foo", "bar", "foo", "bar"]})
    
    patch_shapes = {"foo":(3,17,19), "bar":(3,23,7)}
    patch_batch = {k:torch.tensor(np.random.uniform(0, 1, patch_shapes[k]).astype(np.float32)).unsqueeze(0)
                   for k in patch_shapes}
    img = np.array(Image.open(test_png_1))
    B = 1
    C = img.shape[2]
    H = img.shape[0]
    W = img.shape[1]
    m = torch.tensor(np.random.uniform(0, 1, patch_shapes["foo"]).astype(np.float32))
    
    warp = WarpPatchImplanter(df, patch_shapes, mask={"foo":m, "bar":1.})
    output, _ = warp(patch_batch)
    output2, _ = warp(patch_batch, params=warp.lastsample)
    
    assert output.shape == (B, C, H, W)
    assert np.mean((output.detach().numpy() - output2.detach().numpy())**2) < 1e-6


