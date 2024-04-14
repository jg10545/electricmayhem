import torch
import kornia.geometry.transform

def warp_and_implant_batch(patch_batch, target_batch, coord_batch, mask=None):
    """
    :patch_batch: (B,C,H',W') tensor containing batch of patches
    :target_batch: (B,C,H,W) tensor containing batch of target images
    :coord_batch: (B,4,2) tensor containing corner coordinates for implanting the patch in each image
    :mask: optional, batch of masks
    """
    assert patch_batch.shape[0] == target_batch.shape[0], "batch dimensions need to line up"
    assert target_batch.shape[0] == coord_batch.shape[0], "batch dimensions need to line up"
    
    # get transformation matrix
    patch_border = torch.tensor([[0.,0.],
                                 [patch_batch.shape[3], 0], 
                                 [patch_batch.shape[3], patch_batch.shape[2]], 
                                 [0., patch_batch.shape[2]]]) # (4,2)
    patch_border = torch.stack([patch_border for _ in range(patch_batch.shape[0])],0) # (B,4,2)
    
    tfm = kornia.geometry.transform.get_perspective_transform(patch_border, coord_batch) # (B,3,3)
    # apply transformation to get a warped patch with green background
    warped = kornia.geometry.transform.warp_perspective(patch_batch, tfm,
                                                      (target_batch.shape[2], target_batch.shape[3]),
                                                      padding_mode="fill", 
                                                      fill_value=torch.tensor([0,1,0])) # (B,C,H,W)
    # use the green background to create a mask
    warpmask = ((warped[:,0,:,:] == 0)&(warped[:,1,:,:] == 1)&(warped[:,2,:,:] == 0)).type(torch.float32) # (B,H,W)
    warpmask = warpmask.unsqueeze(1) # (B,1,H,W)
    
    if mask is not None:
        # apply same transforms to batch of masks, but fill with zeros. patch will only show through
        # in places where mask > 0
        mask_pw = kornia.geometry.transform.warp_perspective(mask, tfm,
                                                      (target_batch.shape[2], target_batch.shape[3]),
                                                      padding_mode="zeros") # (B,C,H,W) or (1,1,H,W)
        # update the warp mask to exclude the patch wherever the mask is zero
        warpmask = (warpmask + (1-mask_pw)).clip(0,1)
    
    return target_batch*warpmask + warped*(1-warpmask)