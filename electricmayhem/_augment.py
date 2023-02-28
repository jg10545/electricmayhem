import numpy as np
import torch
import kornia.geometry, kornia.enhance, kornia.filters



def augment_image(image, warp, scale, gamma, blur):
    """
    Apply augmentations to an image
    
    :image: pytorch Tensor in channel-first format (C,H,W)
    :warp: float; scaling factor for perspective warp
    :scale: float; resize by this factor
    :gamma: float; adjust gamma using this parameter
    :blur: float; Gaussian blur kernel size. 0 to disable
    
    Returns a pytorch Tensor in channel-first format
    """
    C,H,W = image.shape
    image = image.unsqueeze(0)
    # warp perspective
    image = kornia.geometry.transform.warp_perspective(
        image, warp.unsqueeze(0),[H,W])
    # rescale
    image = kornia.geometry.transform.resize(
        image, (int(scale*H), int(scale*W)))
    # adjust gamma
    image = kornia.enhance.adjust_gamma(image, gamma)
    # gaussian blur
    if blur > 0:
        image = kornia.filters.gaussian_blur2d(
            image,
            (2*int(blur)+1, 2*int(blur)+1), (blur, blur))
    
    return image.squeeze()


def generate_aug_params(perspective_scale=1e-4, scale=(0.5, 1.5), 
                        gamma=(1,3.5), blur=[0, 3, 5]):
    """
    Randomly sample a set of augmentation parameters
    
    :perspective_scale: float; scaling factor for the magnitude of perspective warping
    :scale: tuple of floats; min and max scaling factor for resizing image
    :gamma: tuple of floats; min and max gamma adjust factor
    :blur: list of ints; kernel sizes for gaussian blur
    
    Returns a dictionary of augmentation parameters
    """
    return {
        "warp":torch.Tensor(np.eye(3)+np.random.normal(0, perspective_scale, 
                                              (3,3,)
                                             ).astype(np.float32)),
        "scale":float(np.random.uniform(*scale)),
        "gamma":float(np.random.uniform(*gamma)),
        "blur":float(np.random.choice(blur))
    }


def wiggle_mask_and_perturbation(mask, perturbation, angle_scale=1, translate_scale=5):
    """
    Apply a small transformation to the mask and perturbation with respect to the 
    original image, to make sure we're not overfitting to details of where the 
    patch is placed
    
    :mask: mask tensor in channel-first format
    :perturbation: perturbation tensor in channel-first format
    :angle_scale: standard deviation, in degrees, of normal distribution angle will be
        chosen from
    :translate_scale: standard deviation, in pixels, of normal distribution x and y translations
        will be chosen from
        
    Returns modified mask and perturbation
    """
    if angle_scale > 0:
        angle = torch.Tensor(np.random.normal(0, angle_scale, size=(1,)))
        mask = kornia.geometry.transform.rotate(mask.unsqueeze(0), angle).squeeze()
        pert = kornia.geometry.transform.rotate(perturbation.unsqueeze(0), angle).squeeze()
    if translate_scale > 0:
        translate = torch.Tensor(np.random.normal(0, translate_scale, size=(1,2)))
        mask = kornia.geometry.transform.translate(mask.unsqueeze(0), translate).squeeze()
        pert = kornia.geometry.transform.translate(pert.unsqueeze(0), translate).squeeze()
    return mask, pert


def compose(img, mask, perturbation, angle_scale=1, translate_scale=2,
            augment=None):
    """
    Paste a perturbation on top of an image using a mask. If necessary, resizes
    the perturbation and randomly jitters the perturbation/mask with respect to the image.
    
    :img: torch.Tensor in (C,H,W) format; victim image
    :mask: torch.Tensor in (1,H,W) format
    :perturbation: torch.Tensor in (C',H,W) format
    :angle_scale: standard deviation (in degrees) of normal distribution to sample from for jitter angle
    :translate_scale: standard deviation (in pixels) of normal distribution to sample from for jitter distance
    
    Returns image with perturbation as a 
    """
    C,H,W = img.shape
    # resize the perturbation if we need to
    if (perturbation.shape[1] != H)|(perturbation.shape[2] != W):
        perturbation = kornia.geometry.transform.resize(perturbation, (H,W))
    # if we're jittering the mask, do that now 
    if (angle_scale > 0)|(translate_scale > 0):
        mask, perturbation = wiggle_mask_and_perturbation(mask, perturbation, 
                                                            angle_scale, translate_scale)
    # glue it all together
    img_w_pert = torch.clamp(img*(1-mask) + perturbation*mask, 0, 1)
    
    # want to augment the image too? do that now.
    if augment is not None:
        img_w_pert = augment_image(img_w_pert, **augment)
    
    return img_w_pert