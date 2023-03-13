import numpy as np
import torch
import kornia.geometry, kornia.enhance, kornia.filters



def augment_image(image, warp=None, scale=0, gamma=0, blur=0, 
                  angle=0, translate_x=0, translate_y=0,
                  mask=None, perturbation=None):
    """
    Apply augmentations to an image, including composition augmentations.
    
    :image: pytorch Tensor in channel-first format (C,H,W)
    :warp: float; scaling factor for perspective warp
    :scale: float; resize by this factor
    :gamma: float; adjust gamma using this parameter
    :blur: float; Gaussian blur kernel size. 0 to disable
    :angle: angle to rotate mask and perturbation by
    :translate_x: pixels to shift mask/perturbation by in x dimension
    :translate_y: pixels to shift mask/perturbation by in y dimension
    :mask: torch.Tensor in (1,H,W) format
    :pert: torch.Tensor in (C',H,W) format
    
    Returns a pytorch Tensor in channel-first format
    """
    C,H,W = image.shape
    
    # if we need to compose an image with a perturbation, do that first
    if (mask is not None)&(perturbation is not None):
        if scale > 0:
            translate_x *= scale
            translate_y *= scale
        image = compose(image, mask, perturbation, angle=angle,
                        translate_x=translate_x, translate_y=translate_y)
    
    
    image = image.unsqueeze(0)
    # warp perspective
    if warp is not None:
        image = kornia.geometry.transform.warp_perspective(
            image, warp.unsqueeze(0),[H,W])
    # rescale
    if scale > 0:
        image = kornia.geometry.transform.resize(
            image, (int(scale*H), int(scale*W)))
    # adjust gamma
    if gamma > 0:
        image = kornia.enhance.adjust_gamma(image, gamma)
    # gaussian blur
    if blur > 0:
        image = kornia.filters.gaussian_blur2d(
            image,
            (2*int(blur)+1, 2*int(blur)+1), (blur, blur))
    
    return image.squeeze()


def generate_aug_params(perspective_scale=1e-4, scale=(0.5, 1.5), 
                        gamma=(1,3.5), blur=[0, 3, 5],
                        angle=1, translate=1):
    """
    Randomly sample a set of augmentation parameters
    
    :perspective_scale: float; scaling factor for the magnitude of perspective warping
    :scale: tuple of floats; min and max scaling factor for resizing image
    :gamma: tuple of floats; min and max gamma adjust factor
    :blur: list of ints; kernel sizes for gaussian blur
    
    Returns a dictionary of augmentation parameters
    """
    outdict = {}
    if perspective_scale > 0:
        outdict["warp"] = torch.Tensor(np.eye(3)+np.random.normal(0,
                                        perspective_scale, 
                                        (3,3,)).astype(np.float32))
    if isinstance(scale, tuple):
        outdict["scale"] = float(np.random.uniform(*scale))
    if isinstance(gamma, tuple):
        outdict["gamma"] = float(np.random.uniform(*gamma))
    if isinstance(blur, list):
        outdict["blur"] = float(np.random.choice(blur))
    if angle > 0:
        outdict["angle"] = float(np.random.normal(0, angle))
    if translate > 0:
        outdict["translate_x"] = float(np.random.normal(0, translate))
        outdict["translate_y"] = float(np.random.normal(0, translate))
    return outdict




def compose(img, mask, pert, angle=0, translate_x=0, translate_y=0):
    """
    Paste a perturbation on top of an image using a mask. If necessary, resizes
    the perturbation and randomly jitters the perturbation/mask with respect to the image. 
    
    :img: torch.Tensor in (C,H,W) format; victim image
    :mask: torch.Tensor in (1,H,W) format
    :pert: torch.Tensor in (C',H,W) format
    :angle: angle to rotate mask and perturbation by
    :translate_x: pixels to shift mask/perturbation by in x dimension
    :translate_y: pixels to shift mask/perturbation by in y dimension
    
    Returns image with perturbation as a torch.Tensor
    """
    C,H,W = img.shape
    # resize the perturbation if we need to
    if (pert.shape[1] != H)|(pert.shape[2] != W):
        pert = kornia.geometry.transform.resize(pert, (H,W))
    # if we're jittering the mask, do that now 
    if angle != 0:
        angle = torch.Tensor([angle])
        mask = kornia.geometry.transform.rotate(mask.unsqueeze(0), angle).squeeze()
        pert = kornia.geometry.transform.rotate(pert.unsqueeze(0), angle).squeeze()
    if (translate_x != 0)|(translate_y != 0):
        translate = torch.Tensor([[translate_x, translate_y]])
        mask = kornia.geometry.transform.translate(mask.unsqueeze(0),
                                                   translate).squeeze()
        pert = kornia.geometry.transform.translate(pert.unsqueeze(0),
                                                   translate).squeeze()
        
    # glue it all together
    img = torch.clamp(img*(1-mask) + pert*mask, 0, 1)
    
    return img