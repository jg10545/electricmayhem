import numpy as np
import torch
from PIL import Image, ImageCms

from electricmayhem.whitebox._pipeline import PipelineBase


def _tensor_batch_to_imgs(t):
    """
    Convert a tensor containing a stack of imags into
    a list of PIL.Image objects
    """
    imgs = []
    t = t.cpu().detach()
    for i in range(t.shape[0]):
        im = Image.fromarray((t[i].permute(1,2,0).numpy()*255).astype(np.uint8))
        imgs.append(im)
    return imgs

def _imgs_to_tensor_batch(imgs):
    """
    Convert a list of PIL.Image objects to a stack of torch Tensors
    """
    tensors = []
    for im in imgs:
        t = torch.tensor(np.array(im).astype(np.float32)/255).permute(2,0,1)
        tensors.append(t)
    return torch.stack(tensors,0)

class SoftProofer(PipelineBase):
    """
    Class to simulate soft-proofing during evaluation steps only- replaces
    the batch of patches with an estimate of what they'd look like printed
    out, where any out-of-gamut colors could be burned.

    This is mostly a light wrapper on tool's inside pillow's ImageCms module.
    """
    name = "SoftProofer"

    def __init__(self, target_profile, screen_profile=None, rendering_intent=0):
        """
        :target_profile: ImageCmsProfile or path to a saved .icc file; the ICC profile for
            the printer
        :screen_profile: ImageCmsProfile or path to a saved .icc file; the ICC profile for 
            the screen. If None, sRGB is used.
        :rendering_intent: integer 0-3
            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3
        """
        super().__init__()
        self.params = {"rendering_intent":rendering_intent}

        # make sure both the source and target profiles are loaded as ImageCmsProfiles
        if isinstance(target_profile, str):
            target_profile = ImageCms.getOpenProfile(target_profile)
        if isinstance(screen_profile, str):
            screen_profile = ImageCms.getOpenProfile(screen_profile)
        elif screen_profile is None:
            screen_profile = ImageCms.createProfile("sRGB")

        # create soft proofing transform
        self.tfm = ImageCms.buildProofTransform(screen_profile, 
                                                screen_profile,
                                                target_profile,
                                                "RGB", "RGB",
                                                renderingIntent=rendering_intent,
                                                proofRenderingIntent=rendering_intent)
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        intents = ["perceptual", "relative colorimetric", "saturation",
                   "absolute colorimetric"]
        return f"**SoftProofer:** rendering intent `{intents[self.params['rendering_intent']]}`"
    
    def softproof(self, x):
        """
        Use PIL to soft-proof a batch of tensors
        """
        # convert batch of image tensors to list of PIL.Image objects
        imgs = _tensor_batch_to_imgs(x)
        # soft-proof each image
        imgs_tfm = [self.tfm.apply(im) for im in imgs]
        # convert back to tensors
        newtensors = _imgs_to_tensor_batch(imgs_tfm)
        return newtensors
    
    def forward(self, x, control=False, evaluate=False, params=None, **kwargs):
        if evaluate:
            device = x.device
            proofed = self.softproof(x).to(device)
            return proofed
        else:
            return x
        
    def get_last_sample_as_dict(self):
        return {}
        
    def log_vizualizations(self, x, x_control, writer, step):
        """
        For this pipeline stage- just compute visualizations on the first element in the batch
        """
        x = x[:1,:,:,:].cpu().detach()
        proofed = self.softproof(x)
        writer.add_image("proofed_patch", proofed[0], global_step=step, dataformats='CHW')
        # how much did proofing change the patch?
        rms_diff = np.sqrt(np.mean((proofed.numpy()-x.numpy())**2))
        writer.add_scalar("proofed_patch_rms_change", rms_diff, global_step=step)