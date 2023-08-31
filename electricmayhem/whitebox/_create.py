import numpy as np
import torch
import kornia.geometry
import logging

from ._pipeline import PipelineBase



class PatchResizer(PipelineBase):
    """
    Class for resizing a batch of patches to a fixed size. Wraps
    kornia.geometry.
    """
    name = "PatchResizer"
    
    def __init__(self, size, interpolation="bilinear"):
        """
        :size: length-2 tuple giving target size (H,W)
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        """
        super().__init__()
        assert len(size) == 2, "expected a length-2 tuple (H,W) for the size"
        self.size = size
        
        self.params = {"size":list(size), "interpolation":interpolation}
        self.lastsample = {}
        
    
    def forward(self, patches, control=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        return kornia.geometry.resize(patches, self.size, 
                                      interpolation=self.params["interpolation"])
    
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}
    
    
class PatchStacker(PipelineBase):
    """
    Turn a 1-channel patch into a 3-channel patch
    """
    name = "PatchStacker"
    
    def __init__(self, num_channels=3):
        """
        :size: length-2 tuple giving target size (H,W)
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        """
        super().__init__()
        
        self.params = {"num_channels":num_channels}
        self.lastsample = {}
        
    
    def forward(self, patches, control=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        # dimension 0 is batch dimension; dimension 1 is channels
        return torch.concat([patches for _ in range(self.params["num_channels"])], 1)
    
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}