import numpy as np
import torch
import kornia.geometry
import logging

from ._pipeline import PipelineBase


class PatchSaver(PipelineBase):
    """
    Pass-through for a patch that'll log it to tensorboard
    """
    name = "PatchSaver"
    
    def __init__(self, logviz=True):
        """
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called."
        """
        super().__init__()
        
        self.params = {}
        self.lastsample = {}
        self._logviz = logviz
        
    
    def forward(self, patches, control=False, evaluate=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        return patches
    
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}
    
    def log_vizualizations(self, x, x_control, writer, step):
        """
        """
        if self._logviz:
            img = self(x)[0]
            # check to make sure this is an RGB image
            if img.shape[0] == 3:
                writer.add_image(f"{self.name}_patch", img, global_step=step)


class PatchResizer(PatchSaver):
    """
    Class for resizing a batch of patches to a fixed size. Wraps
    kornia.geometry.
    """
    name = "PatchResizer"
    
    def __init__(self, size, interpolation="bilinear", logviz=True):
        """
        :size: length-2 tuple giving target size (H,W)
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()
        assert len(size) == 2, "expected a length-2 tuple (H,W) for the size"
        self.size = size
        
        self.params = {"size":list(size), "interpolation":interpolation}
        self.lastsample = {}
        self._logviz = logviz
        
    
    def forward(self, patches, control=False, evaluate=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        return kornia.geometry.resize(patches, self.size, 
                                      interpolation=self.params["interpolation"])
    
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.name}:** resize to {self.size}"
        
    
    
class PatchStacker(PatchSaver):
    """
    Turn a 1-channel patch into a 3-channel patch
    """
    name = "PatchStacker"
    
    def __init__(self, num_channels=3, logviz=True):
        """
        :size: length-2 tuple giving target size (H,W)
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()
        
        self.params = {"num_channels":num_channels}
        self.lastsample = {}
        self._logviz = logviz
        
    
    def forward(self, patches, control=False, evaluate=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        # dimension 0 is batch dimension; dimension 1 is channels
        return torch.concat([patches for _ in range(self.params["num_channels"])], 1)
    
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.name}:** stack to {self.params['num_channels']} channels"
    
    
class PatchTiler(PatchSaver):
    """
    Class for tiling a batch of tiny patches to a fixed size. 
    """
    name = "PatchTiler"
    
    def __init__(self, size, logviz=True):
        """
        :size: length-2 tuple giving target size (H,W)
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()
        assert len(size) == 2, "expected a length-2 tuple (H,W) for the size"
        self.size = size
        
        self.params = {"size":list(size)}
        self.lastsample = {}
        self._logviz = logviz
        
    
    def forward(self, patches, control=False, evaluate=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        H,W = self.size
        # figure out how many times to tile along vertical direction
        numtiles_H = H//patches.shape[2] + 1
        patchcolumn = torch.concat([patches for _ in range(numtiles_H)],2)[:,:,:H,:]
        # same basic computation for horizontal
        numtiles_W = W//patches.shape[3] + 1
        output = torch.concat([patchcolumn for _ in range(numtiles_W)], 3)[:,:,:,:W]
        return output
    
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.name}:** tile to {self.size}"
        