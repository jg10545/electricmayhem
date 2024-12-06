import numpy as np
import torch
import kornia.geometry
import logging

from ._pipeline import PipelineBase


def scroll_single_image(x, offset_x=0, offset_y=0):
    """
    helper function for PatchScroller. take an image and translate it,
    wrapping with toroidal boundary conditions

    :x: torch tensor for a single image in channel-first format (C,H,W)
    :offset_x: int; number of pixels to offset in x direction
    :offset_y: int; number of pixels to offset in y direction
    """

    x = torch.concat([x[:, :, offset_x:], x[:, :, :offset_x]], 2)
    x = torch.concat([x[:, offset_y:, :], x[:, :offset_y, :]], 1)
    return x


class PatchSaver(PipelineBase):
    """
    Pipeline stage that does nothing except log the patch to tensorboard
    """
    def __init__(self, logviz=True):
        """
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called."
        """
        super().__init__()

        self.params = {}
        self.lastsample = {}
        self._logviz = logviz

    def _forward_single(self, patches, control=False, evaluate=False, params={}, key=None, **kwargs):
        """
        Implant a batch of patches in a batch of images

        :patches: torch Tensor; stack of patches
        :control: no effect on this function
        :kwargs: no effect on this function
        """
        return patches, kwargs

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}

    def _log_single_image(self, img, name, writer, step):
        # if it's a batch take the first element
        if len(img.shape) == 4:
            img = img[0]
        # check to make sure it's a 3-channel image
        if img.shape[0] == 3:
            writer.add_image(name, img, global_step=step)

    def log_vizualizations(self, x, x_control, writer, step, logging_to_mlflow=False):
        """ """
        if self._logviz:
            if isinstance(x, dict):
                for k in x:
                    self._log_single_image(x[k], f"{self.__class__.__name__}_patch_{k}", writer, step)
            else:
                self._log_single_image(x, f"{self.__class__.__name__}_patch", writer, step)


class PatchResizer(PatchSaver):
    """
    Class for resizing a batch of patches to a fixed size. Wraps
    kornia.geometry.
    """
    def __init__(self, size, interpolation="bilinear", logviz=True):
        """
        :size: length-2 tuple giving target size (H,W). for resizing multiple patches use
            a dictionary of tuples. Leave a patch key out if you don't want to resize that one.
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()

        self.params = {"interpolation": interpolation}
        if isinstance(size, dict):
            for k in size:
                assert len(size[k]) == 2, "expected a length-2 tuple (H,W) for the size"
            self.params["size"] = {k: list(size[k]) for k in size}
        else:
            assert len(size) == 2, "expected a length-2 tuple (H,W) for the size"
            self.params["size"] = list(size)
        self.size = size

        self.lastsample = {}
        self._logviz = logviz

    
    def _forward_single(self, x, control=False, evaluate=False, params={}, key=None, **kwargs):
        # single patch or applying on all patches
        if (key is None)|isinstance(self.size, tuple):
            size = self.size
        # multiple patches
        elif key in self.size:
            size = self.size[key]
        else:
            return x, kwargs
        resized = kornia.geometry.resize(
                x, size, interpolation=self.params["interpolation"]
            )
        return resized, kwargs


    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        if isinstance(self.size, dict):
            newsize = ", ".join([f"{k}: {str(self.size[k])}" for k in self.size])
        else:
            newsize = self.size
        return f"**{self.__class__.__name__}:** resize to {newsize}"


class PatchStacker(PatchSaver):
    """
    Turn a 1-channel patch into a 3-channel patch
    """
    def __init__(self, num_channels=3, logviz=True, keys=None):
        """
        :size: length-2 tuple giving target size (H,W)
        :interpolation: string; 'bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', or 'area'
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        :keys: None or list of strings. If using multiple patches, use this to specify which patches to
            apply to when forward() is called.
        """
        super().__init__()

        self.params = {"num_channels": num_channels}
        self.lastsample = {}
        self._logviz = logviz
        self.keys = keys
        if keys is not None:
            self.params["keys"] = keys

    def _forward_single(self, x, control=False, evaluate=False, params={}, key=None, **kwargs):
        return torch.concat([x for _ in range(self.params["num_channels"])], 1), kwargs

    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.__class__.__name__}:** stack to {self.params['num_channels']} channels"


class PatchTiler(PatchSaver):
    """
    Class for tiling a batch of tiny patches to a fixed size.
    """
    def __init__(self, size, logviz=True):
        """
        :size: length-2 tuple giving target size (H,W) or dictionary of tuples if using
            multiple patches. Any patch keys not included will not be tiled.
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()
        if isinstance(size, dict):
            for k in size:
                assert len(size[k]) == 2, "expected a length-2 tuple (H,W) for the size"
            self.params["size"] = {k: list(size[k]) for k in size}
        else:
            assert len(size) == 2, "expected a length-2 tuple (H,W) for the size"
            self.params = {"size": list(size)}
        self.size = size

        self.lastsample = {}
        self._logviz = logviz

    def _tile_single_patch(self, patches, size):
        H, W = size
        # figure out how many times to tile along vertical direction
        numtiles_H = H // patches.shape[2] + 1
        patchcolumn = torch.concat([patches for _ in range(numtiles_H)], 2)[:, :, :H, :]
        # same basic computation for horizontal
        numtiles_W = W // patches.shape[3] + 1
        output = torch.concat([patchcolumn for _ in range(numtiles_W)], 3)[:, :, :, :W]
        return output
    
    def _forward_single(self, x, control=False, evaluate=False, params={}, key=None, **kwargs):
        if key is None:
            size = self.size
        elif key in self.size:
            size = self.size[key]
        else:
            return x, kwargs
        return self._tile_single_patch(x, size), kwargs
    
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        if isinstance(self.size, dict):
            tileto = ", ".join([f"{k}: {str(self.size[k])}" for k in self.size])

        else:
            tileto = self.size
        return f"**{self.__class__.__name__}:** tile to {tileto}"


class PatchScroller(PipelineBase):
    """
    Class for translating a patch with toroidal boundary conditions.

    Returns the unchanged patch during evaluation steps.
    """
    def __init__(self, logviz=True, keys=None):
        """
        :keys: optional; list of keys of patches to scroll for multi-patch case
        """
        super().__init__()
        self.params = {}
        self.lastsample = {}
        self._logviz = logviz
        if keys is not None:
            self.params["keys"] = keys

    def _forward_single(self, x, control=False, evaluate=False, params={}, key=None, **kwargs):
        # don't scroll patch during evaluate steps
        if evaluate:
            return x, kwargs
        if key is not None:
            offsetx = f"offset_x_{key}"
            offsety = f"offset_y_{key}"
        else:
            offsetx = "offset_x"
            offsety = "offset_y"

        if not control:
            N, C, H, W = x.shape
            self.lastsample[offsetx] = params.get(offsetx, torch.randint(low=0, high=H, size=[N]))
            self.lastsample[offsety] = params.get(offsety, torch.randint(low=0, high=H, size=[N]))
            
        s = self.lastsample
        shifted = torch.stack(
            [
                scroll_single_image(x[i], s[offsetx][i], s[offsety][i])
                for i in range(x.shape[0])
            ],
            0,
        )
        return shifted, kwargs
       

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {k: [i for i in self.lastsample[k].cpu().detach().numpy()] 
                for k in self.lastsample}
