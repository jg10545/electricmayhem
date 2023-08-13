import numpy as np
import torch
import kornia.geometry
import logging
import matplotlib.pyplot as plt
import matplotlib

from ._util import _img_to_tensor
from ._pipeline import PipelineBase



class RectanglePatchImplanter(PipelineBase):
    """
    Class for adding a patch to an image, with noise
    
    Assume all images are the same dimensions.
    
    Use validate() to make sure all your bounding boxes fit a patch.
    """
    name = "RectanglePatchImplanter"
    
    def __init__(self, imagedict, boxdict, scale=(0.75,1.25)):
        """
        :imagedict: dictionary mapping keys to images, as PIL.Image objects or 3D numpy arrays
        :boxdict: dictionary mapping the same keys to lists of bounding boxes, i.e.
            {"img1":[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]}
        :scale: tuple of floats; range of scaling factors
        """
        self.imgkeys = list(imagedict.keys())
        self.images = [_img_to_tensor(imagedict[k]) for k in self.imgkeys]
        self.boxes = [boxdict[k] for k in self.imgkeys]
        
        self.params = {"scale":list(scale), "imgkeys":self.imgkeys}
        self.lastsample = {}
        
        assert len(imagedict) == len(boxdict), "should be same number of images and boxes"
        
    def sample(self, n, **kwargs):
        """
        Sample implantation parameters for batch size n, overwriting with
        kwargs if necessary.
        """
        sampdict = {k:kwargs[k] for k in kwargs}
        if "scale" not in kwargs:
            sampdict["scale"] = torch.FloatTensor(n).uniform_(self.params["scale"][0], self.params["scale"][1])
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(self.images), size=[n])
        if "box" not in kwargs:
            i = torch.tensor([torch.randint(low=0, high=len(self.boxes[j]), size=[]) for j in sampdict["image"]])
            sampdict["box"] = i
        if "offset_frac_x" not in kwargs:
            sampdict["offset_frac_x"] = torch.rand([n])
        if "offset_frac_y" not in kwargs:
            sampdict["offset_frac_y"] = torch.rand([n])
            
        self.lastsample = sampdict
        
    def validate(self, patch):
        """
        Check to see whether any of your patch/scale/image/box combinations could throw an error
        """
        all_validated = True
        max_y = int(self.params["scale"][1]*patch.shape[1])
        max_x = int(self.params["scale"][1]*patch.shape[2])
        
        for i in range(len(self.images)):
            for j in range(len(self.boxes[i])):
                b = self.boxes[i][j]
                dy = b[3] - b[1]
                dx = b[2] - b[0]
                if (max_y >= dy)|(max_x >= dx):
                    logging.warning(f"box {j} of image {self.imgkeys[i]} is too small for this patch and scale")
                    all_validated = False
        return all_validated
        
    def _implant_patch(self, image, patch, offset_x, offset_y):
        implanted = []
        for i in range(len(image)):
            C, H, W = image[i].shape
            pC, pH, pW = patch[i].shape
        
            imp = image[i].clone().detach()
            imp[:, offset_y[i]:offset_y[i]+pH, offset_x[i]:offset_x[i]+pW] = patch[i]
            implanted.append(imp)
        
        return torch.stack(implanted,0)
    
    def apply(self, patches, dont_implant=False, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :dont_implant: if True, leave the patches off (for diagnostics)
        :kwargs: passed to self.sample()
        """
        # sample parameters if necessary
        self.sample(patches.shape[0], **kwargs)
        s = self.lastsample
        if self.params["scale"][1] > self.params["scale"][0]:
            patchlist = [kornia.geometry.rescale(patches[i].unsqueeze(0), (s["scale"][i], s["scale"][i])).squeeze(0) 
                                  for i in range(patches.shape[0])]
            
        # figure out offset of patches
        bs = len(s["box"])
        dx = torch.zeros(bs)
        dy = torch.zeros(bs)
        boxx = torch.zeros(bs)
        boxy = torch.zeros(bs)
        for i in range(bs):
            box = self.boxes[s["image"][i]][s["box"][i]]
            dy[i] = box[3] - box[1] - patchlist[i].shape[1]
            dx[i] = box[2] - box[0] - patchlist[i].shape[2]
            boxx[i] = box[0]
            boxy[i] = box[1]
            
        offset_y = (dy*s["offset_frac_y"] + boxy).type(torch.IntTensor)
        offset_x = (dx*s["offset_frac_x"] + boxx).type(torch.IntTensor)
        images = [self.images[i] for i in s["image"]]
        
        if dont_implant:
            return torch.stack(images,0)
        
        return self._implant_patch(images, patchlist, offset_x, offset_y)
    
    def plot_boxes(self):
        """
        Quick visualization with matplotlib of the victim images and box regions
        """
        n = len(self.images)
        d = int(np.ceil(np.sqrt(n)))
        fig, axs = plt.subplots(nrows=d, ncols=d, squeeze=False)

        i = 0
        for axrow in axs:
            for ax in axrow:
                ax.set_axis_off()
                if i < n:
                    ax.imshow((self.images[i].permute(1,2,0).numpy()))
                    
                    for j in range(len(self.boxes[i])):
                        b = self.boxes[i][j]
                        xw = (b[0], b[1])
                        width = b[2]-b[0]
                        height = b[3]-b[1]
                        rect = matplotlib.patches.Rectangle(xw, width, height, linewidth=2, fill=False, color="r")
                        ax.add_artist(rect)
                        ax.text(xw[0], xw[1], str(j))
                    ax.set_title(self.imgkeys[i])
                
                i += 1
        return fig
        