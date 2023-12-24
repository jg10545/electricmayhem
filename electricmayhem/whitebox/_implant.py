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
    Class for adding a patch to an image, with noise. Assume all images are 
    the same dimensions.
    
    Use validate() to make sure all your bounding boxes fit a patch.
    
    If eval_imagedict and eval_boxdict aren't passed, evaluation will be done
    one training images/boxes.
    """
    name = "RectanglePatchImplanter"
    
    def __init__(self, imagedict, boxdict, eval_imagedict=None,
                 eval_boxdict=None, scale=(0.75,1.25)):
        """
        :imagedict: dictionary mapping keys to images, as PIL.Image objects or 3D numpy arrays
        :boxdict: dictionary mapping the same keys to lists of bounding boxes, i.e.
            {"img1":[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]}
        :eval_imagedict: separate dictionary of images to evaluate on
        :eval_boxdict: separate dictionary of lists of bounding boxes for evaluation
        :scale: tuple of floats; range of scaling factors
        """
        super(RectanglePatchImplanter, self).__init__()
        # save training image/box information
        self.imgkeys = list(imagedict.keys())
        self.images = torch.nn.ParameterList([_img_to_tensor(imagedict[k]) for k in self.imgkeys])
        self.boxes = [boxdict[k] for k in self.imgkeys]
        
        # save eval image/box information
        if eval_imagedict is None:
            eval_imagedict = imagedict
            eval_boxdict = boxdict
            
        self.eval_imgkeys = list(eval_imagedict.keys())
        self.eval_images = torch.nn.ParameterList([_img_to_tensor(eval_imagedict[k]) for k in
                                                   self.eval_imgkeys])
        self.eval_boxes = [eval_boxdict[k] for k in self.eval_imgkeys]
            
        
        
        self.params = {"scale":list(scale), "imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys}
        self.lastsample = {}
        
        assert len(imagedict) == len(boxdict), "should be same number of images and boxes"
        
    def get_min_dimensions(self):
        """
        Find the minimum height and width of any training/eval box
        """
        minheight = 1e6
        minwidth = 1e6
        
        for boxes in self.boxes+self.eval_boxes:
            for b in boxes:
                minheight = min(minheight, b[3]-b[1])
                minwidth = min(minwidth, b[2]-b[0])
        return {"minheight":minheight, "minwidth":minwidth}
        
    def sample(self, n, evaluate=False, **kwargs):
        """
        Sample implantation parameters for batch size n, overwriting with
        kwargs if necessary.
        """
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes
        
        sampdict = {k:kwargs[k] for k in kwargs}
        if "scale" not in kwargs:
            sampdict["scale"] = torch.FloatTensor(n).uniform_(self.params["scale"][0], self.params["scale"][1])
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(images), size=[n])
        if "box" not in kwargs:
            i = torch.tensor([torch.randint(low=0, high=len(boxes[j]), size=[]) for j in sampdict["image"]])
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
                    
        for i in range(len(self.eval_images)):
            for j in range(len(self.eval_boxes[i])):
                b = self.eval_boxes[i][j]
                dy = b[3] - b[1]
                dx = b[2] - b[0]
                if (max_y >= dy)|(max_x >= dx):
                    logging.warning(f"box {j} of eval image {self.imgkeys[i]} is too small for this patch and scale")
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
    
    def forward(self, patches, control=False, evaluate=False, params={}, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches
        :control: if True, leave the patches off (for diagnostics)
        :params: dictionary of params to override random sampling
        :kwargs: passed to self.sample()
        """
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes
        
        if control:
            params = self.lastsample
        # sample parameters if necessary
        #self.sample(patches.shape[0], control=control, **params)
        self.sample(patches.shape[0], evaluate=evaluate, **params)
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
            box = boxes[s["image"][i]][s["box"][i]]
            dy[i] = box[3] - box[1] - patchlist[i].shape[1]
            dx[i] = box[2] - box[0] - patchlist[i].shape[2]
            boxx[i] = box[0]
            boxy[i] = box[1]
            
        offset_y = (dy*s["offset_frac_y"] + boxy).type(torch.IntTensor)
        offset_x = (dx*s["offset_frac_x"] + boxx).type(torch.IntTensor)
        images = [images[i] for i in s["image"]]
        
        if control:
            return torch.stack(images,0)
        
        return self._implant_patch(images, patchlist, offset_x, offset_y)
    
    def plot_boxes(self, evaluate=False):
        """
        Quick visualization with matplotlib of the victim images and box regions
        """
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
            imgkeys = self.eval_imgkeys
        else:
            images = self.images
            boxes = self.boxes
            imgkeys = self.imgkeys
            
        n = len(images)
        d = int(np.ceil(np.sqrt(n)))
        fig, axs = plt.subplots(nrows=d, ncols=d, squeeze=False)

        i = 0
        for axrow in axs:
            for ax in axrow:
                ax.set_axis_off()
                if i < n:
                    ax.imshow((images[i].permute(1,2,0).detach().cpu().numpy()))
                    
                    for j in range(len(boxes[i])):
                        b = boxes[i][j]
                        xw = (b[0], b[1])
                        width = b[2]-b[0]
                        height = b[3]-b[1]
                        rect = matplotlib.patches.Rectangle(xw, width, height, linewidth=2, fill=False, color="r")
                        ax.add_artist(rect)
                        ax.text(xw[0], xw[1], str(j))
                    ax.set_title(imgkeys[i])
                
                i += 1
        return fig
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for k in self.lastsample:
            if k == "image":
                outdict["image"] = [self.imgkeys[i] for i in self.lastsample["image"].cpu().detach().numpy()]
        else:
            outdict[k] = [float(i) for i in self.lastsample[k].cpu().detach().numpy()]
        return outdict
    
    def get_description(self):
        return f"**{self.name}:** {len(self.imgkeys)} training and {len(self.eval_imgkeys)} eval images"
        