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
                 eval_boxdict=None, scale=(0.75,1.25), offset_frac_x=None,
                 offset_frac_y=None, mask=None):
        """
        :imagedict: dictionary mapping keys to images, as PIL.Image objects or 3D numpy arrays
        :boxdict: dictionary mapping the same keys to lists of bounding boxes, i.e.
            {"img1":[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]}
        :eval_imagedict: separate dictionary of images to evaluate on
        :eval_boxdict: separate dictionary of lists of bounding boxes for evaluation
        :scale: tuple of floats; range of scaling factors
        :offset_frac_x: None or float between 0 and 1- optionally specify a relative x position within the target box for the patch.
        :offset_frac_y: None or float between 0 and 1- optionally specify a relative y position within the target box for the patch.
        :mask: None, scalar between 0 and 1, or torch.Tensor on the unit interval to use for masking the patch
        """
        super(RectanglePatchImplanter, self).__init__()
        # save training image/box information
        self.imgkeys = list(imagedict.keys())
        self.images = torch.nn.ParameterList([_img_to_tensor(imagedict[k]) for k in self.imgkeys])
        self.boxes = [boxdict[k] for k in self.imgkeys]
        self.mask = mask
        
        # save eval image/box information
        if eval_imagedict is None:
            eval_imagedict = imagedict
            eval_boxdict = boxdict
            
        self.eval_imgkeys = list(eval_imagedict.keys())
        self.eval_images = torch.nn.ParameterList([_img_to_tensor(eval_imagedict[k]) for k in
                                                   self.eval_imgkeys])
        self.eval_boxes = [eval_boxdict[k] for k in self.eval_imgkeys]
            
        
        
        self.params = {"scale":list(scale), "imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys,
                       "offset_frac_x":offset_frac_x,
                       "offset_frac_y":offset_frac_y}
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
        p = self.params
        
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
            self._eval_last = True
        else:
            images = self.images
            boxes = self.boxes
            self._eval_last = False
        
        sampdict = {k:kwargs[k] for k in kwargs}
        if "scale" not in kwargs:
            sampdict["scale"] = torch.FloatTensor(n).uniform_(p["scale"][0], p["scale"][1])
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(images), size=[n])
        if "box" not in kwargs:
            i = torch.tensor([torch.randint(low=0, high=len(boxes[j]), size=[]) for j in sampdict["image"]])
            sampdict["box"] = i
        if "offset_frac_x" not in kwargs:
            if p["offset_frac_x"] is None:
                sampdict["offset_frac_x"] = torch.rand([n])
            else:
                sampdict["offset_frac_x"] = torch.tensor(n*[p["offset_frac_x"]])
        if "offset_frac_y" not in kwargs:
            if p["offset_frac_y"] is None:
                sampdict["offset_frac_y"] = torch.rand([n])
            else:
                sampdict["offset_frac_y"] = torch.tensor(n*[p["offset_frac_y"]])
            
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
    
    def _get_mask(self, patch):
        """
        Input a patch and return a mask, either as a scalar
        or as a (1,H,W) tensor with the same spatial dimensions
        as the patch.

        This method will handle cases where either no mask was
        specified, a scalar mask was specified, or a mask image
        was specified as a 2D or 3D tensor
        """
        # no mask? easy
        if self.mask is None:
            return 1.
        else:
            # if mask is a tensor...
            if isinstance(self._mask, torch.Tensor):
                with torch.no_grad():
                    mask = self.mask.clone().detach()
                    # if mask was specified as 2D, add a batch dimension so
                    # it will broadcast directly
                    if len(mask.shape) == 2:
                        mask = mask.unsqueeze(0)
                    # resize mask to match patch
                    mask = kornia.geometry.resize(mask, (patch.shape[1], patch.shape[2]))
            # scalar mask case
            else:
                mask = self.mask
        return mask


        
    def _implant_patch(self, image, patch, offset_x, offset_y):
        implanted = []
        for i in range(len(image)):
            C, H, W = image[i].shape
            pC, pH, pW = patch[i].shape
        
            imp = image[i].clone().detach()

            # if there's a mask we need to mix the patch with the part of
            # the image it's replacing
            if self.mask is not None:
                # get the corresponding mask
                mask = self._get_mask(patch[i])
                # get a copy of the part of the image we're cutting out
                with torch.no_grad():
                    cutout = imp.clone().detach()[:, offset_y[i]:offset_y[i]+pH, offset_x[i]:offset_x[i]+pW]
                replace = patch[i]*mask + cutout*(1-mask)
            # otherwise we're just replacing with the patch
            else:
                replace = patch[i]
            imp[:, offset_y[i]:offset_y[i]+pH, offset_x[i]:offset_x[i]+pW] = replace
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
        self.sample(patches.shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        if self.params["scale"][1] > self.params["scale"][0]:
            patchlist = [kornia.geometry.rescale(patches[i].unsqueeze(0), (s["scale"][i], s["scale"][i])).squeeze(0) 
                                  for i in range(patches.shape[0])]
        else:
            patchlist = [patches[i] for i in range(patches.shape[0])]
            
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
        if self._eval_last:
            imgkeys = self.eval_imgkeys
        else:
            imgkeys = self.imgkeys
        outdict = {}
        for k in self.lastsample:
            if k == "image":
                outdict["image"] = [imgkeys[i] for i in self.lastsample["image"].cpu().detach().numpy()]
            else:
                outdict[k] = [float(i) for i in self.lastsample[k].cpu().detach().numpy()]
        return outdict
    
    def get_description(self):
        return f"**{self.name}:** {len(self.imgkeys)} training and {len(self.eval_imgkeys)} eval images"
        
    
    
    
    
    
    

class FixedRatioRectanglePatchImplanter(RectanglePatchImplanter):
    """
    Variation on RectanglePatchImplanter that scales the patch to a fixed
    size with respect to each target box.
    
    """
    name = "FixedRatioRectanglePatchImplanter"
    
    def __init__(self, imagedict, boxdict, eval_imagedict=None,
                 eval_boxdict=None, frac=0.5, scale_by="min",
                 offset_frac_x=None, offset_frac_y=None, mask=None):
        """
        :imagedict: dictionary mapping keys to images, as PIL.Image objects or 3D numpy arrays
        :boxdict: dictionary mapping the same keys to lists of bounding boxes, i.e.
            {"img1":[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]}
        :eval_imagedict: separate dictionary of images to evaluate on
        :eval_boxdict: separate dictionary of lists of bounding boxes for evaluation
        :frac: float; relative size
        :scale_by: str; whether to use the "height", "width", of the box, or "min" 
            of the two for scaling
        :offset_frac_x: None or float between 0 and 1- optionally specify a relative x position within the target box for the patch.
        :offset_frac_y: None or float between 0 and 1- optionally specify a relative y position within the target box for the patch.
        :mask: None, scalar between 0 and 1, or torch.Tensor on the unit interval to use for masking the patch
        """
        super(RectanglePatchImplanter, self).__init__()
        # save training image/box information
        self.imgkeys = list(imagedict.keys())
        self.images = torch.nn.ParameterList([_img_to_tensor(imagedict[k]) for k in self.imgkeys])
        self.boxes = [boxdict[k] for k in self.imgkeys]
        self.mask = mask
        
        # save eval image/box information
        if eval_imagedict is None:
            eval_imagedict = imagedict
            eval_boxdict = boxdict
            
        self.eval_imgkeys = list(eval_imagedict.keys())
        self.eval_images = torch.nn.ParameterList([_img_to_tensor(eval_imagedict[k]) for k in
                                                   self.eval_imgkeys])
        self.eval_boxes = [eval_boxdict[k] for k in self.eval_imgkeys]
            
        
        
        self.params = {"frac":frac, "imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys, 
                       "scale_by":scale_by,
                       "offset_frac_x":offset_frac_x,
                       "offset_frac_y":offset_frac_y}
        self.lastsample = {}
        
        assert len(imagedict) == len(boxdict), "should be same number of images and boxes"
        
        
    def sample(self, n, evaluate=False, **kwargs):
        """
        Sample implantation parameters for batch size n, overwriting with
        kwargs if necessary.
        """
        p = self.params
        
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
            self._eval_last = True
        else:
            images = self.images
            boxes = self.boxes
            self._eval_last = False
        
        sampdict = {k:kwargs[k] for k in kwargs}
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(images), size=[n])
        if "box" not in kwargs:
            i = torch.tensor([torch.randint(low=0, high=len(boxes[j]), size=[]) for j in sampdict["image"]])
            sampdict["box"] = i
        if "offset_frac_x" not in kwargs:
            if p["offset_frac_x"] is None:
                sampdict["offset_frac_x"] = torch.rand([n])
            else:
                sampdict["offset_frac_x"] = torch.tensor(n*[p["offset_frac_x"]])
        if "offset_frac_y" not in kwargs:
            if p["offset_frac_y"] is None:
                sampdict["offset_frac_y"] = torch.rand([n])
            else:
                sampdict["offset_frac_y"] = torch.tensor(n*[p["offset_frac_y"]])
            
        self.lastsample = sampdict
        
    
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
        self.sample(patches.shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        
        patchlist = [patches[i] for i in range(patches.shape[0])]
        patchlist= []
            
        # figure out offset of patches
        bs = len(s["box"])
        dx = torch.zeros(bs)
        dy = torch.zeros(bs)
        boxx = torch.zeros(bs)
        boxy = torch.zeros(bs)
        # for each image in the batch
        for i in range(bs):
            box = boxes[s["image"][i]][s["box"][i]]
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            # gotta rescale the patch
            C,H,W = patches[i].shape
            # figure out which axis to scale by
            scale_by = self.params["scale_by"]
            if scale_by == "min":
                if box_width < box_height:
                    scale_by = "width"
                else:
                    scale_by = "height"
            if scale_by == "width":
                factor = self.params["frac"]*box_width/W
            else:
                factor = self.params["frac"]*box_height/H
            patchlist.append(
                kornia.geometry.transform.rescale(patches[i].unsqueeze(0), 
                                                  factor).squeeze(0))
            
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
    
    
    


class ScaleToBoxRectanglePatchImplanter(RectanglePatchImplanter):
    """
    Rectangle patch implanter that resizes the patch to fit the box. Assume 
    all images are the same dimensions.
    
    If eval_imagedict and eval_boxdict aren't passed, evaluation will be done
    one training images/boxes.
    """
    name = "ScaleToBoxRectanglePatchImplanter"
    
    def __init__(self, imagedict, boxdict, eval_imagedict=None,
                 eval_boxdict=None, mask=None):
        """
        :imagedict: dictionary mapping keys to images, as PIL.Image objects or 3D numpy arrays
        :boxdict: dictionary mapping the same keys to lists of bounding boxes, i.e.
            {"img1":[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]}
        :eval_imagedict: separate dictionary of images to evaluate on
        :eval_boxdict: separate dictionary of lists of bounding boxes for evaluation
        :mask: None, scalar between 0 and 1, or torch.Tensor on the unit interval to use for masking the patch
        """
        super(RectanglePatchImplanter, self).__init__()
        # save training image/box information
        self.imgkeys = list(imagedict.keys())
        self.images = torch.nn.ParameterList([_img_to_tensor(imagedict[k]) for k in self.imgkeys])
        self.boxes = [boxdict[k] for k in self.imgkeys]
        self.mask = mask
        
        # save eval image/box information
        if eval_imagedict is None:
            eval_imagedict = imagedict
            eval_boxdict = boxdict
            
        self.eval_imgkeys = list(eval_imagedict.keys())
        self.eval_images = torch.nn.ParameterList([_img_to_tensor(eval_imagedict[k]) for k in
                                                   self.eval_imgkeys])
        self.eval_boxes = [eval_boxdict[k] for k in self.eval_imgkeys]
            
        
        
        self.params = {"imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys}
        self.lastsample = {}
        
        assert len(imagedict) == len(boxdict), "should be same number of images and boxes"
        
        
    def sample(self, n, evaluate=False, **kwargs):
        """
        Sample implantation parameters for batch size n, overwriting with
        kwargs if necessary.
        """
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
            self._eval_last = True
        else:
            images = self.images
            boxes = self.boxes
            self._eval_last = False
        
        sampdict = {k:kwargs[k] for k in kwargs}
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(images), size=[n])
        if "box" not in kwargs:
            i = torch.tensor([torch.randint(low=0, high=len(boxes[j]), size=[]) for j in sampdict["image"]])
            sampdict["box"] = i
            
        self.lastsample = sampdict
        
    
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
        self.sample(patches.shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        
        #if self.params["scale"][1] > self.params["scale"][0]:
        #    patchlist = [kornia.geometry.rescale(patches[i].unsqueeze(0), (s["scale"][i], s["scale"][i])).squeeze(0) 
        #                          for i in range(patches.shape[0])]
        #else:
        #    patchlist = [patches[i] for i in range(patches.shape[0])]
            
        patchlist = []
            
        # figure out offset of patches
        bs = len(s["box"])
        dx = torch.zeros(bs)
        dy = torch.zeros(bs)
        boxx = torch.zeros(bs)
        boxy = torch.zeros(bs)
        for i in range(bs):
            box = boxes[s["image"][i]][s["box"][i]]
            box_h = box[3] - box[1]
            box_w = box[2] - box[0]
            patchlist.append(
                kornia.geometry.resize(patches[i].unsqueeze(0),
                                        (box_h, box_w)).squeeze(0)
                )
            boxx[i] = box[0]
            boxy[i] = box[1]
            
        offset_y = (dy + boxy).type(torch.IntTensor)
        offset_x = (dx + boxx).type(torch.IntTensor)
        images = [images[i] for i in s["image"]]
        
        if control:
            return torch.stack(images,0)
        
        return self._implant_patch(images, patchlist, offset_x, offset_y)
    
    def validate(self, patch):
        """
        we don't actually need to check box sizes since we're rescaling
        """
        logging.warning("nothing to validate here")
    
    
    
        