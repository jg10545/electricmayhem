import torch
import kornia.geometry.transform

from ._implant import RectanglePatchImplanter

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
    patch_border = torch.stack([patch_border for _ in range(patch_batch.shape[0])],0).to(patch_batch.device) # (B,4,2)
    
    tfm = kornia.geometry.transform.get_perspective_transform(patch_border, coord_batch) # (B,3,3)
    # apply transformation to get a warped patch with green background
    warped_patch = kornia.geometry.transform.warp_perspective(patch_batch, tfm,
                                                      (target_batch.shape[2], target_batch.shape[3]),
                                                      padding_mode="fill", 
                                                      fill_value=torch.tensor([0,1,0])) # (B,C,H,W)
    # use the green background to create a mask for deciding where to overwrite the target image
    # with the patch
    warpmask = ((warped_patch[:,0,:,:] == 0)&(warped_patch[:,1,:,:] == 1)&(warped_patch[:,2,:,:] == 0)).type(torch.float32) # (B,H,W)
    warpmask = warpmask.unsqueeze(1) # (B,1,H,W)
    # so every place where warpmask=1 will be the target image; every place where it's 0 will be the patch
    
    if mask is not None:
        with torch.no_grad():
            # IMAGE MASK CASE
            if isinstance(mask, torch.Tensor):
                # apply same transforms to batch of masks, but fill with zeros. patch will only show through
                # in places where mask > 0
                mask_pw = kornia.geometry.transform.warp_perspective(mask, tfm,
                                                      (target_batch.shape[2], target_batch.shape[3]),
                                                      padding_mode="zeros") # (B,C,H,W) or (1,1,H,W)
                # update the warp mask to exclude the patch wherever the mask is zero
                warpmask = (warpmask + (1-mask_pw)).clip(0,1)
            # SCALAR MASK CASE
            else:
                # we want every place where warpmask=1 to still be 1.
                # we want every place where warpmask=0 to be the scalar mask value
                # mask=1 should leave warpmask unchanged and mask=0 should be 1 everywhere
                warpmask = warpmask*mask + 1 - mask

    
    return target_batch*warpmask + warped_patch*(1-warpmask)




class WarpPatchImplanter(RectanglePatchImplanter):
    """
    Class for adding a patch to an image, with arbitrary corners that may be warped from
    a rectangle. Assume all target images are the same dimensions.
    
    
    If eval_imagedict and eval_boxdict aren't passed, evaluation will be done
    one training images/boxes.
    """
    name = "WarpPatchImplanter"
    
    def __init__(self, imagedict, boxdict, eval_imagedict=None,
                 eval_boxdict=None, mask=None):
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
        #super(RectanglePatchImplanter, self).__init__()
        super().__init__(imagedict, boxdict, eval_imagedict=eval_imagedict,
                         eval_boxdict=eval_boxdict, mask=mask)

        """
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
                       "offset_frac_y":offset_frac_y,
                       "mask":self._get_mask_summary(mask)}
        self.lastsample = {}
        
        assert len(imagedict) == len(boxdict), "should be same number of images and boxes"
        """
        # some of the parameters in the parent class aren't used here- let's remove some clutter
        for key in ["scale", "offset_frac_x", "offset_frac_y"]:
            if key in self.params:
                del(self.params[key])



    def get_min_dimensions(self):
        assert False, "not implemented for this implanter"
        
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

        self.lastsample = sampdict
        
    def validate(self, patch):
        """
        Check to see whether any of your patch/scale/image/box combinations could throw an error
        """
        logging.info("nothing implemented for WarpPatchImplanter yet")
        return True
    
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
            if isinstance(self.mask, torch.Tensor):
                with torch.no_grad():
                    mask = self.mask.clone().detach().type(torch.float32)
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

        # dictionary of sampled parameters
        s = self.lastsample

        # get mask
        mask = self._get_mask(patches)

        # build a batch of target images
        target_images = torch.stack([images[s["image"][i]] 
                                     for i in range(patches.shape[0])], 0).to(patches.device)
        
        # if it's a control batch, skip the implanting step
        if control:
            return target_images

        # build a batch of box coordinates
        coords = torch.stack([torch.tensor(boxes[s["image"][i]][s["box"][i]]).type(torch.float32)
                              for i in range(patches.shape[0]) ],0).to(patches.device)

        # implant patch
        implanted_images = warp_and_implant_batch(patches, target_images, coords, mask=mask)
        #implanted_images = warp_and_implant_batch(target_images, coords, patches, mask=mask)

        return implanted_images
    
    def plot_boxes(self, evaluate=False):
        """
        Quick visualization with matplotlib of the victim images and box regions
        """
        assert False, "not yet implemented"
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
    