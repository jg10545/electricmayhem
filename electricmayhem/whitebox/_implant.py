import numpy as np
import torch
import kornia.geometry
import logging
import matplotlib.pyplot as plt
import matplotlib
import mlflow, mlflow.data

from .._convenience import load_to_tensor
from ._util import _img_to_tensor
from ._pipeline import PipelineBase

def get_pixel_offsets_from_fractional_offsets(s, boxes, patchlist, key="patch"):
    """
    Convert offsets on the unit interval into the actual pixel coordinates for
    the corners of a batch of patches

    :s: dictionary of sampled parameters
    :boxes: list of lists of box pixel coordinates
    :patchlist: batch of tensors as a list
    :key:
    """
    # figure out offset of patches
    bs = len(s[f"box_{key}"])
    dx = torch.zeros(bs)
    dy = torch.zeros(bs)
    boxx = torch.zeros(bs)
    boxy = torch.zeros(bs)
    # for every element in the batch
    for i in range(bs):
        # get coordinates of the box we have to place the patch in
        box = boxes[s["image"][i]][key][s[f"box_{key}"][i].item()]
        # vertical range is the height of the box minus height of the patch
        dy[i] = box[3] - box[1] - patchlist[i].shape[1]
        # horizontal range is with width of the box minus width of the patch
        dx[i] = box[2] - box[0] - patchlist[i].shape[2]
        boxx[i] = box[0]
        boxy[i] = box[1]
            
    offset_y = (dy*s.get(f"offset_frac_y_{key}", 0) + boxy).type(torch.IntTensor)
    offset_x = (dx*s.get(f"offset_frac_x_{key}", 0) + boxx).type(torch.IntTensor)
    
    return offset_x, offset_y

def _unpack_rectangle_frame(df):
    """
    Unpack one subset of a rectangle dataframe
    """
    img_keys = list(df["image"].unique())
    images = [load_to_tensor(i) for i in img_keys]
    boxes = []
    for i in img_keys:
        subset = df[df["image"] == i]
        img_boxes = {}
        for p in subset["patch"].unique():
            s = subset[subset["patch"] == p]
            img_boxes[p] = [[r.xmin, r.ymin, r.xmax, r.ymax] for e,r in s.iterrows()]
        boxes.append(img_boxes)
    return img_keys, images, boxes



def _unpack_rectangle_dataset(df):
    """
    Assumes dataframe has columns:
        -image: path to image file
        -xmin, mxax, ymin, ymax: box coordinates

    And may have columns:
        -split: "train" or "eval"
        -patch: string giving patch key
    """
    df = df.copy()
    # add a patch column if needed
    if "patch" not in df.columns:
        df["patch"] = "patch"
    patch_keys = list(df["patch"].unique())

    # if no train/eval split
    if "split" not in df.columns:
        logging.warning("no 'split' column found in dataset; using same images for train and eval")
        img_keys, images, boxes = _unpack_rectangle_frame(df)
        return df, patch_keys, img_keys, images, boxes, img_keys, images, boxes
    
    else:
        img_keys_train, images_train, boxes_train = _unpack_rectangle_frame(df[df["split"]=="train"])
        img_keys_eval, images_eval, boxes_eval = _unpack_rectangle_frame(df[df["split"]=="eval"])
        return df, patch_keys, img_keys_train, images_train, boxes_train, img_keys_eval, images_eval, boxes_eval
    
def _prep_masks(patch_keys, mask):
    if isinstance(mask,dict):
        for k in patch_keys:
            assert k in mask, f"patch key {k} not in mask dict"
        assert len(patch_keys) == len(mask), "mask and patch keys don't line up"
        return mask
    else:
        return {k:mask for k in patch_keys}

class RectanglePatchImplanter(PipelineBase):
    """
    Class for adding a patch to an image, with noise. Assume all images are 
    the same dimensions.
    
    Use validate() to make sure all your bounding boxes fit a patch.
    
    If eval_imagedict and eval_boxdict aren't passed, evaluation will be done
    one training images/boxes.
    """
    name = "RectanglePatchImplanter"
    def __init__(self, df, scale=(0.75,1.25), offset_frac_x=None,
                 offset_frac_y=None, mask=1, scale_brightness=False, dataset_name=None):
        """
        :df: dataframe containing an "image" column with paths to images, xmin/ymin/xmax/ymax columns
            giving box coordinates, and (optionally) "patch" column (specifying which patch name the
            box is for) and "split" column (train/eval)
        :scale: tuple of floats; range of scaling factors
        :offset_frac_x: None or float between 0 and 1- optionally specify a relative x position within the target box for the patch.
        :offset_frac_y: None or float between 0 and 1- optionally specify a relative y position within the target box for the patch.
        :mask: alpha value between 0 and 1, torch.Tensor on the unit interval to use for masking the patch,
            or dictionary of values or tensors for masking different patches
        :scale_brightness: if True, adjust brightness of patch to match the average brightness of the section of
            image it's replacing
        :dataset_name: None or str; name of dataset to be logged in mlflow
        """
        super(RectanglePatchImplanter, self).__init__()

        df, patch_keys, img_keys_train, images_train, boxes_train, img_keys_eval, images_eval, boxes_eval = _unpack_rectangle_dataset(df)
        self.df = df
        self.patch_keys = patch_keys
        # save training image/box information
        self.imgkeys = img_keys_train
        self.images = torch.nn.ParameterList(images_train)
        self.boxes = boxes_train

        self.eval_imgkeys = img_keys_eval
        self.eval_images = torch.nn.ParameterList(images_eval)
        self.eval_boxes = boxes_eval

        # MASK STUFF
        self.mask = torch.nn.ParameterDict(_prep_masks(patch_keys, mask))

        self._dataset_name = dataset_name
        self._sample_scale = True # whether self.sample() should sample a random scaling factor for patch
        self._sample_offsets = True # whether self.sample() should sample offsets relative to the box
        
        self.params = {"scale":list(scale), "imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys,
                       "offset_frac_x":offset_frac_x,
                       "offset_frac_y":offset_frac_y,
                       "mask":self._get_mask_summary(mask),
                       "scale_brightness":scale_brightness}
        self.lastsample = {}

    def _get_mask_summary(self, mask):
        """
        get a mask type description, and make sure it's in the unit interval
        """
        if mask is None:
            return 1
        elif isinstance(mask, torch.Tensor):
            assert torch.max(mask) <= 1 and torch.min(mask) >= 0, "mask should be between 0 and 1"
            return "tensor"
        elif isinstance(mask, dict):
            for k in mask:
                if isinstance(mask[k], torch.Tensor):
                    assert torch.max(mask[k]) <= 1 and torch.min(mask[k]) >= 0, "mask should be between 0 and 1"
                else:
                    assert mask[k] >= 0 and mask[k] <= 1, "mask should be between 0 and 1"
                return "dict"
        else:
            assert mask >= 0 and mask <= 1, "mask should be between 0 and 1"
            return mask
        
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
        
        # target image is sampled once per batch element
        if "image" not in kwargs:
            sampdict["image"] = torch.randint(low=0, high=len(images), size=[n])
        # scale, box, and offsets are sampled separately for each patch being implanted
        for k in self.patch_keys:
            if self._sample_scale:
                if f"scale_{k}" not in kwargs:
                    sampdict[f"scale_{k}"] = torch.FloatTensor(n).uniform_(p["scale"][0], p["scale"][1])

            if f"box_{k}" not in kwargs:
                i = torch.tensor([torch.randint(low=0, high=len(boxes[j][k]), size=[]) for j in sampdict["image"]])
                sampdict[f"box_{k}"] = i
            if self._sample_offsets:
                if f"offset_frac_x_{k}" not in kwargs:
                    if p["offset_frac_x"] is None:
                        sampdict[f"offset_frac_x_{k}"] = torch.rand([n])
                    else:
                        sampdict[f"offset_frac_x_{k}"] = torch.tensor(n*[p["offset_frac_x"]])
                if f"offset_frac_y_{k}" not in kwargs:
                    if p["offset_frac_y"] is None:
                        sampdict[f"offset_frac_y_{k}"] = torch.rand([n])
                    else:
                        sampdict[f"offset_frac_y_{k}"] = torch.tensor(n*[p["offset_frac_y"]])
            
        self.lastsample = sampdict
        
    def validate(self, patch):
        """
        Check to see whether any of your patch/scale/image/box combinations could throw an error
        """
        # if passed a single patch batch, wrap it in a dictionary
        if isinstance(patch, torch.Tensor):
            patch = {"patch":patch}

        all_validated = True
        
        # check each patch
        for k in patch:
            if "scale" in self.params:
                scale = self.params["scale"]
            else:
                scale = (1,1)
            max_y = int(scale[1]*patch[k].shape[1])
            max_x = int(scale[1]*patch[k].shape[2])
            # in each image
            for i in range(len(self.images)):
                # in each box defined for that patch in that image
                for j in range(len(self.boxes[i][k])):
                    b = self.boxes[i][k][j]
                    dy = b[3] - b[1]
                    dx = b[2] - b[0]
                    if (max_y >= dy)|(max_x >= dx):
                        logging.warning(f"{self.name}: box {j} of image {self.imgkeys[i]} is too small for patch {k} and scale {self.params['scale'][1]}")
                        all_validated = False
                    
            for i in range(len(self.eval_images)):
                for j in range(len(self.eval_boxes[i][k])):
                    b = self.eval_boxes[i][k][j]
                    dy = b[3] - b[1]
                    dx = b[2] - b[0]
                    if (max_y >= dy)|(max_x >= dx):
                        logging.warning(f"{self.name}: box {j} of eval image {self.imgkeys[i]} is too small for patch {k} and scale {self.params['scale'][1]}")
                        all_validated = False
        return all_validated
    
    def _get_mask(self, patch):  # DEPRECATE??
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


        
    def _implant_patch(self, image, patch, offset_x, offset_y, mask, scale_brightness=False):
        implanted = []
        for i in range(len(image)):
            C, H, W = image[i].shape
            pC, pH, pW = patch[i].shape
            
            imp = image[i].clone().detach()
            # get a copy of the part of the image we're cutting out
            with torch.no_grad():
                cutout = imp.clone().detach()[:, offset_y[i]:offset_y[i]+pH, offset_x[i]:offset_x[i]+pW]
            if scale_brightness:
                with torch.no_grad():
                    scale = torch.mean(cutout)/torch.mean(patch[i])
            else:
                scale = 1

            # if there's a mask we need to mix the patch with the part of
            # the image it's replacing
            #if self.mask is not None:
            if mask is not None:
                # get the corresponding mask
                #mask = self._get_mask(patch[i])
                if isinstance(mask, torch.Tensor):
                    with torch.no_grad():
                        mask = kornia.geometry.resize(mask, (patch[i].shape[1], patch[i].shape[2]))
                replace = scale*patch[i]*mask + cutout*(1-mask)
            # otherwise we're just replacing with the patch
            else:
                replace = scale*patch[i]
            imp[:, offset_y[i]:offset_y[i]+pH, offset_x[i]:offset_x[i]+pW] = replace
            implanted.append(imp)
        
        return implanted
    
    def forward(self, patches, control=False, evaluate=False, params={}, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches or a dictionary of patch batches
        :control: if True, leave the patches off (for diagnostics)
        :params: dictionary of params to override random sampling
        :kwargs: passed to self.sample()
        """
        # if passed a single patch batch, wrap it in a dictionary
        if isinstance(patches, torch.Tensor):
            patches = {"patch":patches}

        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes
        
        if control:
            params = self.lastsample
        # sample parameters if necessary
        self.sample(patches[self.patch_keys[0]].shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        # expand each patch batch into a list of tensors, resizing each separately if necessary
        if self.params["scale"][1] > self.params["scale"][0]:
            patchlist = {k:[kornia.geometry.rescale(patches[k][i].unsqueeze(0), 
                                                    (s[f"scale_{k}"][i], s[f"scale_{k}"][i])).squeeze(0) 
                                  for i in range(patches[k].shape[0])] for k in patches}
        else:
            patchlist = {k:[patches[k][i] for i in range(patches[k].shape[0])] for k in patches}

        images = [images[i] for i in s["image"]]
        if control:
            return torch.stack(images,0), kwargs

        # iterate over patches, implanting one at a time
        for k in patches:
            offset_x, offset_y = get_pixel_offsets_from_fractional_offsets(s, boxes, patchlist[k], k)
            images = self._implant_patch(images, patchlist[k], offset_x, offset_y, 
                                         self.mask[k], self.params["scale_brightness"])

        return torch.clamp(torch.stack(images,0), 0, 1), kwargs
    
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
                    
                    for k in self.patch_keys:
                        for j in range(len(boxes[i][k])):
                            b = boxes[i][k][j]
                            xw = (b[0], b[1])
                            width = b[2]-b[0]
                            height = b[3]-b[1]
                            rect = matplotlib.patches.Rectangle(xw, width, height, linewidth=2, fill=False, color="r")
                            ax.add_artist(rect)
                            ax.text(xw[0], xw[1], f"{k} {j}")
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
        return f"**{self.name}:** {len(self.imgkeys)} training and {len(self.eval_imgkeys)} eval images"#, mask: {mask_desc}"
        
    def log_params_to_mlflow(self):
        """
        In addition to logging whatever's in self.params, if there's a tensor mask we
        should keep that someplace
        """
        if hasattr(self, "df"):
            mlflow.log_input(mlflow.data.from_pandas(self.df, name=self._dataset_name))
    
    

class FixedRatioRectanglePatchImplanter(RectanglePatchImplanter):
    """
    Variation on RectanglePatchImplanter that scales the patch to a fixed
    size with respect to each target box.
    
    """
    name = "FixedRatioRectanglePatchImplanter"
    def __init__(self, df, frac, scale_by="min", offset_frac_x=None,
                 offset_frac_y=None, mask=1, scale_brightness=False, dataset_name=None):
        """
        :df: dataframe containing an "image" column with paths to images, xmin/ymin/xmax/ymax columns
            giving box coordinates, and (optionally) "patch" column (specifying which patch name the
            box is for) and "split" column (train/eval)
        :frac: float; relative size
        :scale_by: str; whether to use the "height", "width", of the box, or "min" 
            of the two for scaling
        :offset_frac_x: None or float between 0 and 1- optionally specify a relative x position within the target box for the patch.
        :offset_frac_y: None or float between 0 and 1- optionally specify a relative y position within the target box for the patch.
        :mask: None, scalar between 0 and 1, or torch.Tensor on the unit interval to use for masking the patch
        :scale_brightness: if True, adjust brightness of patch to match the average brightness of the section of
            image it's replacing
        :dataset_name: None or str; name of dataset to be logged in mlflow
        """
        super(RectanglePatchImplanter, self).__init__()
        df, patch_keys, img_keys_train, images_train, boxes_train, img_keys_eval, images_eval, boxes_eval = _unpack_rectangle_dataset(df)
        self.df = df
        self.patch_keys = patch_keys
        # save training image/box information
        self.imgkeys = img_keys_train
        self.images = torch.nn.ParameterList(images_train)
        self.boxes = boxes_train

        self.eval_imgkeys = img_keys_eval
        self.eval_images = torch.nn.ParameterList(images_eval)
        self.eval_boxes = boxes_eval

        # MASK STUFF
        self.mask = torch.nn.ParameterDict(_prep_masks(patch_keys, mask))

        self._dataset_name = dataset_name
        self._sample_scale = False # scaling factor doesn't need to be sampled independently for this case
        self._sample_offsets = True # whether self.sample() should sample offsets relative to the box
        
        self.params = {"frac":frac, "imgkeys":self.imgkeys, "scale_by":scale_by,
                       "eval_imgkeys":self.eval_imgkeys,
                       "offset_frac_x":offset_frac_x,
                       "offset_frac_y":offset_frac_y,
                       "mask":self._get_mask_summary(mask),
                       "scale_brightness":scale_brightness}
        self.lastsample = {}
        
    def _add_patch_scales_to_sampdict(self, s, patches, evaluate=False):
        """
        For this implanter, once we've sampled the target image for each batch element, and the box for each
        patch and box element, then amount we have to scale each patch for each batch element is deterministic.

        This function works out how much to resize each patch to hit a fixed fraction of the box size for every
        patch in every batch element. It just shoehorns that info back into the sampled dictionary, which might not
        be the most elegant approach.

        :s: dictionary of sampled parameters created by self.sample(); should be missing elements like "scale_patch"
        :patches: dictionary of patches (which we'll use to get the unscaled dimensions)
        :evalute: bool; whether this is an evaluate step
        """
        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes

        bs = len(s["image"])
        all_patch_scales = {}
        for k in self.patch_keys:
            N,C,H,W = patches[k].shape
            patch_scales = []
            for i in range(bs):
                box = boxes[s["image"][i]][k][s[f"box_{k}"][i].item()]
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
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
                patch_scales.append(factor)
            s[f"scale_{k}"] = torch.Tensor(patch_scales) # does this need to be a tensor or could I 
            # leave as a list?

    def forward(self, patches, control=False, evaluate=False, params={}, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches or a dictionary of patch batches
        :control: if True, leave the patches off (for diagnostics)
        :params: dictionary of params to override random sampling
        :kwargs: passed to self.sample()
        """
        # if passed a single patch batch, wrap it in a dictionary
        if isinstance(patches, torch.Tensor):
            patches = {"patch":patches}

        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes
        
        if control:
            params = self.lastsample
        # sample parameters if necessary
        self.sample(patches[self.patch_keys[0]].shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        
        images = [images[i] for i in s["image"]]
        if control:
            return torch.stack(images,0), kwargs
        
        # for this implanter the patch scales are deterministic
        self._add_patch_scales_to_sampdict(s, patches, evaluate=evaluate)
        patchlist = {k:[kornia.geometry.rescale(patches[k][i].unsqueeze(0), 
                                                    (s[f"scale_{k}"][i], s[f"scale_{k}"][i])).squeeze(0) 
                                  for i in range(patches[k].shape[0])] for k in patches}
        
        # iterate over patches, implanting one at a time
        for k in patches:
            offset_x, offset_y = get_pixel_offsets_from_fractional_offsets(s, boxes, patchlist[k], k)
            images = self._implant_patch(images, patchlist[k], offset_x, offset_y, 
                                         self.mask[k], self.params["scale_brightness"])

        return torch.clamp(torch.stack(images,0), 0, 1), kwargs
    
    


class ScaleToBoxRectanglePatchImplanter(RectanglePatchImplanter):
    """
    Rectangle patch implanter that resizes the patch to fit the box. Assume 
    all images are the same dimensions.
    
    If eval_imagedict and eval_boxdict aren't passed, evaluation will be done
    one training images/boxes.
    """
    name = "ScaleToBoxRectanglePatchImplanter"
    def __init__(self, df, mask=1, scale_brightness=False, dataset_name=None):
        """
        :df: dataframe containing an "image" column with paths to images, x and y coordinates for columns
            giving box coordinates, and (optionally) "patch" column (specifying which patch name the
            box is for) and "split" column (train/eval)
        :scale: tuple of floats; range of scaling factors
        :offset_frac_x: None or float between 0 and 1- optionally specify a relative x position within the target box for the patch.
        :offset_frac_y: None or float between 0 and 1- optionally specify a relative y position within the target box for the patch.
        :mask: alpha value between 0 and 1, torch.Tensor on the unit interval to use for masking the patch,
            or dictionary of values or tensors for masking different patches
        :scale_brightness: if True, adjust brightness of patch to match the average brightness of the section of
            image it's replacing
        :dataset_name: None or str; name of dataset to be logged in mlflow
        """
        super(RectanglePatchImplanter, self).__init__()

        df, patch_keys, img_keys_train, images_train, boxes_train, img_keys_eval, images_eval, boxes_eval = _unpack_rectangle_dataset(df)
        self.df = df
        self.patch_keys = patch_keys
        # save training image/box information
        self.imgkeys = img_keys_train
        self.images = torch.nn.ParameterList(images_train)
        self.boxes = boxes_train

        self.eval_imgkeys = img_keys_eval
        self.eval_images = torch.nn.ParameterList(images_eval)
        self.eval_boxes = boxes_eval

        # MASK STUFF
        self.mask = torch.nn.ParameterDict(_prep_masks(patch_keys, mask))

        self._dataset_name = dataset_name
        self._sample_scale = False # don't need to sample a scaling factor
        self._sample_offsets = False # or x/y offsets
        
        self.params = {"imgkeys":self.imgkeys,
                       "eval_imgkeys":self.eval_imgkeys,
                       "mask":self._get_mask_summary(mask),
                       "scale_brightness":scale_brightness}
        self.lastsample = {}
        
    
    def forward(self, patches, control=False, evaluate=False, params={}, **kwargs):
        """
        Implant a batch of patches in a batch of images
        
        :patches: torch Tensor; stack of patches or a dictionary of patch batches
        :control: if True, leave the patches off (for diagnostics)
        :params: dictionary of params to override random sampling
        :kwargs: passed to self.sample()
        """
        # if passed a single patch batch, wrap it in a dictionary
        if isinstance(patches, torch.Tensor):
            patches = {"patch":patches}

        if evaluate:
            images = self.eval_images
            boxes = self.eval_boxes
        else:
            images = self.images
            boxes = self.boxes
        
        if control:
            params = self.lastsample
        # sample parameters if necessary
        self.sample(patches[self.patch_keys[0]].shape[0], evaluate=evaluate, **params)
        s = self.lastsample
        
        images = [images[i] for i in s["image"]]
        if control:
            return torch.stack(images,0), kwargs
        
        # for this implanter the patch scales are deterministic. for each
        # patch, for each batch element, figure out what shape to resize
        # it to
        patchlist = {}
        bs = len(images)
        for k in patches:
            patchbatch = []
            for i in range(bs):
                box = self.boxes[s["image"][i]][k][s[f"box_{k}"][i]]
                box_h = box[3] - box[1]
                box_w = box[2] - box[0]
                patchbatch.append(
                    kornia.geometry.resize(patches[k][i].unsqueeze(0),
                                           (box_h, box_w)).squeeze(0))
            patchlist[k] = patchbatch
        
        # iterate over patches, implanting one at a time
        for k in patches:
            offset_x, offset_y = get_pixel_offsets_from_fractional_offsets(s, boxes, patchlist[k], k)
            images = self._implant_patch(images, patchlist[k], offset_x, offset_y,
                                         self.mask[k], self.params["scale_brightness"])

        return torch.clamp(torch.stack(images,0), 0, 1), kwargs
        
    
    def validate(self, patch):
        """
        we don't actually need to check box sizes since we're rescaling
        """
        logging.warning("nothing to validate here")
    
    
    
        