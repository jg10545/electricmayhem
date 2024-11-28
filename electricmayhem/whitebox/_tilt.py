import numpy as np
import torch
import kornia.geometry

from electricmayhem.whitebox._pipeline import PipelineBase

def sample_perspective_transforms(N,H,W, max_relative_height=0.5, max_relative_width=0.2,
                                 min_distortion=0, scales=None):
    """
    Sample a batch of homography matrices for perspective transformation

    :N: int; batch size
    :H: int; image height
    :W: int; image width
    :max_relative_height: float on unit interval; how high up the image the "horizon" will be at max distortion
    :max_relative_width: float between 0.2 and 1; relative width of the distorted image at the "horizon"
    :min_distortion: float on unit interval; minimum amount of distortion to sample.
    :scales: None or numpy array of length N on the unit interval- if not None, override random sampling. scale of 0
        yields no perspective distortion 

    Returns
    :transforms: (N,3,3) torch.tensor containing a batch of perspective transform matrices
    :scales: (N,) numpy.ndarray giving sampled scales
    """
    # sample distortion scales
    if scales is None:
        scales = np.random.uniform(min_distortion, 1, size=N)
    else:
        assert len(scales) == N

    # Sample perspective matrices by drawing the boundaries we want the distorted image to fill,
    # then inferring the matrices from the boundaries
    x1 = torch.tensor([[[0.,0.], [W,0.], [W, H], [0., H]] for _ in range(N)]).type(torch.float32) # (N,4,2)

    a = (1-max_relative_width)/2 
    b = max_relative_height
    x2 = torch.tensor([
        [[s*a*W, s*b*H], [(1-s*a)*W, s*b*H], [W, H], [0, H]]
        for s in scales
    ]).type(torch.float32) # (N,4,2)

    warps = kornia.geometry.transform.get_perspective_transform(x1, x2) # (N,3,3)

    return warps, scales


class PerspectiveTilter(PipelineBase):
    """
    
    """
    name = "PerspectiveTilter"

    def __init__(self, max_relative_height=0.5, max_relative_width=0.2, min_distortion=0, logviz=True):
        """
        :max_relative_height:
        :max_relative_width:
        :min_distortion:
        """
        super().__init__()
        self.params = {"max_relative_height":max_relative_height,
                       "max_relative_width":max_relative_width,
                       "min_distortion":min_distortion}
        self._logviz = logviz
        
    def forward(self, x, control=False, evaluate=False, params={}, **kwargs):
        N, C, H, W = x.shape
        # maybe throw an error if params specified AND control=True
        if control & ("scale" in params):
            assert False, "can't override scale in a control batch"

        # retrieve last sample if control=True
        if control:
            scales = self.lastsample
        elif "scale" in params:
            scales = params["scale"]
        else:
            scales = None


        # sample matrices and scales
        with torch.no_grad():
            warps, scales = sample_perspective_transforms(N,H,W, 
                                                          max_relative_height=self.params["max_relative_height"],
                                                          max_relative_width=self.params["max_relative_width"],
                                                          min_distortion=self.params["min_distortion"],
                                                          scales=scales)
        # save scales as lastsample
        self.lastsample = scales

        warped = kornia.geometry.warp_perspective(x, warps, (H,W))

        return warped, kwargs
        
    def get_last_sample_as_dict(self):
        return {"scale":list(self.lastsample)}
        
    def log_vizualizations(self, x, x_control, writer, step):
        """
        """
        if self._logviz:
            img = self(x)[0]
            # if this is a batch of images, take the first one
            if len(img.shape) == 4:
                img = img[0]
            # check to make sure this is an RGB image
            if img.shape[0] == 3:
                writer.add_image("tilted_image", img, global_step=step)
         
    def get_description(self):
        return "**PerspectiveTilter**"