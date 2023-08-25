import numpy as np 
import torch

def generate_rectangular_frame_mask(W, H, xmin, ymin, xmax, ymax, frame_width=30, 
                                    num_channels=3, return_torch=False):
    """
    Generate a mask for a rectangular frame.
    
    :W,H: dimensions of the image
    :xmin, ymin, xmax, ymax: pixel coords of INSIDE of frame
    :frame_width: pixel value of thickness of frame
    :num_channels: number of image channels
    :return_torch: if True, return pytorch Tensors in channel-first 
        format
    """
    final_mask = np.zeros((H,W,num_channels))

    final_mask[:ymin,:] = 1
    final_mask[ymax:,:] = 1
    final_mask[:,:xmin] = 1
    final_mask[:,xmax:] = 1

    final_mask[:ymin-frame_width,:] = 0
    final_mask[ymax+frame_width:,:] = 0
    final_mask[:,:xmin-frame_width] = 0
    final_mask[:,xmax+frame_width:] = 0
    
    initial_mask = np.zeros((H,W,num_channels))
    initial_mask[ymin-frame_width:ymax+frame_width,
                    :xmin-frame_width:xmax+frame_width] += 1
    initial_mask = np.ones((H,W,num_channels))
    initial_mask[:ymin-frame_width,:] = 0
    initial_mask[ymax+frame_width:,:] = 0
    initial_mask[:,:xmin-frame_width] = 0
    initial_mask[:,xmax+frame_width:] = 0
    
    if return_torch:
        return torch.Tensor(initial_mask).permute(2,0,1), torch.Tensor(final_mask).permute(2,0,1)
    
    return initial_mask, final_mask

def generate_priority_mask(init_mask, final_mask):
    """
    Build a tensor with values between 0 and 1 that allows us to
    tune between the initial and final masks, using
    
    (priority_mask > alpha).float() 
    
    Where alpha=0 gives the initial mask and alpha=0.9999
    would give the final mask
    
    :init_mask: torch Tensor; initial mask in (C,H,W) format
    :final_mask: torch Tensor; final mask in (C,H,W) format
    """
    C, H, W = final_mask.shape
    priority = torch.Tensor(np.random.uniform(0, 1, size=(1, H, W)))
    priority_mask = final_mask + priority*(init_mask-final_mask)
    return priority_mask



def random_subset_mask(m, frac=0):
    """
    Take a random subset of the mask, within some 
    :m: mask as a pytorch tensor in channel-first format
    :frac: size of random mask, as a fraction of the max
        dimensions of the patch. 0 to disable subsetting.
    """
    # disable
    if frac <= 0:
        return m
    # find the dimensions of the extent of nonzero parts of the mask
    mn = m.clone().detach().numpy()
    C,H,W = mn.shape  
    yvals = np.arange(H)[mn.sum(0).sum(1).astype(bool)]
    xvals = np.arange(W)[mn.sum(0).sum(0).astype(bool)]
    ymin, ymax = yvals.min(), yvals.max()
    xmin, xmax = xvals.min(), xvals.max()
    # half-dimensions of our subset
    dy = int(frac*(ymax-ymin)/2)
    dx = int(frac*(xmax-xmin)/2)
    
    # pick a random point. we expect masks to have complicated shapes
    # including holes, so sample from the nonzero mask elements instead
    # of uniformly from the extent
    mnr = mn.sum(0).ravel()
    options = np.arange(len(mnr))[mnr > 0]
    # pick a point for the centroid of our subset
    choice = np.random.choice(options)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    ychoice = yy.ravel()[choice]
    xchoice = xx.ravel()[choice]
    # now create a rectangular mask around that point
    choicemask = np.zeros_like(mn)
    choicemask[:, ychoice-dy:ychoice+dy, xchoice-dx:xchoice+dx] = 1
    # return the conjunction of our subset and the initial mask
    return torch.tensor(choicemask)*m