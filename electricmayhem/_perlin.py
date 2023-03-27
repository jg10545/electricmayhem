import numpy as np
from noise import pnoise2

def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)


def perlin(H,W, period_y, period_x, octave, freq_sine, lacunarity = 2):
    """
    Wrapper for the noise.pnoise2() perlin noise generator.
    
    :H: int; height of output array
    :W: int; width of output array
    :period_y: float; RELATIVE spatial period in y direction. Will be scaled by H to get it in pixel coordinates
    :period_x: float; RELATIVE spatial period in x direction. Will be scaled by W to get it in pixel coordinates
    :octave: positive integer; number of iterations
    :freq_sine: float; frequency of sine function to put noise through
    :lacunarity: float; frequency increase at each octave
    
    Returns a 2D array
    """
    # convert period from relative pixel coords to absolute.
    period_y = period_y*H
    period_x = period_x*W
    # Perlin noise
    noise = np.empty((H,W,1), dtype = np.float32)
    for x in range(W):
        for y in range(H):
            noise[y,x,0] = pnoise2(x/period_x, y/period_y, octaves = octave, lacunarity = lacunarity)
            
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)

def _get_patch_outer_box_from_mask(mask):
    mask_numpy = mask.permute(1,2,0).numpy()
    # round up to a 2D array
    mask_numpy = mask_numpy.max(-1)
    H,W = mask_numpy.shape
    
    # INFER LEFT, TOP, X, AND Y
    y_range = np.arange(H)[mask_numpy.max(1).astype(bool)]
    x_range = np.arange(W)[mask_numpy.max(0).astype(bool)]
    top = y_range[0]
    left = x_range[0]
    y = y_range[-1] - y_range[0] + 1
    x = x_range[-1] - x_range[0] + 1
    return {"top":top, "left":left, "height":y, "width":x}