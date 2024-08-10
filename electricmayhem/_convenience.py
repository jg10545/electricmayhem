import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import torch
import io

def load_to_tensor(i):
    """
    Input a path to an image (or a PIL.Image object) and convert
    to a normalized tensor in channel-first format
    
    Similar to the pil_to_tensor in torchvision, but does normalization
    and strips alpha channel if there is one
    """
    if isinstance(i, str):
        i = Image.open(i)
        
    # convert to a normalized float numpy array
    i = np.array(i).astype(np.float32)/255
    # convert to tensor in channel-first format
    t = torch.Tensor(i).permute(2,0,1)
    # return first three channels at most
    return t[:3,:,:]


def plot(i):
    """
    Matplotlib macro for plotting a channel-first image tensor
    
    :i: torch.Tensor containing normalized image in channel-first format
    """
    # in case it's on the GPU
    i = i.detach().cpu()
    # if there's a batch dimension just take it off
    if len(i.shape) == 4:
        i = i.squeeze(0)
    plt.imshow(i.permute(1,2,0))
    plt.axis(False)


def save(i, filepath):
    """
    Matplotlib/PIL macro for saving a channel-first image tensor
    
    :i: torch.Tensor containing normalized image in channel-first format
    """
    img = Image.fromarray((i.permute(1,2,0).numpy()*255).astype(np.uint8))
    img.save(filepath)
    
    
def _plt_figure_to_image(fig):
    """
    Convert a matplotlib figure to a PIL Image
    
    You can use this to generate an animated GIF from a bunch of figures;
    convert all to PIL images and run
    
    figs[0].save('giftest.gif',
               save_all=True, append_images=figs[1:], optimize=False, duration=40, loop=0)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')#, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)