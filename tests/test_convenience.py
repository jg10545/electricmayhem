import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from  electricmayhem import _convenience



def test_plt_figure_to_image():
    plt.ioff()
    fig, ax = plt.subplots(1)
    ax.plot([1,2,3,4,3,2,1], "o-")
    plt.ion()
    img = _convenience._plt_figure_to_image(fig)
    assert isinstance(img, Image.Image)
    assert np.array(img).shape == (480,640,4)