import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch

from  electricmayhem import _convenience


def test_load_to_tensor_from_path(tmp_path_factory):
    # save a fake PIL image with an alpha channel
    img = Image.fromarray(np.random.randint(0,255, size=(32,32,4)).astype(np.uint8))
    fn = str(tmp_path_factory.mktemp("images"))
    path = os.path.join(fn, "img.png")
    img.save(path)
    
    tens = _convenience.load_to_tensor(path)
    assert isinstance(tens, torch.Tensor)
    assert tens.shape == (3,32,32)
    assert tens.numpy().max() <= 1.0
    
    
def test_load_to_tensor_from_PIL():
    # save a fake PIL image with an alpha channel
    img = Image.fromarray(np.random.randint(0,255, size=(32,32,4)).astype(np.uint8))
    
    tens = _convenience.load_to_tensor(img)
    assert isinstance(tens, torch.Tensor)
    assert tens.shape == (3,32,32)
    assert tens.numpy().max() <= 1.0


def test_plt_figure_to_image():
    plt.ioff()
    fig, ax = plt.subplots(1)
    ax.plot([1,2,3,4,3,2,1], "o-")
    plt.ion()
    img = _convenience._plt_figure_to_image(fig)
    assert isinstance(img, Image.Image)
    assert np.array(img).shape == (480, 640, 4)
    
    
    
def test_figure_to_image():
    fig, ax = plt.subplots()
    ax.plot([193,45,2])
    img = _convenience._plt_figure_to_image(fig)
    assert isinstance(img, Image.Image)