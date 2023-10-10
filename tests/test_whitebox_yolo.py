import numpy as np
import matplotlib
import torch

from electricmayhem.whitebox import _yolo


def test_plot_detections():
    img_tensor = torch.tensor(np.random.uniform(0,1,size=(1,3,600,480)))
    N = 1
    M = 10
    classes = ["foo", "bar", "foobar"]
    detections = np.random.uniform(0, 1, size=(N,M,5+len(classes)))
    detections[:,:,0] *= 480
    detections[:,:,2] *= 480
    detections[:,:,1] *= 600
    detections[:,:,3] *= 600
    fig = _yolo.plot_detections(img_tensor, torch.tensor(detections), 
                                classnames=classes)
    assert isinstance(fig, matplotlib.figure.Figure)