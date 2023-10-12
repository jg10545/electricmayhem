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
    fig = _yolo.plot_detections(img_tensor, torch.tensor(detections), 0, 
                                classnames=classes)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    

def test_plot_detection_pair_to_array():
    img_tensor = torch.tensor(np.random.uniform(0,1,size=(1,3,600,480)))
    N = 1
    M = 10
    classes = ["foo", "bar", "foobar"]
    detections = np.random.uniform(0, 1, size=(N,M,5+len(classes)))
    detections[:,:,0] *= 480
    detections[:,:,2] *= 480
    detections[:,:,1] *= 600
    detections[:,:,3] *= 600
    fig = _yolo._plot_detection_pair_to_array(img_tensor, torch.tensor(detections), 
                                              img_tensor, torch.tensor(detections),
                                              0, classnames=classes)
    assert isinstance(fig, np.ndarray)
    
    
def test_IoU_same_box():
    box1 = [0, 10, 20, 30]
    iou = _yolo.IoU(box1, box1)
    assert iou == 1
    

def test_IoU_half_overlap():
    box1 = [0, 10, 20, 30]
    box2 = [0, 10, 10, 30]
    iou = _yolo.IoU(box1, box2)
    assert iou == 0.5

def test_IoU_no_overlap():
    box1 = [0, 10, 20, 30]
    box2 = [1000, 10, 1020, 30]
    iou = _yolo.IoU(box1, box2)
    assert iou == 0