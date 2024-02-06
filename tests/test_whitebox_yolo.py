import numpy as np
import matplotlib
import torch

from electricmayhem.whitebox import _yolo
from .modelgenerators import *



def test_xywh2xyxy():
    xywh = torch.tensor([[50, 100, 10, 20]])
    xyxy = _yolo.xywh2xyxy(xywh)
    for i,v in enumerate([45, 90, 55, 110]):
        assert xyxy[0,i] == v

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
    
    
    
def test_convert_v4_to_v5_correct_output_shape():
    batch = 3
    num_classes = 5
    num_boxes = 7
    H = 64
    W = 64
    v4_output = [torch.tensor(np.random.normal(0, 1, size=(batch,num_boxes ,1, 4))),
                 torch.tensor(np.random.normal(0, 1, size=(batch,num_boxes,
                                                           num_classes)))]
    
    v5_output = _yolo.convert_v4_to_v5_format(v4_output, H, W)
    assert isinstance(v5_output, list)
    assert v5_output[0].shape == (batch, num_boxes, 5+num_classes)