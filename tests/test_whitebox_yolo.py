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
    
    
def test_convert_ultralytics_to_v5_format_correct_output_shape():
    N = 3
    num_classes = 7
    num_boxes = 13

    input = torch.tensor(np.random.uniform(0, 1, size=(N, 4+num_classes, num_boxes)).astype(int))
    output = _yolo.convert_ultralytics_to_v5_format(input)

    assert output[0].shape == (N, num_boxes, 5+num_classes)
    
def test_yolowrapper_single_v5_model(tmp_path_factory):
    num_classes = 7
    num_boxes = 11
    classnames = [str(i) for i in range(num_classes)]
    model = DummyYOLO(num_boxes=num_boxes, num_classes=num_classes)
    
    yolo = _yolo.YOLOWrapper(model, classnames=classnames)
    
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,13,23)).astype(np.float32))
    y, _ = yolo(x)
    assert y[0].shape == (x.shape[0], num_boxes, 5+num_classes)
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    yolo.log_vizualizations(x, x, writer, 0)
    

def test_yolowrapper_single_v8_model(tmp_path_factory):
    """
    This should look like v5 outputs but with an additional column
    """
    num_classes = 7
    num_boxes = 11
    classnames = [str(i) for i in range(num_classes)]
    model = DummyYOLOv8(num_boxes=num_boxes, num_classes=num_classes)
    
    yolo = _yolo.YOLOWrapper(model, classnames=classnames, yolo_version=8)
    
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,13,23)).astype(np.float32))
    assert model(x)[0].shape == (x.shape[0], 4+num_classes, num_boxes)
    assert _yolo.convert_ultralytics_to_v5_format(model(x)[0])[0].shape == (x.shape[0], num_boxes, 5+num_classes)
    y, _ = yolo(x)
    assert y[0].shape == (x.shape[0], num_boxes, 5+num_classes)
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    yolo.log_vizualizations(x, x, writer, 0)
    

    
def test_yolowrapper_single_v4_model(tmp_path_factory):
    num_classes = 7
    num_boxes = 11
    classnames = [str(i) for i in range(num_classes)]
    model = DummyYOLOv4(num_boxes=num_boxes, num_classes=num_classes)
    
    yolo = _yolo.YOLOWrapper(model, classnames=classnames, yolo_version=4)
    
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,13,23)).astype(np.float32))
    y, _ = yolo(x)
    assert y[0].shape == (x.shape[0], num_boxes, 5+num_classes)
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    yolo.log_vizualizations(x, x, writer, 0)
    
    

    
def test_yolowrapper_different_train_eval_models(tmp_path_factory):
    num_classes = 7
    num_boxes = 11
    classnames = [str(i) for i in range(num_classes)]
    model = DummyYOLOv4(num_boxes=num_boxes, num_classes=num_classes)
    
    num_classes_eval = 7
    num_boxes_eval = 13
    model_eval = DummyYOLO(num_boxes=num_boxes_eval, num_classes=num_classes_eval)
    
    yolo = _yolo.YOLOWrapper({"v4":model}, classnames=classnames, 
                             eval_model={"v5":model_eval}, yolo_version={"v4":4, "v5":5})
    
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,13,23)).astype(np.float32))
    y, _ = yolo(x)
    assert y["v4"][0].shape == (x.shape[0], num_boxes, 5+num_classes)
    y_eval, _ = yolo(x, evaluate=True)
    assert y_eval["v5"][0].shape == (x.shape[0], num_boxes_eval, 5+num_classes_eval)
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    yolo.log_vizualizations(x, x, writer, 0)
    
    
    
    
def test_yolowrapper_different_train_eval_model_dicts(tmp_path_factory):
    num_classes = 7
    num_boxes_1 = 11
    num_boxes_2 = 13
    # test option to pass different classnames for each model
    classnames = {key:[str(i) for i in range(num_classes)] for key in
                  ["foo", "bar"]}
    model = {"foo":DummyYOLOv4(num_boxes=num_boxes_1, num_classes=num_classes),
             "bar":DummyYOLOv4(num_boxes=num_boxes_2, num_classes=num_classes)}
    
    num_classes_eval = 12
    num_boxes_eval_1 = 13
    num_boxes_eval_2 = 17
    num_boxes_eval_3 = 23
    model_eval = {"a":DummyYOLO(num_boxes=num_boxes_eval_1, num_classes=num_classes_eval),
                  "b":DummyYOLO(num_boxes=num_boxes_eval_2, num_classes=num_classes_eval),
                  "c":DummyYOLOv8(num_boxes=num_boxes_eval_3, num_classes=num_classes_eval)}
    for key in ["a", "b", "c"]:
        classnames[key] = [str(i) for i in range(num_classes_eval)]
    
    yolo = _yolo.YOLOWrapper(model, classnames=classnames, yolo_version={"foo":4, "bar":4, 
                                                                         "a":5, "b":5, "c":8},
                             eval_model=model_eval)
    
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,13,23)).astype(np.float32))
    y, _ = yolo(x)
    assert isinstance(y, dict)
    assert y["foo"][0].shape == (x.shape[0], num_boxes_1, 5+num_classes)
    assert y["bar"][0].shape == (x.shape[0], num_boxes_2, 5+num_classes)
    
    y_eval, _ = yolo(x, evaluate=True)
    assert isinstance(y_eval, dict)
    assert y_eval["a"][0].shape == (x.shape[0], num_boxes_eval_1, 5+num_classes_eval)
    assert y_eval["b"][0].shape == (x.shape[0], num_boxes_eval_2, 5+num_classes_eval)
    assert y_eval["c"][0].shape == (x.shape[0], num_boxes_eval_3, 5+num_classes_eval)
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    yolo.log_vizualizations(x, x, writer, 0)
    