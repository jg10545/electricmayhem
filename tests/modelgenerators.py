"""
Dummy PyTorch models for use in unit tests
"""
import torch


class DummyConvNet(torch.nn.Module):
    def __init__(self, outdim=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.dense = torch.nn.Linear(5,outdim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.reshape(x, x.shape[:2])
        x = self.dense(x)
        x = torch.nn.functional.relu(x)
        return x
    
    
    
class DummyYOLO(torch.nn.Module):
    """
    Module that inputs an image and outputs something that looks like the
    standard YOLO format. 
    
    Output is a list of length 2; the firt element is a tensor of 
    shape (batch_size, num_boxes, 5+num_classes)
    """
    def __init__(self, num_boxes=11, num_classes=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.dense = torch.nn.Linear(5,(5+num_classes)*num_boxes)
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.reshape(x, x.shape[:2])
        x = self.dense(x)
        x = torch.reshape(x, (x.shape[0], self.num_boxes, self.num_classes+5))
        return [x, None]
    
    
class DummyYOLOv4(torch.nn.Module):
    """
    Module that inputs an image and outputs something that looks like the
    format in the unofficial pytorch YOLOv4 repo.
    
    output[0] is [batch_size, num_boxes, 1, 4]
    output[1] is [batch_size, num_boxes, num_classes]
    """
    def __init__(self, num_boxes=11, num_classes=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.densebox = torch.nn.Linear(5, 4*num_boxes)
        self.denseclass = torch.nn.Linear(5, num_classes*num_boxes)
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.reshape(x, x.shape[:2])
        
        x_box = self.densebox(x)
        x_box = torch.reshape(x_box, (x_box.shape[0], self.num_boxes, 1,4))
        
        x_class = self.denseclass(x)
        x_class = torch.reshape(x_class, (x_class.shape[0], self.num_boxes,
                                          self.num_classes))
        return [x_box, x_class]