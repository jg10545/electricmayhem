import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from ._pipeline import ModelWrapper
from electricmayhem._convenience import _plt_figure_to_image



def xywh2xyxy(x):
    """
    convert boxes from [x,y,w,h] to [left, upper, right, lower]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0] = x[:,0] - x[:,2]/2 # left x
    y[:,1] = x[:,1] - x[:,3]/2 # top y
    y[:,2] = x[:,0] + x[:,2]/2 # right x
    y[:,3] = x[:,1] + x[:,3]/2 # bottom y
    return y


def IoU(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    # find intersection- do the boxes overlap at all horizontally?
    if (right1 < left2) or (left1 > right2):
        intersection = 0
    # what about vertically?
    elif (bottom1 < top2) or (top1 > bottom2):
        intersection = 0
    else:
        xA = max(left1, left2)
        yA = max(top1, top2)
        xB = min(right1, right2)
        yB = min(bottom1, bottom2)
        intersection = max(max(xB-xA,0) * max(yB-yA,0), 0)
        
    area_1 = (right1 - left1)*(bottom1 - top1)
    area_2 = (right2 - left2)*(bottom2 - top2)
    union = area_1 + area_2 - intersection
    if union == 0 and intersection == 0:
        return 0
    else:
        return float(intersection)/union
        

def plot_detections(image, detections, classnames=None, thresh=0.1, iouthresh=0.5):
    """
    visualize the detections from the first element of a batch.
    
    Detections should have format (x1, y1, x2, y2, objectness, class1....classN)
    
    :image: (N,C,H,W) pytorch tensor
    :detections: (N,M,5+num_classes) pytorch tensor of detections. model will only
        output this if it's in .eval() mode
    """
    # pull first image as a numpy array
    im = image[0].permute(1,2,0).detach().cpu().numpy() # should be (H,W,C)
    # find relevant subset of detections- first image in batch
    detections = detections[0].detach().cpu().numpy() # (M,5+num_classes)
    # then above threshold
    detections = detections[detections[:,4] >= thresh]
    boxes = xywh2xyxy(detections[:,:4])
    scores = detections[:,4]
    classes = detections[:,5:].argmax(1)
    
    # need to sort scores to iterate in descending order
    ordering = scores.argsort()[::-1]
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(im)
    ax.set_axis_off()
    indices_plotted = []
    
    #for i in range(boxes.shape[0]):
    for i in ordering:
        plotbox = True
        for j in indices_plotted:
            iou = IoU(boxes[i], boxes[j])
            if iou >= iouthresh:
                plotbox = False
        if plotbox:
            if scores[i] > 0.5:
                color = "blue"
                lw = 2
            elif scores[i] > 0.25:
                color = "orange"
                lw = 2
            else:
                color = "red"
                lw = 1
            det_rect = matplotlib.patches.Rectangle((boxes[i,0], boxes[i,1]), 
                                                boxes[i,2]-boxes[i,0],
                                                boxes[i,3]-boxes[i,1],
                                                facecolor="none",
                                                edgecolor=color, lw=lw)
            ax.add_patch(det_rect)
            if classnames is None:
                c = str(classes[i])
            else:
                c = classnames[int(classes[i])]
            label = f"({scores[i]:.2f}) {c}"
            ax.text(boxes[i,0], boxes[i,1], label, color=color)
        
    return fig




class YOLOWrapper(ModelWrapper):
    """
    Subclass of ModelWrapper to include visualizations specific to
    YOLO models
    """
    name = "ModelWrapper"
    
    def __init__(self, model, logviz=True, classnames=None, thresh=0.1, iouthresh=0.5):
        """
        :model: pytorch YOLO model in eval mode
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        :classnames: list of output category names
        :thresh: objectness score threshold for plotting
        :iouthresh: IoU threshold for non maximal suppression during plotting
        """
        super().__init__(model)
        self._logviz = logviz
        self.classnames = classnames
        self.thresh = thresh
        self.iouthresh = iouthresh
        
    def log_vizualizations(self, x, x_control, writer, step):
        """
        """
        if self._logviz:
            detects = self(x[:1])[0]
            detects_control = self(x_control[:1])[0]
            
            plt.ioff()
            fig = plot_detections(x, detects, classnames=self.classnames,
                                  thresh=self.thresh, iouthresh=self.iouthresh)
            fig_control = plot_detections(x_control, detects_control,
                                          classnames=self.classnames,
                                          thresh=self.thresh, iouthresh=self.iouthresh)
            
            fig_arr = np.array(_plt_figure_to_image(fig))
            fig_control_arr = np.array(_plt_figure_to_image(fig_control))
            combined = np.concatenate([fig_control_arr, fig_arr], 1)
            plt.close(fig)
            plt.close(fig_control)
            plt.ion()
            
            # check to make sure this is an RGB image
            writer.add_image("detections", combined[:,:,:3], global_step=step,
                             dataformats='HWC')
        
    