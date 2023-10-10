import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from ._pipeline import ModelWrapper
from electricmayhem._convenience import _plt_figure_to_image



def plot_detections(image, detections, classnames=None, thresh=0.1):
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
    boxes = detections[:,:4]
    scores = detections[:,4]
    classes = detections[:,5:].argmax(1)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(im)
    ax.set_axis_off()
    
    for i in range(boxes.shape[0]):
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
        label = f"({round(scores[i],2)}) {c}"
        ax.text(boxes[i,0], boxes[i,1], label, color=color)
        
    return fig




class YOLOWrapper(ModelWrapper):
    """
    Subclass of ModelWrapper to include visualizations specific to
    YOLO models
    """
    def __init__(self, logviz=True, classnames=None):
        """
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        """
        super().__init__()
        self._logviz = logviz
        self.classnames = classnames
        
    def log_vizualizations(self, x, x_control, writer, step):
        """
        """
        if self._logviz:
            detects = self(x[:1])[0]
            detects_control = self(x_control[:1])[0]
            
            plt.ioff()
            fig = plot_detections(x, detects, classnames=self.classnames)
            fig_control = plot_detections(x_control, detects_control,
                                          classnames=self.classnames)
            
            fig_arr = np.array(_plt_figure_to_image(fig))
            fig_control_arr = np.array(_plt_figure_to_image(fig_control))
            combined = np.concatenate([fig_arr, fig_control_arr], 1)
            plt.ion()
            
            # check to make sure this is an RGB image
            writer.add_image("detections", combined[:,:,:3], global_step=step)
        
    