import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from ._pipeline import ModelWrapper
from electricmayhem._convenience import _plt_figure_to_image



def xywh2xyxy(x):
    """
    convert boxes from [x,y,w,h] to [left, upper, right, lower]
    
    x should be an (N,4) array 
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0] = x[:,0] - x[:,2]/2 # left x
    y[:,1] = x[:,1] - x[:,3]/2 # top y
    y[:,2] = x[:,0] + x[:,2]/2 # right x
    y[:,3] = x[:,1] + x[:,3]/2 # bottom y
    return y


def IoU(box1, box2):
    """
    compute intersection-over-union for two boxes
    """
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
        
    
def convert_v4_to_v5_format(output, H, W):
    """
    Convert the outputs of a v4 model to v5 format. In the example
    code I've been working with, v4 outputs a list of two tensors:
        output[0] is [batch_size, num_boxes, 1, 4]
        output[1] is [batch_size, num_boxes, num_classes]
        
    v5 outputs a different list:
        output[0] is [batch_size, num_boxes, 5+num_classes]
        output[1] is a list of raw outputs from each detection head
        
    The box formats are also different- v4 appears to be xyxy normalized
    to the unit interval, while v5 outputs xywh in pixel coordinates.
    
    Note that v4 has no separate objectness score. we'll use the max value
    from class scores for that column.
    
    Inputs
    :output: list of two tensors output by a v4 model
    :H: int; image height in pixels
    :W: int; image width in pixels
    
    Outputs a list containing a single tensor in the format of the first v5 output
    """
    box_array = output[0]
    confs = output[1]
    
    max_score, max_score_index = torch.max(confs, -1)
    # convert box dims to other coordinate system
    box_dims = box_array.type(torch.float32)
    box_dims = torch.stack([(box_dims[:,:,:,0]*W+box_dims[:,:,:,2]*W)/2,
                            (box_dims[:,:,:,1]*H+box_dims[:,:,:,3]*H)/2,
                            box_dims[:,:,:,2]*W-box_dims[:,:,:,0]*W,
                            box_dims[:,:,:,3]*H-box_dims[:,:,:,1]*H,
                            ], -1)
    return [torch.concatenate([box_dims[:,:,0,:],
                               max_score.unsqueeze(-1),
                               confs], -1)]
    

def plot_detections(image, detections, k, classnames=None, thresh=0.1, iouthresh=0.5):
    """
    visualize the detections from one element of a batch.
    
    Detections should have format (x1, y1, x2, y2, objectness, class1....classN)
    
    :image: (N,C,H,W) pytorch tensor
    :detections: (N,M,5+num_classes) pytorch tensor of detections. model will only
        output this if it's in .eval() mode
    :k: which batch index to pull from
    :classnames: optional; list of class names for display
    :thresh: minimum objectness score for plotting
    :iousthresh: minimum IoU threshold for non-maximal suppression
    """
    # pull first image as a numpy array
    im = image[k].permute(1,2,0).detach().cpu().numpy() # should be (H,W,C)
    # find relevant subset of detections- first image in batch
    detections = detections[k].detach().cpu().numpy() # (M,5+num_classes)
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
            indices_plotted.append(i)
        
    return fig


def _plot_detection_pair_to_array(im1, dets1, im2, dets2, k, classnames=None, 
                         thresh=0.1, iouthresh=0.5):
    """
    Wrapper for plot_detections that plots a pair of images, converts to a
    numpy array, and concatenates them
    """
    fig1 = plot_detections(im1, dets1, k, classnames=classnames,
                          thresh=thresh, iouthresh=iouthresh)
    fig2 = plot_detections(im2, dets2, k, classnames=classnames,
                                  thresh=thresh, iouthresh=iouthresh)
    
    fig_arr1 = np.array(_plt_figure_to_image(fig1))
    fig_arr2 = np.array(_plt_figure_to_image(fig2))
    combined = np.concatenate([fig_arr1, fig_arr2], 1)
    plt.close(fig1)
    plt.close(fig2)
    return combined




class YOLOWrapper(ModelWrapper):
    """
    Subclass of ModelWrapper to include visualizations specific to
    YOLO models
    """
    name = "ModelWrapper"
    
    def __init__(self, model, logviz=True, classnames=None, thresh=0.1, iouthresh=0.5,
                 v4=False):
        """
        :model: pytorch YOLO model in eval mode
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        :classnames: list of output category names
        :thresh: objectness score threshold for plotting
        :iouthresh: IoU threshold for non maximal suppression during plotting
        :v4: bool; if True outputs are expected in VOLOv4 format instead of v5
        """
        super().__init__(model)
        self._logviz = logviz
        self.classnames = classnames
        self.thresh = thresh
        self.iouthresh = iouthresh
        self.params = {"v4":v4}
        
    def forward(self, x, control=False, evaluate=False, **kwargs):
        outputs = self.model(x)
        
        if self.params["v4"]:
            outputs = convert_v4_to_v5_format(outputs, x.shape[2], x.shape[3])

        return outputs
        
    def log_vizualizations(self, x, x_control, writer, step, logging_to_mlflow=False):
        """
        Log a batch of image pairs (control and with patch), showing model
        detections.
        """
        if self._logviz:
            fig_arrays = []
            detects = self(x)[0]
            detects_control = self(x_control)[0]
            plt.ioff()
            for i in range(x.shape[0]):
                fig_arrays.append(
                    _plot_detection_pair_to_array(x_control, detects_control, 
                                                  x, detects, i,
                                                  classnames=self.classnames,
                                                  thresh=self.thresh,
                                                  iouthresh=self.iouthresh)
                    )
            plt.ion()
            fig_arrays = np.stack(fig_arrays, 0)
            writer.add_images("detections", fig_arrays[:,:,:,:3], global_step=step,
                             dataformats='NHWC')
        
    