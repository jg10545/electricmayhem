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
    

def convert_ultralytics_to_v5_format(detections):
    """
    Convert newer ultralytics detections to the old v5 format
    
    :detections: [batch_size, 4+num_classes, num_boxes] torch.Tensor
    """
    maxdetect = torch.max(detections[:,4:,:], 1)[0].unsqueeze(1) # [batch_size, 1, num_boxes]
    newdets = torch.concat([detections[:,:4,:], # [batch_size, 4, num_boxes]
                         maxdetect,  # [batch_size, 1, num_boxes]
                         detections[:,4:,:]], # [batch_size, num_classes, num_boxes]
                           1) # [batch_size, 5+num_classes, num_boxes]
    return newdets.permute(0,2,1) # [batch_size, num_boxes, 5+num_classes]


def plot_detections(image, detections, k, classnames=None, thresh=0.1, iouthresh=0.5):
    """
    visualize the detections from one element of a batch.
    
    Detections should have format (x1, y1, x2, y2, objectness, class1....classN)
    
    :image: (N,C,H,W) pytorch tensor
    :detections: (N,M,5+num_classes) pytorch tensor of detections. model will only
        output this if it's in .eval() mode
    :k: int; which batch index to pull from
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
    ModelWrapper subclass for YOLO object detection models. This pipeline stage will
    add some tensorboard visualizations showing images with bounding box detections.

    Remember to set your model to eval() mode; YOLO uses batch normalization which will
    do weird things to your patch training if you don't freeze it.

    This has been tested on a few YOLO variants:

    -YOLOv4: The official YOLOv4 implementation still relies on DarkNet, but there are
        a couple unofficial PyTorch variants. We tested on the one at
        https://github.com/Tianxaomo/pytorch-YOLOv4 , which outputs results in a different
        forma than other YOLO implementations. If you use this version of YOLO models- set
        the v4 flag to True, and YOLOWrapper will check outputs for the different format and 
        attempt to convert them.
    -YOLOv5: models loaded with the official Ultralytics pytorch codebase should work fine.
    -YOLOv8 and later: models loaded with the ultralytics python package have a wrapper layer
        with a bunch of automation and postprocessing that doesn't all preserve gradients in
        inference. After loading, pull out the "model" attribute and pass that to ModelWrapper.
        For example:

        nano = ultralytics.YOLO("yolo11n.pt").model
        small = ultralytics.YOLO("yolo11s.pt").model
        nano.eval()
        small.eval()
        yolo = em.YOLOWrapper({"nano":nano, "small":small})

    """
    name = "YOLOWrapper"
    
    def __init__(self, model, eval_model=None, logviz=True, classnames=None, 
                 thresh=0.1, iouthresh=0.5, v4=False):
        """
        :model: pytorch model or dict of models, in eval mode
        :eval_model: optional model or dict of models to use in eval steps
        :logviz: if True, log the patch to TensorBoard every time pipeline.evaluate()
            is called.
        :classnames: list of output category names. if models with different output
            categories are being used, pass a dictionary mapping model name to list
            of category names
        :thresh: objectness score threshold for plotting
        :iouthresh: IoU threshold for non maximal suppression during plotting
        :v4: bool; if True outputs are expected in VOLOv4 format instead of v5
        """
        super().__init__(model, eval_model)
        self._logviz = logviz
        self.classnames = classnames
        self.thresh = thresh
        self.iouthresh = iouthresh
        self.params = {"v4":v4}
        
    def _convert_v4_to_v5_if_you_can(self, x, H, W):
        """
        only convert format from the non-official v4 codebase outputs if
        necessary
        """
        if len(x) == 2:
            if len(x[0].shape) == 4:
                if x[0].shape[2] == 1:
                    if x[0].shape[3] == 4:
                        return convert_v4_to_v5_format(x, H, W)
        return x
        
        
    def forward(self, x, control=False, evaluate=False, **kwargs):
        H, W = x.shape[2], x.shape[3]
        # if this is an evaluation step, check to see if we should run
        # the image through different models
        if evaluate:
            model = self.eval_model
            wraptype = self.eval_wraptype
            with torch.no_grad():
                outputs = self._call_wrapped(model,x)
        else:
            model = self.model
            wraptype = self.wraptype
            outputs = self._call_wrapped(model,x)
        
        # for the non-official v4 format, we might need to convert the results.
        # but in case we're mixing different formats- only run the conversion function
        # on the outputs that appear to be in that format.
        if self.params["v4"]:
            if wraptype == "list":
                outputs = [self._convert_v4_to_v5_if_you_can(o,H,W) for o in outputs]
            elif wraptype == "dict":
                outputs = {o:self._convert_v4_to_v5_if_you_can(outputs[o],H,W)
                           for o in outputs}
            else:
                outputs = self._convert_v4_to_v5_if_you_can(outputs,H,W)

        return outputs, kwargs
    
    def _vizualize_and_log_one_model_detection(self, x, x_control, 
                                               detects, detects_control, 
                                               writer, step, classnames,
                                               suffix=None,
                                               logging_to_mlflow=False):
        
        fig_arrays = []
        plt.ioff()
        for i in range(x.shape[0]):
            fig_arrays.append(
                _plot_detection_pair_to_array(x_control, detects_control, 
                                              x, detects, i,
                                              classnames=classnames,
                                              thresh=self.thresh,
                                              iouthresh=self.iouthresh)
                )
        plt.ion()
        if suffix is None:
            logname = "detections"
        else:
            logname = f"detections_{suffix}"
        fig_arrays = np.stack(fig_arrays, 0)
        writer.add_images(logname, fig_arrays[:,:,:,:3], global_step=step,
                         dataformats='NHWC')
        
        
        
        
    def log_vizualizations(self, x, x_control, writer, step, logging_to_mlflow=False):
        """
        Log a batch of image pairs (control and with patch), showing model
        detections.
        
        This function manages eval outputs that could include multiple models,
        with or without model names specified. It wraps
        _visualize_and_log_one_model_detection() which handles building and logging
        each individual figure.
        """
        # skip everything if we're not logging visualizations
        if self._logviz:
            # run images with and without patch through the model(s)
            detects, _ = self(x, evaluate=True)
            detects_control, _ = self(x_control, evaluate=True, control=True)
            # now log the viz separately for each model. if there's just one model:
            if self.eval_wraptype == "model":
                self._vizualize_and_log_one_model_detection(x, x_control, detects[0],
                                                       detects_control[0], writer, 
                                                       step, self.classnames,
                                                       logging_to_mlflow=logging_to_mlflow)
            # or a list of models- indicate each model by its list index
            elif self.eval_wraptype == "list":
                for i in range(len(detects)):
                    self._vizualize_and_log_one_model_detection(x, x_control, detects[i][0],
                                                           detects_control[i][0], writer, 
                                                           step, self.classnames,
                                                           suffix=i,
                                                           logging_to_mlflow=logging_to_mlflow)
            
            # or a dict of models
            elif self.eval_wraptype == "dict":
                for k in detects:
                    if isinstance(self.classnames, list):
                        classnames = self.classnames
                    elif isinstance(self.classnames, dict):
                        classnames = self.classnames[k]
                    else:
                        classnames = None
                    self._vizualize_and_log_one_model_detection(x, x_control, detects[k][0],
                                                           detects_control[k][0], writer, 
                                                           step, classnames,
                                                           suffix=k,
                                                           logging_to_mlflow=logging_to_mlflow)
    