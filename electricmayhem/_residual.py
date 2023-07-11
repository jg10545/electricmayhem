import numpy as np
import torch
import yaml
import kornia.geometry
import os

from ._graphite import BlackBoxPatchTrainer, estimate_transform_robustness
from electricmayhem import mask, _augment






class ResidualBlackBoxPatchTrainer(BlackBoxPatchTrainer):
    """
    Class to wrap together all the pieces needed for black-box training
    of a physical adversarial patch, following the OpenALPR example in
    the GRAPHITE paper.
    
    This version is modified to use the gradient estimation technique
    detailed in "A New One-Point Residual-Feedback Oracle for Black-Box
    Learning and Control" by Zhang et al (2021)
    """
    
    def __init__(self, img, initial_mask, final_mask, detect_func, logdir,
                 num_augments=100, beta=1, lr=0.1, aug_params={}, tr_thresh=0.25,
                 reduce_steps=10,
                 eval_augments=1000, perturbation=None, mask_thresh=0.99,
                 include_error_as_positive=False,
                 extra_params={}, fixed_augs=None, reduce_mask=True,
                 mlflow_uri=None, experiment_name=None, eval_func=None):
        """
        :img: torch.Tensor in (C,H,W) format representing the image being modified
        :initial_mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :final mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in
        :num_augments: int; number of augmentations to sample for each mask reduction, RGF, and line search step
        :beta: float; RGF smoothing parameter
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :tr_thresh:  float; transform robustness threshold to aim for 
            during mask reudction step
        :reduce_steps: int; number of steps to take during mask reduction
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :perturbation: torch.Tensor in (C,H,W) format. Optional initial perturbation
        :mask_thresh: float; when mask reduction hits this threshold swap
            over to the final_mask
        :include_error_as_positive: bool; whether to count -1s from the detect function as a positive detection ONLY for boosting, not for mask reduction
        :extra_params: dictionary of other parameters you'd like recorded
        :fixed_augs: fixed augmentation parameters to sample from instead of 
            generating new ones each step.
        :reduce_mask: whether to include GRAPHITE's mask reduction step
        :mlflow_uri: string; URI for MLFlow server or directory
        :experiment_name: string; name of MLFlow experiment to log
        :eval_func: function containing any additional evalution metrics. run 
            inside self.evaluate()
        
        """
        self.query_counter = 0
        if reduce_mask:
            self.a = 0
        else:
            self.a = 1
        self.tr = 0
        self.mask_thresh = 0.99
        self.fixed_augs = fixed_augs
        self.eval_func = eval_func
        
        self.img = img
        self.initial_mask = initial_mask
        self.final_mask = final_mask
        self.priority_mask = mask.generate_priority_mask(initial_mask, final_mask)
        self.detect_func = detect_func
        
        if isinstance(eval_augments, int):
            eval_augments = [_augment.generate_aug_params(**aug_params) for _ in range(eval_augments)]
        self.eval_augments = eval_augments
        if perturbation is None:
            perturbation = torch.Tensor(np.random.uniform(0, 1, size=img.shape))
        self.perturbation = perturbation
        
        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)
        
        self.aug_params = aug_params
        self.params = {"num_augments":num_augments, "beta":beta, "lr":lr,
                       "tr_thresh":tr_thresh,
                      "reduce_steps":reduce_steps, 
                      "include_error_as_positive":include_error_as_positive,
                      "reduce_mask":reduce_mask,
                      "num_boost_iters":1}
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
        
    # HERE WE GO
            
    def _estimate_gradient(self):
        """
        Estimate gradient with RGF
        """
        augments = self._sample_augmentations()
        mask = self._get_mask()
        ieap = self.params["include_error_as_positive"]
        
        # if it's our first call to this function, initialize resid
        if not hasattr(self, "_residual_tr"):
            self._residual_tr = estimate_transform_robustness(self.detect_func,
                                                    augments,
                                                    self.img,
                                                    mask=mask,
                                                    pert=self.perturbation,
                                                    include_error_as_positive=ieap)["tr"]
            self.query_counter += self.params["num_augments"]
            
        # sample
        
        # modified from _graphite/estimate_gradient()
        C,H,W = self.perturbation.shape
        # upsample mask if necessary
        if (mask.shape[1] != H)|(mask.shape[2] != W):
            mask_resized = kornia.geometry.transform.resize(mask, (H,W))
        else:
            mask_resized = mask
        # generate random direction and normalize to unit sphere (note: in some
        # papers unit sphere sampling was advantageous for variance of estimator,
        # but Zhang et al used normal samples.)
        u = torch.randn(self.perturbation.shape).type(torch.FloatTensor)*mask_resized[:C,:,:]
        u = u/torch.norm(u)
        
        # shift perturbation in a random direction and compute TR
        shifted = torch.clamp(self.perturbation+self.params["beta"]*u, 0, 1)
        shifted_tr = estimate_transform_robustness(self.detect_func,
                                                augments,
                                                self.img,
                                                mask=mask,
                                                pert=shifted,
                                                include_error_as_positive=ieap)["tr"]
        self.query_counter += self.params["num_augments"]
        
        self.writer.add_scalar("shifted_tr", shifted_tr, 
                               global_step=self.query_counter)
        
        # estimated gradient
        gradient = u * (shifted_tr - self._residual_tr)/self.params["beta"]
        
        self._residual_tr = shifted_tr
        
        return gradient
        
    def _update_perturbation(self, gradient):
        """
        Pick a step size and update the perturbation
        """
        pert = self.perturbation - self.params["lr"]*gradient
        self.perturbation = torch.clamp(pert, 0, 1)