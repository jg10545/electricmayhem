import numpy as np 
import dask
import torch
import torch.utils.tensorboard
import kornia.geometry
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import os
import yaml

import electricmayhem.mask
from electricmayhem import _augment

def estimate_transform_robustness(detect_func, augments, img, 
                                  mask=None, pert=None, 
                                  return_outcomes=False,
                                  include_error_as_positive=False,
                                  use_agresti_coull=True, use_scores=False):
    """
    Estimate transform robustness as an expectation over transformations. 
    
    :detect_func: detection function to wrap; should input a (C,H,W) torch Tensor and
        output a string
    :augments: list of dictionaries giving augmentation parameters
    :img: (C,H,W) torch Tensor containing the original image
    :mask: (C,H,W) torch Tensor containing the mask as 0/1 values
    :pert: (C,H',W') torch Tensor containing the adversarial perturbation
    :return_outcomes: bool; whether to return a list of results for every
        augmentation
    :include_error_as_positive: bool; whether to count -1s from the detect function
        as a positive detection
    :use_scores: if True, use soft scores instead of hard labels
    
    Returns a dictionary containing:
        :crash_frac: fraction of augments where function returned an empty string
        :detect_frac: fraction of augments where the correct license plate was detected
        :no_plates_frac: fraction of augments where no plates were detected
        :tr: transform robustness estimate; fraction of non-crash cases where the plate was 
            not detected
    """
    def _augment_and_detect(i,m,p,a):
        if use_scores:
            return detect_func(_augment.augment_image(i,mask=m, perturbation=p, **a),
                           return_raw=return_outcomes, return_scores=True)
        else:
            return detect_func(_augment.augment_image(i,mask=m, perturbation=p, **a),
                           return_raw=return_outcomes)
    
    tasks = [dask.delayed(_augment_and_detect)(img, mask, pert, a) 
             for a in augments]
    if return_outcomes:
        result = dask.compute(*tasks)
        outcomes = [r[0] for r in result]
        raw = [r[1] for r in result]
    else:
        outcomes = dask.compute(*tasks)
        
    # how often did openALPR crap out?
    crash_frac = np.mean([o == -1 for o in outcomes])
    if include_error_as_positive:
        #num_positives = np.sum([o in (1,-1) for o in outcomes])
        num_positives = np.sum([abs(o) for o in outcomes]) # to work with scores
        n = len(outcomes)
    else:
        # how often did it detect the plate?
        #num_positives = np.sum([o == 1 for o in outcomes])
        num_positives = np.sum([o for o in outcomes if o > 0]) # to work with scores
        n = len([o for o in outcomes if o >= 0])
        #detect_frac = np.mean([o == 1 for o in outcomes])
        #tr = 1-detect_frac/max((1-crash_frac), 1e-5)
    detect_frac = num_positives/n # if using scores, this becomes the mean score
    tr = 1-detect_frac
        
    # Agresti-Coull interval for z=1
    if use_agresti_coull:
        n_tilde = n+1
        p_tilde = (num_positives + 0.5)/n_tilde
        sem = np.sqrt((p_tilde*(1-p_tilde))/n_tilde)
        
    # Wald interval for z=1
    else:
        sem = np.sqrt((tr*(1-tr))/n) 
    
    outdict = {
        "crash_frac":crash_frac,
        "detect_frac":detect_frac,
        "tr":tr,
        "sem":sem
    }
    if return_outcomes:
        return outdict, outcomes, raw
    else:
        return outdict


def reduce_mask(img, priority_mask, perturbation, detect_func, augs, n=10,
                tr_threshold=0.75, maxval=0.9999, minval=0, use_scores=False):
    """
    Method for interpolating between the initial and final masks- DIFFERENT FROM THE GRAPHITE PAPER
    
    Input a "priority_mask" where all pixels we ultimately need to turn off are assigned a number between 0 and 1, and pick a threshold below which to disable them- so our goal is to eventually get the threshold to 1.
    
    The threshold is chosen by binary search aiming for a target transform
    robustness.
    
    :img: torch.Tensor in channel-first format, containing the original image
    :priority_mask: torch.Tensor in channel-first format containing mask
        with random values for interpolating between init and final masks
    :perturbation: torch.Tensor in channel-first format containing the
        adversarial perturbation
    :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
    :augs: list of augmentation parameters
    :n: int; number of steps to take in binary search
    :tr_threshold: transform robustness threshold to aim for
    :maxval: float; max value to search over
    :minval: float; min value to search over
    :use_scores:
        
    Returns final threshold a and a list of dictionaries containing the TR
        results at each search step.
    """
    results = []
    for i in range(n):
        # pick a mask threshold right between the max and min values
        a = 0.5*(maxval+minval)
        # compute the mask
        M = (priority_mask > a).float()
        # estimate transform robustness
        estimate = estimate_transform_robustness(detect_func, augs, img, 
                                                 mask=M,
                                                 pert=perturbation,
                                                 use_scores=use_scores)
        estimate["a"] = a
        results.append(estimate)

        #  if the robustness is too low, lower the maxval
        if estimate["tr"] < tr_threshold:
            maxval = a
        # if robustness is too low, raise the minval
        else:
            minval = a
    return a, results


def estimate_gradient(img, mask, pert, augs, detect_func, tr_estimate, q=10, 
                      beta=1, include_error_as_positive=False, use_scores=False):
    """
    Use Randomized Gradient-Free estimation to compute a gradient.
    
    NOTE should this be normalized by q? it doesn't appear to be in the GRAPHITE codebase.
    
    :img: torch.Tensor in channel-first format, containing the original image
    :mask: torch.Tensor in channel-first format containing mask
        as 0,1 values
    :pert torch.Tensor in channel-first format containing the
        adversarial perturbation
    :augs: list of augmentation parameters
    :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
    :tr_estimate: float; estimated transform robustness of the current
        image/mask/perturbation
    :q: int; number of random vectors to sample
    :beta: float; smoothing parameter setting size of random shifts to perturbation
    :include_error_as_positive: bool; whether to count -1s from the detect function
        as a positive detection
    :use_scores:
        
    Returns gradient as a (C,H,W) torch Tensor
    """
    # if the perturbation is a different size- we need a copy of the mask for normalization
    C,H,W = pert.shape
    if (mask.shape[1] != H)|(mask.shape[2] != W):
        mask_resized = kornia.geometry.transform.resize(mask, (H,W))
    else:
        mask_resized = mask
    us = []
    u_trs = []
    for _ in range(q):
        u = torch.randn(pert.shape).type(torch.FloatTensor)*mask_resized[:C,:,:]
        u = u/torch.norm(u)

        u_est = estimate_transform_robustness(detect_func, augs, img, 
                                              mask=mask, 
                                              pert=pert+beta*u,
                                              include_error_as_positive=include_error_as_positive,
                                              use_scores=use_scores)
        us.append(u)
        u_trs.append(u_est["tr"])
        
    gradient = torch.zeros_like(pert)
    for u,tr in zip(us, u_trs):
        gradient += (tr - tr_estimate)*u/(beta*q)
        
    return gradient


def update_perturbation(img, mask, pert, augs, detect_func, gradient, lrs=None,
                        include_error_as_positive=False, use_scores=False):
    """
    TO DO replace this with backtracking line search
    
    Update the perturbation by computing transform robustness at several different step sizes. Final perturbation is clamped between -1 and 1.
    
    :img: torch.Tensor in channel-first format, containing the original image
    :mask: torch.Tensor in channel-first format containing mask
        as 0,1 values
    :pert torch.Tensor in channel-first format containing the
        adversarial perturbation
    :augs: list of augmentation parameters
    :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
    :gradient: gradient estimate as a torch.Tensor in (C,H,W) format
    :lrs: optional list of learning rates to test
    :angle_scale: float; standard deviation, in degrees, of normal
        distribution angle will be chosen from
    :translate_scale: float; standard deviation, in pixels, of normal
        distribution x and y translations
    :include_error_as_positive: bool; hether to count -1s from the detect function
        as a positive detection
    :use_scores:
        
    Returns updated perturbation as a (C,H,W) torch Tensor, and a dictionary
        containing the chosen learning rate
    """
    if lrs is None:
        lrs = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
        
    # measure transform robustness at each 
    updated_trs = []
    for l in lrs:
        est = estimate_transform_robustness(detect_func, augs, img, 
                                            mask=mask,
                                            pert=pert+l*gradient,
                                           use_scores=use_scores)
        updated_trs.append(est["tr"])
        
    # now just pick whatever value worked best.
    final_lr = lrs[np.argmax(updated_trs)]
    updated_pert = pert + final_lr*gradient
    # there's almost certainly a more sophisticated way to clamp this. but just sticking it to [-1,1] 
    # will prevent it from slowly accumulating enormous values anywhere
    updated_pert = torch.clamp(updated_pert, 0,1)
    return updated_pert, {'lr':final_lr}




class BlackBoxPatchTrainer():
    """
    Class to wrap together all the pieces needed for black-box training
    of a physical adversarial patch, following the OpenALPR example in
    the GRAPHITE paper.
    """
    
    def __init__(self, img, initial_mask, final_mask, detect_func, logdir,
                 num_augments=100, q=10, beta=1, subset_frac=0, aug_params={}, tr_thresh=0.25,
                 reduce_steps=10,
                 eval_augments=1000, perturbation=None, mask_thresh=0.99, use_scores=False,
                 num_boost_iters=1, include_error_as_positive=False,
                 extra_params={}, fixed_augs=None, reduce_mask=True,
                 mlflow_uri=None, experiment_name=None, eval_func=None):
        """
        :img: torch.Tensor in (C,H,W) format representing the image being modified
        :initial_mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :final mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in
        :num_augments: int; number of augmentations to sample for each mask reduction, RGF, and line search step
        :q: int; number of random vectors to use for RGF
        :beta: float; RGF smoothing parameter
        :subset_frac: float;
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :tr_thresh:  float; transform robustness threshold to aim for 
            during mask reudction step
        :reduce_steps: int; number of steps to take during mask reduction
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :perturbation: torch.Tensor in (C,H,W) format. Optional initial perturbation
        :mask_thresh: float; when mask reduction hits this threshold swap
            over to the final_mask
        :use_scores:
        :num_boost_iters: int; number of boosts (RGF/line search) steps to
            run per epoch. GRAPHITE used 5.
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
        self.priority_mask = electricmayhem.mask.generate_priority_mask(initial_mask, final_mask)
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
        self.params = {"num_augments":num_augments, "q":q, "beta":beta,
                       "tr_thresh":tr_thresh,
                      "reduce_steps":reduce_steps, 
                      "num_boost_iters":num_boost_iters,
                      "include_error_as_positive":include_error_as_positive,
                      "reduce_mask":reduce_mask, "subset_frac":subset_frac,
                      "use_scores":use_scores}
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
    def _configure_mlflow(self, uri, expt):
        # set up connection to server, experiment, and start run
        if (uri is not None)&(expt is not None):
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(expt)
            mlflow.start_run()
            self._logging_to_mlflow = True
            
            # now log parameters
            mlflow.log_params(self.aug_params)
            mlflow.log_params(self.params)
            mlflow.log_params(self.extra_params)
        else:
            self._logging_to_mlflow = False
            
    def log_metrics_to_mlflow(self, metdict):
        if self._logging_to_mlflow:
            mlflow.log_metrics(metdict, step=self.query_counter)
        
            
            
    def __del__(self):
        mlflow.end_run()
        
    def _sample_augmentations(self, num_augments=None):
        """
        Randomly sample a list of augmentation parameters
        """
        if num_augments is None:
            num_augments = self.params["num_augments"]
            
        # if fixed augmentations were passed, sample from them
        # without replacement
        if self.fixed_augs is not None:
            return list(np.random.choice(self.fixed_augs, size=num_augments,
                                         replace=False))
        # otherwise generate new ones
        else:
            return [_augment.generate_aug_params(**self.aug_params) 
                    for _ in range(num_augments)]
        
    def _get_mask(self):
        """
        Return a binary mask using the current value of a
        """
        if self.a >= self.mask_thresh:
            return self.final_mask
        else:
            return (self.priority_mask > self.a).float()
    
    def _get_img_with_perturbation(self):
        """
        Return the current version of the image + masked perturbation
        glued on. Does not use composition noise
        """
        mask = self._get_mask()
        return _augment.compose(self.img, mask, self.perturbation, 0, 0)
        
    def _reduce_mask(self):
        """
        GRAPHITE REDUCE_MASK step.
        """
        augments = self._sample_augmentations()
        a, results = reduce_mask(self.img, 
                                           self.priority_mask, 
                                           self.perturbation, 
                                           self.detect_func,
                                           augments,
                                           n=self.params["reduce_steps"],
                                           tr_threshold=self.params["tr_thresh"],
                                           minval=self.a,
                                           use_scores=self.params["use_scores"])
        self.a = a
        self.tr = results[-1]["tr"]
        # for every mask threshold step record stats in tensorboard
        for r in results:
            self.query_counter += self.params["num_augments"]
            self.writer.add_scalar("reduce_mask_transform_robustness", r["tr"], global_step=self.query_counter)
            self.writer.add_scalar("reduce_mask_crash_frac", r["crash_frac"], global_step=self.query_counter)
            self.writer.add_scalar("a", r["a"], global_step=self.query_counter)
            
    def _estimate_gradient(self):
        """
        Estimate gradient with RGF
        """
        augments = self._sample_augmentations()
        mask = self._get_mask()
        # check to see if we need to take a random subset of the mask
        subset_frac = self.params["subset_frac"]
        if subset_frac > 0:
            mask = electricmayhem.mask.random_subset_mask(mask, subset_frac)
            self._subset_mask = mask
        self.tr = estimate_transform_robustness(self.detect_func,
                                                augments,
                                                self.img,
                                                mask=mask,
                                                pert=self.perturbation,
                                                include_error_as_positive=self.params["include_error_as_positive"],
                                                use_scores=self.params["use_scores"])["tr"]
        self.query_counter += self.params["num_augments"]
        self.writer.add_scalar("tr", self.tr, global_step=self.query_counter)
      
        gradient = estimate_gradient(self.img, 
                                     mask, 
                                     self.perturbation, 
                                     augments, 
                                     self.detect_func, 
                                     self.tr, 
                                     q=self.params["q"], 
                                     beta=self.params["beta"],
                                     include_error_as_positive=self.params["include_error_as_positive"],
                                     use_scores=self.params["use_scores"])
        self.query_counter += self.params["num_augments"]*self.params["q"]
        self.writer.add_scalar("gradient_l2_norm", np.sqrt(np.sum(gradient.numpy()**2)), global_step=self.query_counter)
        return gradient
        
    def _update_perturbation(self, gradient, lrs=None):
        """
        Pick a step size and update the perturbation
        """
        augments = self._sample_augmentations()
        # check to see whether we need to load a random subset mask
        if hasattr(self, "_subset_mask"):
            mask = self._subset_mask
        else:
            mask = self._get_mask()
        self.perturbation, resultdict = update_perturbation(self.img, 
                                                        mask,
                                                        self.perturbation,
                                                        augments, 
                                                        self.detect_func, 
                                                        gradient, 
                                                        lrs=lrs,
                                                        include_error_as_positive=self.params["include_error_as_positive"],
                                                        use_scores=self.params["use_scores"])
        self.query_counter += 8*self.params["num_augments"]
        self.writer.add_scalar("learning_rate", resultdict["lr"], global_step=self.query_counter)
        
        
    def evaluate(self):
        """
        Run a suite of evaluation tests on the test augmentations.
        """
        tr_dict, outcomes, raw = estimate_transform_robustness(self.detect_func, 
                                                          self.eval_augments,
                                                          self.img,
                                                          self._get_mask(),
                                                          self.perturbation,
                                                          return_outcomes=True,
                                                          include_error_as_positive=self.params["include_error_as_positive"])
        
        self.writer.add_scalar("eval_transform_robustness", tr_dict["tr"],
                               global_step=self.query_counter)
        self.writer.add_scalar("eval_crash_frac", tr_dict["crash_frac"],
                               global_step=self.query_counter)
        # only log TR to mlflow if we got rid of the mask, otherwise you
        # could trivially get TR=1
        if self.a >= self.mask_thresh:
            self.log_metrics_to_mlflow({"eval_transform_robustness":tr_dict["tr"]})
            # store results in memory too
            self.tr_dict = tr_dict
            
        # visual check for correlations in transform robustness across augmentation params
        coldict = {-1:'k', 1:'b', 0:'r'}
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter([a["scale"] for a in self.eval_augments],
                   [a["gamma"] for a in self.eval_augments],
                   s=[2+2*a["blur"] for a in self.eval_augments],
                   c=[coldict[o] for o in outcomes],
                  alpha=0.5)
        ax.set_xlabel("scale", fontsize=14)
        ax.set_ylabel("gamma", fontsize=14)
        
        self.writer.add_figure("evaluation_augmentations", fig, global_step=self.query_counter)
        
        if self.eval_func is not None:
            self.eval_func(self.writer, self.query_counter, 
                           img=self.img, mask=self._get_mask(),
                           perturbation=self.perturbation,
                           augs=self.eval_augments,
                           tr_dict=tr_dict, outcomes=outcomes, raw=raw,
                           include_error_as_positive=self.params["include_error_as_positive"])
        
    def _log_image(self):
        """
        log image to tensorboard
        """
        self.writer.add_image("img_with_mask_and_perturbation", self._get_img_with_perturbation(), global_step=self.query_counter)
        
    def _save_perturbation(self):
        filepath = os.path.join(self.logdir, f"perturbation_{self.query_counter}.npy")
        np.save(filepath, self.perturbation.numpy())
        
    def _run_one_epoch(self, lrs=None, budget=None):
        if self.a < self.mask_thresh:
            self._reduce_mask()
        if budget is not None:
            if self.query_counter >= budget:
                return
        for _ in range(self.params["num_boost_iters"]):
            gradient = self._estimate_gradient()
            if budget is not None:
                if self.query_counter >= budget:
                    return
                
            self._update_perturbation(gradient, lrs=lrs)
            if budget is not None:
                if self.query_counter >= budget:
                    return

                
    def fit(self, epochs=None, budget=None, lrs=None, eval_every=1):
        """
        Fit the model.
        
        :epochs: int; train for this many perturbation update steps
        :budget: int; train for this many queries to the black-box model
        :lrs: list; learning rates to test for GRAPHITE
        :eval_every: int; wait this many epochs between evaluations
        """
        self._log_image()
        
        if epochs is not None:
            for e in tqdm(range(epochs)):
                self._run_one_epoch(lrs=lrs)
                if (eval_every > 0)&((e+1)%eval_every == 0):
                    self.evaluate()
                    self._save_perturbation()
                    self._log_image()
                
        elif budget is not None:
            i = 1
            progress = tqdm(total=budget)
            while self.query_counter < budget:
                old_qc = self.query_counter
                self._run_one_epoch(lrs=lrs)
                progress.update(n=self.query_counter-old_qc)
                if (eval_every > 0)&(i%eval_every == 0):
                    self.evaluate()
                    self._save_perturbation()
                    self._log_image()
                i += 1
                
            progress.close()
        else:
            print("WHAT DO YOU WANT FROM ME?")       
        