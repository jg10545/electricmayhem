import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from PIL import Image
import torch.utils.tensorboard
import yaml, os, json
from ax.service.ax_client import AxClient, ObjectiveProperties
import ax.modelbridge.generation_strategy 
import ax.modelbridge.registry
import mlflow
import kornia.geometry.transform

from electricmayhem import _augment, mask
from electricmayhem._graphite import estimate_transform_robustness, BlackBoxPatchTrainer, reduce_mask
from electricmayhem._perlin import normalize, _get_patch_outer_box_from_mask

def _inverse_fft(z, latent_shape, patch_shape):
    """
    :z:
    :latent_shape: tuple (H',W'); shape to resize z to before taking IDCT
    :patch_shape: tuple (H,W); shape to resize transformed patch to
    """
    # first half of z is the real component; second half is imaginary
    l = int(len(z)/2)
    z = z[:l] + z[l:]*1.0j
    z = z.reshape(latent_shape)
    # embed this frequency patch as the low-frequency region of
    # an array the same shape as the perturbation
    zprime = np.zeros(patch_shape, dtype=np.complex128)
    zprime[:latent_shape[0],:latent_shape[1]] += z
    # take the inverse transform
    x = scipy.fft.ifft2(zprime)
    # and compute the magnitude
    x = np.sqrt(x.real**2 + x.imag**2)
    # convert to channel-first format and normalize
    x = np.expand_dims(x, 0)
    return normalize(x)





class BlackBoxFourierPatchTrainer(BlackBoxPatchTrainer):
    """
    
    """
    
    def __init__(self, img, initial_mask, final_mask, detect_func, logdir,
                 num_augments=100, q=10, beta=1, aug_params={}, tr_thresh=0.5,
                 reduce_steps=10, r=0.25,
                 eval_augments=1000, freq_perturbation=None, mask_thresh=0.99,
                 num_boost_iters=1, include_error_as_positive=False,
                 extra_params={}, fixed_augs=None,
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
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :tr_thresh:  float; transform robustness threshold to aim for 
            during mask reudction step
        :reduce_steps: int; number of steps to take during mask reduction
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :perturbation: torch.Tensor in (C,H,W) format. Optional initial perturbation
        :mask_thresh: float; when mask reduction hits this threshold swap
            over to the final_mask
        :num_boost_iters: int; number of boosts (RGF/line search) steps to
            run per epoch. GRAPHITE used 5.
        :include_error_as_positive: bool; whether to count -1s from the detect function as a positive detection ONLY for boosting, not for mask reduction
        :extra_params: dictionary of other parameters you'd like recorded
        :fixed_augs: fixed augmentation parameters to sample from instead of 
            generating new ones each step.
        :mlflow_uri: string; URI for MLFlow server or directory
        :experiment_name: string; name of MLFlow experiment to log
        :eval_func: function containing any additional evalution metrics. run 
            inside self.evaluate()
        
        """
        self.query_counter = 0
        self.a = 0
        self.tr = 0
        self.mask_thresh = 0.99
        self.fixed_augs = fixed_augs
        self.eval_func = eval_func
        self.r = r
        
        self.img = img
        self.initial_mask = initial_mask
        self.final_mask = final_mask
        self.priority_mask = mask.generate_priority_mask(initial_mask, final_mask)
        self.detect_func = detect_func
        
        # figure out how large of a box we need
        self.pert_box = _get_patch_outer_box_from_mask(final_mask)
        self._box_params = {
                "H":self.pert_box["height"],
                "W":self.pert_box["width"],
               "r":r,
               "Hprime":int(r*self.pert_box["height"]),
               "Wprime":int(r*self.pert_box["width"]),
            }
        # dimension of the perturbation in frequency domain
        self._box_params["d"] = self._box_params["Hprime"]*self._box_params["Wprime"]*2
        
        if isinstance(eval_augments, int):
            eval_augments = [_augment.generate_aug_params(**aug_params) for _ in range(eval_augments)]
        self.eval_augments = eval_augments
        # generate initial perturbation in frequency domain
        if freq_perturbation is None:
            freq_perturbation = torch.Tensor(np.random.normal(0, 1,
                                            size=self._box_params["d"]))
        self.freq_perturbation = freq_perturbation
        # and precompute a spatial domain perturbation
        self.perturbation = self._generate_perturbation()
        
        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)
        
        self.aug_params = aug_params
        self.params = {"num_augments":num_augments, "q":q, "beta":beta,
                       "tr_thresh":tr_thresh,
                      "reduce_steps":reduce_steps, 
                      "num_boost_iters":num_boost_iters,
                      "include_error_as_positive":include_error_as_positive,
                      "r":r}
        self.extra_params = extra_params
        self.extra_params["d"] = self._box_params["d"]
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
        
    
    def _generate_perturbation(self, z=None):
        b = self.pert_box
        if z is None:
            z = self.freq_perturbation
            
        if not isinstance(z, np.ndarray):
            z = z.numpy()
        
        p = self._box_params
        noise = _inverse_fft(z, (p["Hprime"], p["Wprime"]),
                             (p["H"], p["W"]))
 
        perturbation = np.zeros((1, self.img.shape[1], self.img.shape[2]))
        perturbation[:,b["top"]:b["top"]+b["height"],b["left"]:b["left"]+b["width"]] += noise
        return torch.Tensor(perturbation)
        
    
    def _update_perturbation(self, gradient, lrs=None):
        """
        Pick a step size and update the perturbation
        
        returned perturbation is in the FREQUENCY DOMAIN
        """
        augments = self._sample_augmentations()
        mask = self._get_mask()
        
        if lrs is None:
            lrs = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
            
        # measure transform robustness at each 
        updated_trs = []
        for l in lrs:
            zprime = self.freq_perturbation + l*gradient
            perturbation = self._generate_perturbation(z=zprime)
            est = estimate_transform_robustness(self.detect_func, 
                                                augments, 
                                                self.img, 
                                                mask=mask,
                                                pert=perturbation)
            updated_trs.append(est["tr"])
            
        # now just pick whatever value worked best.
        final_lr = lrs[np.argmax(updated_trs)]
        updated_pert = self.freq_perturbation + final_lr*gradient
        # there's almost certainly a more sophisticated way to clamp this. but just sticking it to [0,1] 
        # will prevent it from slowly accumulating enormous values anywhere
        updated_pert = torch.clamp(updated_pert, 0,1)
        resultdict = {'lr':final_lr}
        
        self.freqpert = updated_pert
        self.perturbation = self._generate_perturbation()
        
        self.query_counter += len(lrs)*self.params["num_augments"]
        self.writer.add_scalar("learning_rate", resultdict["lr"],
                               global_step=self.query_counter)
    
    
    def _reduce_mask(self):
        """
        GRAPHITE REDUCE_MASK step. ASSUMES SPATIAL PERTURBATION
        HAS ALREADY BEEN UPDATED
        """
        augments = self._sample_augmentations()
        a, results = reduce_mask(self.img, 
                                           self.priority_mask, 
                                           self.perturbation, 
                                           self.detect_func,
                                           augments,
                                           n=self.params["reduce_steps"],
                                           tr_threshold=self.params["tr_thresh"],
                                           minval=self.a)
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
        Estimate gradient with RGF.
        
        """
        augments = self._sample_augmentations()
        mask = self._get_mask()
        pert = self.perturbation
        freqpert = self.freq_perturbation
        tr_estimate = self.tr
        beta = self.params["beta"]
        ieap = self.params["include_error_as_positive"]
        q = self.params["q"]
        
        C,H,W = pert.shape
        if (mask.shape[1] != H)|(mask.shape[2] != W):
            mask_resized = kornia.geometry.transform.resize(mask, (H,W))
        else:
            mask_resized = mask
        us = []
        u_trs = []
        for _ in range(q):
            # unlike normal graphite we don't scale by the mask since we're
            # doing gradient estimation in the frequency domain
            u = torch.randn(freqpert.shape).type(torch.FloatTensor)
            u = u/torch.norm(u)
            
            # now get the real-space perturbation for this trial
            perturbation = self._generate_perturbation(z=freqpert + beta*u)

            u_est = estimate_transform_robustness(self.detect_func, augments, 
                                                  self.img, 
                                                  mask=mask_resized, 
                                                  pert=perturbation,
                                                  include_error_as_positive=ieap)
            us.append(u)
            u_trs.append(u_est["tr"])
            
        gradient = torch.zeros_like(freqpert)
        for u,tr in zip(us, u_trs):
            gradient += (tr - tr_estimate)*u/(beta*q)
            
        self.query_counter += self.params["num_augments"]*self.params["q"]
        self.writer.add_scalar("gradient_l2_norm", np.sqrt(np.sum(gradient.numpy()**2)), global_step=self.query_counter)
        return gradient
        
    