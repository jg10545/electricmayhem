import numpy as np
import matplotlib.pyplot as plt
import torch
import mlflow
import yaml
import json
import os 
from tqdm import tqdm
from ax.service.ax_client import AxClient, ObjectiveProperties

import ax.plot.diagnostic, ax.plot.scatter,  ax.plot.contour
import ax.utils.notebook.plotting
import ax.modelbridge.cross_validation

from noise import pnoise2
from ._graphite import BlackBoxPatchTrainer, estimate_transform_robustness
#import mask
from electricmayhem import _augment

def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)


def perlin(H,W, period_y, period_x, octave, freq_sine, lacunarity=2, phase=0):
    """
    Wrapper for the noise.pnoise2() perlin noise generator.
    
    :H: int; height of output array
    :W: int; width of output array
    :period_y: float; RELATIVE spatial period in y direction. Will be scaled by H to get it in pixel coordinates
    :period_x: float; RELATIVE spatial period in x direction. Will be scaled by W to get it in pixel coordinates
    :octave: positive integer; number of iterations
    :freq_sine: float; frequency of sine function to put noise through
    :lacunarity: float; frequency increase at each octave
    
    Returns a 2D array
    """
    # convert period from relative pixel coords to absolute.
    period_y *= H
    period_x *= W
    freq_sine *= (H+W/2)
    # Perlin noise
    noise = np.empty((1,H,W), dtype = np.float32)
    for x in range(W):
        for y in range(H):
            noise[0,y,x] = pnoise2(x/period_x, y/period_y, octaves=octave, lacunarity=lacunarity)
            
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi + phase)
    return normalize(noise)

def _get_patch_outer_box_from_mask(mask):
    mask_numpy = mask.permute(1,2,0).numpy()
    # round up to a 2D array
    mask_numpy = mask_numpy.max(-1)
    H,W = mask_numpy.shape
    
    # INFER LEFT, TOP, X, AND Y
    y_range = np.arange(H)[mask_numpy.max(1).astype(bool)]
    x_range = np.arange(W)[mask_numpy.max(0).astype(bool)]
    top = y_range[0]
    left = x_range[0]
    y = y_range[-1] - y_range[0] + 1
    x = x_range[-1] - x_range[0] + 1
    return {"top":top, "left":left, "height":y, "width":x}




class BayesianPerlinNoisePatchTrainer(BlackBoxPatchTrainer):
    """
    Black box patch trainer that attempts to generate an adversarial pattern
    using Perlin noise. Noise parameters are optimized using a Gaussian Process.
    """
    
    def __init__(self, img, final_mask, detect_func, logdir,
                 num_augments=100, aug_params={}, tr_thresh=0.5,
                 reduce_steps=10, eval_augments=1000, 
                 mask_thresh=0.99,
                 tune_lacunarity=True, 
                 tune_phase=False,
                 include_error_as_positive=False,
                 extra_params={}, fixed_augs=None,
                 mlflow_uri=None, experiment_name="perlin_noise", eval_func=None,
                 load_from_json_file=None):
        """
        :img: torch.Tensor in (C,H,W) format representing the image being modified
        :final mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in
        :num_augments: int; number of augmentations to sample for each mask reduction, RGF, and line search step
        
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :tr_thresh:  float; transform robustness threshold to aim for 
            during mask reudction step
        :reduce_steps: int; number of steps to take during mask reduction
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :mask_thresh: float; when mask reduction hits this threshold swap
            over to the final_mask
        :include_error_as_positive: bool; whether to count -1s from the detect function as a positive detection ONLY for boosting, not for mask reduction
        :extra_params: dictionary of other parameters you'd like recorded
        :fixed_augs: fixed augmentation parameters to sample from instead of 
            generating new ones each step.
        :mlflow_uri: string; URI for MLFlow server or directory
        :experiment_name: string; name of MLFlow experiment to log
        :eval_func: function containing any additional evalution metrics. run 
            inside self.evaluate()
        :load_from_json_file: load a pretrained Gaussian Process from a saved file.
            DOES NOT CHECK to make sure the parameters are consistent with this
            trainer object.
        """
        self.best_tr_so_far = 0
        self.query_counter = 0
        self.a = 0.99999#0
        self.tr = 0
        self.mask_thresh = 0.99
        self.fixed_augs = fixed_augs
        self.eval_func = eval_func
        
        self.img = img
        #self.initial_mask = initial_mask
        self.final_mask = final_mask
        #self.priority_mask = mask.generate_priority_mask(initial_mask, final_mask)
        self.detect_func = detect_func
        self.pert_box = _get_patch_outer_box_from_mask(final_mask)
        self._perlin_params = {"H":self.pert_box["height"],
                               "W":self.pert_box["width"], 
                               "period_y":1, 
                               "period_x":1, 
                               "octave":2, 
                               "freq_sine":1, 
                               "lacunarity":2}
        self.last_p_val = self._perlin_params
        
        if isinstance(eval_augments, int):
            eval_augments = [_augment.generate_aug_params(**aug_params) for _ in range(eval_augments)]
        self.eval_augments = eval_augments
        
        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)
        
        # set up Ax client
        if load_from_json_file is not None:
            self.client = AxClient.load_from_json_file(load_from_json_file)
        else:
            self.client = AxClient(verbose_logging=False)
            self.params = self._build_params(tune_lacunarity, tune_phase)
            self.client.create_experiment(
                name=experiment_name,
                parameters=self.params,
                objectives={"transform_robustness":ObjectiveProperties(minimize=False)}
                )
        
        self.aug_params = aug_params
        self.params = {"num_augments":num_augments,
                       "tr_thresh":tr_thresh,
                      "reduce_steps":reduce_steps, 
                      "include_error_as_positive":include_error_as_positive,
                      "tune_lacunarity":tune_lacunarity,
                      "tune_phase":tune_phase}
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
    def _build_params(self, tune_lacunarity=True, tune_phase=True):
        # period_y, period_x, octave, freq_sine
        params = [
            {"name":"period_x",
             "type":"range",
             "value_type":"float",
             "bounds":[0.1,2.]
                },
            {"name":"period_y",
             "type":"range",
             "value_type":"float",
             "bounds":[0.1,2.]
               },
            
            {"name":"octave",
             "type":"range",
             "value_type":"int",
             "bounds":[1,4]#[1,8]
             },
            {"name":"freq_sine",
             "type":"range",
             "value_type":"float",
             "bounds":[0.01,1]#[0.01,50.]#[0.01,10.]
                }]
        if tune_lacunarity:
            params.append(
            {"name":"lacunarity",
             "type":"range",
             "value_type":"float",
             "bounds":[1.,3.]
                })
        if tune_phase:
            params.append(
            {"name":"phase",
             "type":"range",
             "value_type":"float",
             "bounds":[-3.14, 3.14]
                })
        return params
        
    def _generate_perturbation(self, **kwargs):
        """
        kwargs overwrite defaults in self._perlin_params
        """
        b = self.pert_box
        if len(kwargs) > 0:
            p = kwargs
        else:
            p = self.last_p_val
        perl_dict = {}
        for k in self._perlin_params:
            if k in p:
                perl_dict[k] = p[k]
            else:
                perl_dict[k] = self._perlin_params[k]
        noise = perlin(**perl_dict)  
        perturbation = np.zeros((1, self.img.shape[1], self.img.shape[2]))
        perturbation[:,b["top"]:b["top"]+b["height"],b["left"]:b["left"]+b["width"]] += noise
        return torch.Tensor(perturbation)
        
        
    def _get_img_with_perturbation(self, **kwargs):
        """
        Return the current version of the image + masked perturbation
        glued on. Does not use composition noise
        """
        self.perturbation = self._generate_perturbation(**kwargs)
        return super()._get_img_with_perturbation()
    
    
    def _run_trial(self, p):
        """
        
        """
        for k in p:
            self.writer.add_scalar(k, p[k], global_step=self.query_counter)
        self.last_p_val = p
        # get the new perturbation
        self.perturbation = self._generate_perturbation(**p)
        #
        augments = self._sample_augmentations()
        #
        tr_dict = estimate_transform_robustness(
                self.detect_func,
                augments,
                self.img,
                mask=self.final_mask,
                pert=self.perturbation,
                include_error_as_positive=self.params["include_error_as_positive"]
            )
        self.query_counter += len(augments)
        self.writer.add_scalar("transform_robustness", tr_dict["tr"],
                               global_step=self.query_counter)
        self.writer.add_scalar("crash_frac", tr_dict["crash_frac"],
                               global_step=self.query_counter)
        

        d = {"transform_robustness":(tr_dict["tr"], tr_dict["sem"])}
        j = self.client.to_json_snapshot()
        json.dump(j, open(os.path.join(self.logdir, "log.json"), "w"))
        return d
    
        
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
        

        
        
    def _run_one_epoch(self, lrs=None, budget=None):
        parameters, trial_index = self.client.get_next_trial()
        self.client.complete_trial(trial_index=trial_index, 
                                   raw_data=self._run_trial(parameters))


    def contour_plot(self):
        """
        
        """
        ax.utils.notebook.plotting.init_notebook_plotting(offline=True)
        ax.utils.notebook.plotting.render(
            ax.plot.contour.interact_contour(
                model=self.client.generation_strategy.model,
                metric_name="transform_robustness")
            )
        
        
    def crossvalidation_plot(self):
        """
        
        """
        ax.utils.notebook.plotting.init_notebook_plotting(offline=True)
        cv_results = ax.modelbridge.cross_validation.cross_validate(
            model=self.client.generation_strategy.model)
        ax.utils.notebook.plotting.render(
            ax.plot.diagnostic.interact_cross_validation(
                cv_results
            )
        )
        
    
                   
