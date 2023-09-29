import numpy as np 
import matplotlib.pyplot as plt
import torch
import yaml
import json
import os 
from ax.service.ax_client import AxClient, ObjectiveProperties

import ax.plot.diagnostic, ax.plot.scatter,  ax.plot.contour
import ax.utils.notebook.plotting
import ax.modelbridge.cross_validation
import ax.modelbridge.generation_strategy 
import ax.modelbridge.registry 

from electricmayhem import _perlin, _augment
from ._graphite import estimate_transform_robustness


def sigmoid(x):
    return 1/(1+np.exp(-x))


def g(XX, YY, H, W, K, a_sq, F0, w0, x0=0, y0=0):
    """
    for a single wavelet
    """
    return K*np.exp(-np.pi*a_sq*((XX-x0)**2 + (YY-y0)**2))*np.cos(2*np.pi*F0*((XX-x0)*np.cos(w0)/W+(YY-y0)*np.sin(w0)/W))


def _generate_wavelet_noise_image(H, W, num_wavelets, params):
    """
    
    """
    img = np.zeros((H,W))
    XX,YY = np.meshgrid(np.arange(W), np.arange(H))
    a_sq = 1./(H*W)
    
    for n in range(num_wavelets):
        img += g(XX, YY, H, W, 1, a_sq,
                 params[f"F0_{n}"],
                 params[f"w0_{n}"],
                 params[f"x0_{n}"],
                 params[f"y0_{n}"])
        
    return img



class BayesianWaveletNoisePatchTrainer(_perlin.BayesianPerlinNoisePatchTrainer):
    """
    Black box patch trainer that attempts to generate an adversarial pattern
    using wavelet noise. Noise parameters are optimized using a Gaussian Process.
    """
    
    def __init__(self, img, final_mask, detect_func, logdir,
                 num_augments=100, aug_params={}, 
                 eval_augments=1000, 
                 num_wavelets=5,
                 num_sobol=5,
                 include_error_as_positive=False,
                 use_scores=False,
                 extra_params={}, fixed_augs=None,
                 mlflow_uri=None, experiment_name="wavelet_noise", eval_func=None,
                 load_from_json_file=None):
        """
        :img: torch.Tensor in (C,H,W) format representing the image being modified
        :final mask: torch.Tensor in (C,H,W) starting mask as an image with 0,1 values
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in
        :num_augments: int; number of augmentations to sample for each mask reduction, RGF, and line search step
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :num_wavelets: int; how many wavelets to superpose
        :num_sobol: int; number of Sobol sampling steps before switching to Gaussian
            process
        :include_error_as_positive: bool; whether to count -1s from the detect function as a positive detection ONLY for boosting, not for mask reduction
        :use_scores: incorporate scores instead of hard labels (training only)
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
        self.pert_box = _perlin._get_patch_outer_box_from_mask(final_mask)
        self._wavelet_params = {"H":self.pert_box["height"],
                               "W":self.pert_box["width"], 
                               "num_wavelets":num_wavelets}
        #self.last_p_val = self._perlin_params
        
        if isinstance(eval_augments, int):
            eval_augments = [_augment.generate_aug_params(**aug_params) for _ in range(eval_augments)]
        self.eval_augments = eval_augments
        
        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)
        
        # set up Ax client
        if load_from_json_file is not None:
            self.client = AxClient.load_from_json_file(load_from_json_file)
        else:
            model = ax.modelbridge.registry.Models.GPEI
            gs = ax.modelbridge.generation_strategy.GenerationStrategy(
                steps=[
                    # Quasi-random initialization step
                    ax.modelbridge.generation_strategy.GenerationStep(
                        model=ax.modelbridge.registry.Models.SOBOL,
                        num_trials=num_sobol,  
                        ),
                    # Bayesian optimization step using the custom acquisition function
                    ax.modelbridge.generation_strategy.GenerationStep(
                        model=model,
                        num_trials=-1, 
                        ),
                    ]
                )
            
            self.client = AxClient(generation_strategy=gs,
                                   verbose_logging=False)
            self.params = self._build_params(self.pert_box["height"], 
                                             self.pert_box["width"], 
                                             num_wavelets)
            self.client.create_experiment(
                name=experiment_name,
                parameters=self.params,
                objectives={"transform_robustness":ObjectiveProperties(minimize=False)}
                )
        
        self.aug_params = aug_params
        self.params = {"num_augments":num_augments,
                      "include_error_as_positive":include_error_as_positive,
                      "num_wavelets":num_wavelets,
                      "num_sobol":num_sobol,
                      "use_scores":use_scores}
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
    def _build_params(self, H, W, num_wavelets):
        # period_y, period_x, octave, freq_sine
        params = [
            {"name":"scale",
             "type":"range",
             "value_type":"float",
             "log_scale":True,
             "bounds":[0.5,5.] 
                },
            ]
        
        for n in range(num_wavelets):
            waveparams = [
                {"name":f"F0_{n}",
                 "type":"range",
                 "value_type":"float",
                 "log_scale":True,
                 "bounds":[0.1,25.]                    
                    },
                {"name":f"w0_{n}",
                 "type":"range",
                 "value_type":"float",
                 "bounds":[0.,2*np.pi]                    
                    },
                {"name":f"x0_{n}",
                 "type":"range",
                 "value_type":"float",
                 "bounds":[0., float(W)]                
                    },
                {"name":f"y0_{n}",
                 "type":"range",
                 "value_type":"float",
                 "bounds":[0., float(H)]                 
                    },
                ]
            params += waveparams
            
        return params
        
    def _generate_perturbation(self, **kwargs):
        """
        kwargs overwrite defaults in self._perlin_params
        """
        b = self.pert_box
        if len(kwargs) > 0:
            p = kwargs
        elif hasattr(self, "last_p_val"):
            p = self.last_p_val
        else:
            return torch.tensor(np.zeros((1,b["height"], b["width"])))
        perl_dict = {}
        for k in self._wavelet_params:
            if k in p:
                perl_dict[k] = p[k]
            else:
                perl_dict[k] = self._wavelet_params[k]
        #noise = perlin(**perl_dict)  
        noise = _generate_wavelet_noise_image(b["height"], b["width"], 
                                              self.params["num_wavelets"], 
                                              p)
        
        perturbation = np.zeros((1, self.img.shape[1], self.img.shape[2]))
        perturbation[:,b["top"]:b["top"]+b["height"],b["left"]:b["left"]+b["width"]] += noise
        perturbation = sigmoid(p["scale"]*perturbation)
        return torch.Tensor(perturbation)#.unsqueeze(0)
        
        
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
        #for k in p:
        #    self.writer.add_scalar(k, p[k], global_step=self.query_counter)
        self.writer.add_scalar("scale", p["scale"], global_step=self.query_counter)
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
                include_error_as_positive=self.params["include_error_as_positive"],
                use_scores=self.params["use_scores"]
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
    
        
        
        
    
                   
