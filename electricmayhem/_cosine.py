import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from PIL import Image
import torch.utils.tensorboard
import yaml, os, json
from ax.service.ax_client import AxClient, ObjectiveProperties
import ax.modelbridge.generation_strategy 
import ax.modelbridge.registry

from electricmayhem import _augment
from electricmayhem._graphite import estimate_transform_robustness
from electricmayhem._perlin import normalize, _get_patch_outer_box_from_mask, BayesianPerlinNoisePatchTrainer
from electricmayhem._baxus import BAxUS

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


def _inverse_cosine_transform(z, latent_shape, patch_shape):
    """
    :z:
    :latent_shape: tuple (H',W'); shape to resize z to before taking IDCT
    :patch_shape: tuple (H,W); shape to resize transformed patch to
    """
    z = z.reshape(latent_shape)
    
    zprime = np.zeros(patch_shape)
    zprime[:latent_shape[0],:latent_shape[1]] = z
        
    x = scipy.fft.idctn(zprime)
        
    x = np.expand_dims(x, 0)
    return normalize(x)



class BayesianCosinePatchTrainer(BayesianPerlinNoisePatchTrainer):
    """
    """
    
    def __init__(self, img, final_mask, detect_func, logdir,
                 num_augments=100, aug_params={}, 
                 eval_augments=1000,  
                 num_sobol=5,
                 freq_scale=0.5, fft=False,
                 include_error_as_positive=False,
                 extra_params={}, fixed_augs=None,
                 mlflow_uri=None, experiment_name="cosine_noise", eval_func=None,
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
        :num_sobol: int; number of Sobol sampling steps before switching to Gaussian
            process
        :freq_scale: work in frequency space downsampled from the patch size by this 
            factor in each direction. So freq_scale=0.5 reduces the dimension of the
            latent fourier vector by a factor of 4.
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
        self.freq_scale = freq_scale
        
        self.img = img
        self.final_mask = final_mask
        self.detect_func = detect_func
        self.pert_box = _get_patch_outer_box_from_mask(final_mask)
        self._cosine_params = {
                "H":self.pert_box["height"],
                "W":self.pert_box["width"],
               "freq_scale":freq_scale,
               "Hprime":int(freq_scale*self.pert_box["height"]),
               "Wprime":int(freq_scale*self.pert_box["width"]),
               "fft":fft,
            }
        self._cosine_params["d"] = self._cosine_params["Hprime"]*self._cosine_params["Wprime"]*2
        #self._perlin_params = {"H":self.pert_box["height"],
        #                       "W":self.pert_box["width"], 
        #                       "period_y":1, 
        #                       "period_x":1, 
        #                       "octave":2, 
        #                       "freq_sine":1, 
        #                       "lacunarity":2}
        #self.last_p_val = self._perlin_params
        self.z = np.random.normal(0, 1, size=self._cosine_params["d"])
        
        if isinstance(eval_augments, int):
            eval_augments = [_augment.generate_aug_params(**aug_params) for _ in range(eval_augments)]
        self.eval_augments = eval_augments
        
        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)
        
        self.aug_params = aug_params
        self.params = {"num_augments":num_augments,
                      "include_error_as_positive":include_error_as_positive,
                      "num_sobol":num_sobol,
                      "freq_scale":freq_scale,
                      "fft":fft}
        
        # set up BAxUS object
        self.baxus = BAxUS(self._run_trial, 
                           self._cosine_params["d"], 
                           num_sobol,0)
        # baxus object will go ahead and do the sobol sampling,
        # so update counter accordingly
        self.query_counter += num_sobol*num_augments
        #print(self.baxus.X_baxus_target)
        #print(self.baxus.Y_baxus)
        
        # set up Ax client
        #if load_from_json_file is not None:
        #    self.client = AxClient.load_from_json_file(load_from_json_file)
        #else:
        #    model = ax.modelbridge.registry.Models.GPEI
        #    gs = ax.modelbridge.generation_strategy.GenerationStrategy(
        #        steps=[
        #            # Quasi-random initialization step
        #            ax.modelbridge.generation_strategy.GenerationStep(
        #                model=ax.modelbridge.registry.Models.SOBOL,
        #                num_trials=num_sobol,  
        #                ),
        #            # Bayesian optimization step using the custom acquisition function
        #            ax.modelbridge.generation_strategy.GenerationStep(
        #                model=model,
        #                num_trials=-1, 
        #                ),
        #            ]
        #        )
        #    
        #    self.client = AxClient(generation_strategy=gs,
        #                           verbose_logging=False)
        #    self.params = self._build_params(self._cosine_params["d"])
        #    self.client.create_experiment(
        #        name=experiment_name,
        #        parameters=self.params,
        #        objectives={"transform_robustness":ObjectiveProperties(minimize=False)}
        #        )
        
        
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump({"params":self.params, "aug_params":self.aug_params,
                   "extra_params":self.extra_params},
                  open(os.path.join(logdir, "config.yml"), "w"))
        
        
    def _build_params(self, d):
        params = [{
            "name":f"z_{i}",
            "type":"range",
            "value_type":"float",
            "bounds":[-1,1]
            } for i in range(d)]
        
        return params
        
    def _generate_perturbation(self, z=None, **kwargs):
        """
        kwargs overwrite defaults in self._perlin_params
        """
        b = self.pert_box
        if z is None:
            z = self.z
        #if len(kwargs) > 0:
        #    z = np.array([kwargs[f"z_{i}"] for i in range(len(kwargs))])
        if not isinstance(z, np.ndarray):
            z = z.numpy()
        
        p = self._cosine_params
        if p["fft"]:
            noise = _inverse_fft(z,
                                          (p["Hprime"], p["Wprime"]),
                                          (p["H"], p["W"]))
        else:
            noise = _inverse_cosine_transform(z,
                                          (p["Hprime"], p["Wprime"]),
                                          (p["H"], p["W"]))
 
        perturbation = np.zeros((1, self.img.shape[1], self.img.shape[2]))
        perturbation[:,b["top"]:b["top"]+b["height"],b["left"]:b["left"]+b["width"]] += noise
        return torch.Tensor(perturbation)
        
    
    
    def _run_trial(self, z):
        """
        
        """
        #for k in p:
        #    self.writer.add_scalar(k, p[k], global_step=self.query_counter)
        #self.last_p_val = p
        #z = np.array([z[f"z_{i}"] for i in range(len(z))])
        self.z = z
        # get the new perturbation
        self.perturbation = self._generate_perturbation(z=z)
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
        

        #d = {"transform_robustness":(tr_dict["tr"], tr_dict["sem"])}
        #j = self.client.to_json_snapshot()
        #json.dump(j, open(os.path.join(self.logdir, "log.json"), "w"))
        #return d
        return tr_dict["tr"]
    
    def _run_one_epoch(self, lrs=None, budget=None):
        #parameters, trial_index = self.client.get_next_trial()
        #self.client.complete_trial(trial_index=trial_index, 
        #                           raw_data=self._run_trial(parameters))
        self.baxus.run_one_step()
        
    
    def evaluate(self, use_best=False):
        """
        Run a suite of evaluation tests on the test augmentations.
        """
        if use_best:
            #ind, p, _ = self.client.get_best_trial()
            z = self.baxus.get_best_X()
            #self.perturbation = self._generate_perturbation(**p)
            self.perturbation = self._generate_perturbation(z=z)
        
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
        
    
        
    def contour_plot(self):
        """
        
        """
        assert False, "NOT IMPLEMENTED"