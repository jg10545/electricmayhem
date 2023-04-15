import numpy as np 
import torch
from tqdm import tqdm
import json
import os
import time
import kornia.geometry.transform

from ax.service.ax_client import AxClient, ObjectiveProperties

from electricmayhem._graphite import BlackBoxPatchTrainer
from electricmayhem._perlin import BayesianPerlinNoisePatchTrainer





class BlackBoxOptimizer():
    """
    Wrapper class that uses AX to search hyperparameter space for
    the BlackBoxPatchTrainer class
    """
    
    def __init__(self, img, initial_mask, final_mask, detect_func, logdir,
                 budget=20000,
                 num_augments=[10,200], q=[5,50], 
                 beta=[0.01, 5], downsample=[1,2,4,8,16],
                 aug_params={}, eval_augments=1000, 
                 num_channels=3, perturbation=None, 
                 include_error_as_positive=False,
                 eval_func=None,
                 fixed_augs=None,
                 mlflow_uri=None,
                 experiment_name="graphite_optimization", 
                 load_from_json_file=None):
        """
        :img: torch.Tensor in channel-first format; victim image
        :initial_mask: torch.Tensor in channel-first format; initial mask
        :final_mask: torch.Tensor in channel-first format; final mask
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in. this directory needs to
            already exist. individual runs will be saved in subdirectories
        :budget: int; query budget for each run
        :num_augments: list of two integers; range of values for num_augments
        :q: list of two integers; range of values for q
        :beta: list of two floats; range of values for beta
        :downsample: list of two integers; range of values for factors to downsample
            perturbation by. So a downsample of 1 will create a perturbation of the
            same size as the victim; 2 will have each dimension cut in half
        :aug_params: dictionary of augmentation paramaters
        :eval_augments: int or list of dictionaries; number of augmentations to
            evaluate on
        :num_channels: 1 or 3; number of channels for the perturbation
        :perturbation: torch.Tensor in channel-first format; perturbation to start
            from. if None, initialize a gray tensor for each run.
        :fixed_augs:
        :mflow_uri: string; URI of MLflow server
        :experiment_name: string; name of experiment. Will be used both for AX
            and MLFlow
        :load_from_json_file: string; pass the location of a previous experiments'
            JSON file to start from.
        """
        self.C, self.H, self.W = img.shape
        self.num_channels = num_channels
        self.logdir = logdir
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.aug_params = aug_params
        self.eval_augments = eval_augments
        self.inputs = [img, initial_mask, final_mask, detect_func]
        self.budget = budget
        self.perturbation = None
        self.fixed_augs = fixed_augs
        self.include_error_as_positive = include_error_as_positive
        self.eval_func = eval_func
        
        # if we're resuming from a previous experiment, load it here.
        if load_from_json_file is not None:
            self.client = AxClient.load_from_json_file(load_from_json_file)
        else:
            self.client = AxClient()    
            # set up the experiment!
            self.params = self._build_params(num_augments, q, beta, downsample)
            self.client.create_experiment(
                name=experiment_name,
                parameters=self.params,
                objectives={"eval_transform_robustness":ObjectiveProperties(minimize=False)},
            )
        

        
    
    def _build_params(self, num_augments, q, beta, downsample):
        """
        Format parameter ranges for AX
        """
        return [
            {"name":"num_augments",
             "type":"range",
             "bounds":num_augments,
             "value_type":"int",
             "log_scale":True},
            
            {"name":"q",
             "type":"range",
             "bounds":q,
             "value_type":"int",
             "log_scale":True},
            
            {"name":"beta",
             "type":"range",
             "bounds":beta,
             "value_type":"float",
             "log_scale":True},
            
            {"name":"downsample",
             "type":"choice", #"range",
             #"bounds":downsample,
             "value_type":"int",
             "values":downsample,
             "is_ordered":True},
             #"log_scale":False},
         ]
    

    
    def _evaluate_trial(self, p):
        """
        
        """
        # build a trainer for this step of the experiment
        logdir = os.path.join(self.logdir, str(len(list(os.listdir(self.logdir)))))
        
        # initialize a new perturbation
        H = int(self.H/p["downsample"])
        W = int(self.W/p["downsample"])
        if self.perturbation is None:
            perturbation = 0.5*np.ones((self.num_channels, H, W))
            perturbation = torch.Tensor(perturbation)
        else:
            perturbation = self.perturbation.clone().unsqueeze(0)
            perturbation = kornia.geometry.transform.resize(perturbation, (H,W))
            perturbation = perturbation.squeeze(0)
        
        trainer = BlackBoxPatchTrainer(*self.inputs, logdir,
                                       num_augments=p["num_augments"],
                                       q=p["q"],
                                       beta=p["beta"],
                                       aug_params=self.aug_params,
                                       eval_augments=self.eval_augments,
                                       perturbation=perturbation,
                                       extra_params={"downsample":p["downsample"]},
                                       fixed_augs=self.fixed_augs,
                                       mlflow_uri=self.mlflow_uri,
                                       experiment_name=self.experiment_name,
                                       include_error_as_positive=self.include_error_as_positive,
                                       eval_func=self.eval_func)
        # fit the patch
        trainer.fit(budget=self.budget)
        
        # get TR and SEM results to send back to Bayesian optimizer. If we didn't
        # get through at least one epoch, return zero for both
        if hasattr(trainer, "tr_dict"):
            d = {"eval_transform_robustness":(trainer.tr_dict["tr"],
                                              trainer.tr_dict["sem"])}
        else:
            d = {"eval_transform_robustness":(0,0)}
        # get the trainer out of memory- had issues with mlflow getting confused if
        # we just do this in a for loop
        del(trainer)
        time.sleep(5)
        # save metadata about the AX client to JSON
        j = self.client.to_json_snapshot()
        json.dump(j, open(os.path.join(self.logdir, "log.json"), "w"))
            
        return d
    
    
    def fit(self, n=100):
        """
        Run optimization experiments
        
        :n: int; number of runs
        """
        for i in tqdm(range(n)):
            parameters, trial_index = self.client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            self.client.complete_trial(trial_index=trial_index,
                                       raw_data=self._evaluate_trial(parameters))
            
            
            
            
            

class PerlinOptimizer():
    """
    Wrapper class that uses AX to search hyperparameter space for
    the BayesianPerlinNoisePatchTrainer class
    """
    
    def __init__(self, img, final_mask, detect_func, logdir,
                 budget=20000,
                 num_augments=[10,200], 
                 num_sobol=[5,25],
                 max_freq=[0.01,1],
                 aug_params={}, eval_augments=1000, 
                 num_channels=3,  
                 include_error_as_positive=False,
                 eval_func=None,
                 fixed_augs=None,
                 mlflow_uri=None,
                 experiment_name="graphite_optimization", 
                 load_from_json_file=None):
        """
        :img: torch.Tensor in channel-first format; victim image
        :initial_mask: torch.Tensor in channel-first format; initial mask
        :final_mask: torch.Tensor in channel-first format; final mask
        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in. this directory needs to
            already exist. individual runs will be saved in subdirectories
        :budget: int; query budget for each run
        :num_augments: list of two integers; range of values for num_augments
        :num_sobol:
        :max_freq:
        :aug_params: dictionary of augmentation paramaters
        :eval_augments: int or list of dictionaries; number of augmentations to
            evaluate on
        :num_channels: 1 or 3; number of channels for the perturbation
        :perturbation: torch.Tensor in channel-first format; perturbation to start
            from. if None, initialize a gray tensor for each run.
        :fixed_augs:
        :mflow_uri: string; URI of MLflow server
        :experiment_name: string; name of experiment. Will be used both for AX
            and MLFlow
        :load_from_json_file: string; pass the location of a previous experiments'
            JSON file to start from.
        """
        self.C, self.H, self.W = img.shape
        self.num_channels = num_channels
        self.logdir = logdir
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.aug_params = aug_params
        self.eval_augments = eval_augments
        self.inputs = [img, final_mask, detect_func]
        self.budget = budget
        self.perturbation = None
        self.fixed_augs = fixed_augs
        self.include_error_as_positive = include_error_as_positive
        self.eval_func = eval_func
        
        
        def sample():
            return {
                "num_augments":int(np.random.randint(num_augments[0], num_augments[1])),
                "num_sobol":int(np.random.randint(num_sobol[0], num_sobol[1])),
                "max_freq":float(np.random.uniform(max_freq[0], max_freq[1]))
                }
        self.sample = sample
        #"""
        # if we're resuming from a previous experiment, load it here.
        if load_from_json_file is not None:
            self.client = AxClient.load_from_json_file(load_from_json_file)
        else:
            self.client = AxClient()    
            # set up the experiment!
            self.params = self._build_params(num_augments, num_sobol, max_freq)
            self.client.create_experiment(
                name=experiment_name,
                parameters=self.params,
                objectives={"eval_transform_robustness":ObjectiveProperties(minimize=False)},
            )#"""
        

        
    
    def _build_params(self, num_augments, num_sobol, max_freq):
        """
        Format parameter ranges for AX
        """
        return [
            {"name":"num_augments",
             "type":"range",
             "bounds":num_augments,
             "value_type":"int",
             "log_scale":True},
            
            {"name":"num_sobol",
             "type":"range",
             "bounds":num_sobol,
             "value_type":"int",
             "log_scale":True},
            
            {"name":"max_freq",
             "type":"range",
             "bounds":max_freq,
             "value_type":"float",
             "log_scale":True},
         ]
    

    
    def _evaluate_trial(self, p):
        """
        
        """
        print(p)
        # build a trainer for this step of the experiment
        logdir = os.path.join(self.logdir, str(len(list(os.listdir(self.logdir)))))

        trainer = BayesianPerlinNoisePatchTrainer(
            *self.inputs, logdir,
            **p,
            #num_augments = p["num_augments"], #np.random.randint(5,10),#5,#int(p["num_augments"]),
            #num_sobol = p["num_sobol"],
            #max_freq = p["max_freq"],
            aug_params = self.aug_params,
            eval_augments = self.eval_augments,
            fixed_augs = self.fixed_augs,
            include_error_as_positive = self.include_error_as_positive,
            mlflow_uri = self.mlflow_uri,
            experiment_name = self.experiment_name,
            eval_func = self.eval_func)
        trainer.fit(budget=self.budget, eval_every=10*self.budget)
        trainer.evaluate(use_best=True)
        
        # get TR and SEM results to send back to Bayesian optimizer. If we didn't
        # get through at least one epoch, return zero for both
        if hasattr(trainer, "tr_dict"):
            d = {"eval_transform_robustness":(trainer.tr_dict["tr"],
                                              trainer.tr_dict["sem"])}
        else:
            d = {"eval_transform_robustness":(0,0)}
        # get the trainer out of memory- had issues with mlflow getting confused if
        # we just do this in a for loop
        del(trainer)
        time.sleep(5)
        # save metadata about the AX client to JSON
        j = self.client.to_json_snapshot()
        json.dump(j, open(os.path.join(self.logdir, "log.json"), "w"))
            
        return d
    
    
    def fit(self, n=100):
        """
        Run optimization experiments
        
        :n: int; number of runs
        """
        for i in tqdm(range(n)):
            #"""
            parameters, trial_index = self.client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            self.client.complete_trial(trial_index=trial_index,
                                       raw_data=self._evaluate_trial(parameters))#"""
            #self._evaluate_trial(self.sample())