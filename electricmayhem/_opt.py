import numpy as np
import torch
from tqdm import tqdm
import json
import os
import time

from ax.service.ax_client import AxClient, ObjectiveProperties

from electricmayhem._graphite import BlackBoxPatchTrainer





class BlackBoxOptimizer():
    """
    Wrapper class that uses AX to search hyperparameter space for
    the BlackBoxPatchTrainer class
    """
    
    def __init__(self, img, initial_mask, final_mask, detect_func, logdir,
                 budget=20000,
                 num_augments=(10,200), q=(5,50), 
                 beta=(0.01, 5), downsample=(1,10),
                 aug_params={}, eval_augments=1000, mlflow_uri=None,
                 experiment_name="graphite_optimization", 
                 json_path=None,
                 load_from_json_file=None):
        """
        
        """
        self.C, self.H, self.W = img.shape
        self.logdir = logdir
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.json_path = json_path
        self.aug_params = aug_params
        self.eval_augments = eval_augments
        self.inputs = [img, initial_mask, final_mask, detect_func]
        self.budget = budget
        
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
        

        pass
    
    def _build_params(self, num_augments, q, beta, downsample):
        """
        Format parameter ranges for AX
        """
        return [
            {"name":"num_augments",
             "type":"range",
             "bounds":num_augments,
             "value_type":"int",
             "log_scale":False},
            
            {"name":"q",
             "type":"range",
             "bounds":q,
             "value_type":"int",
             "log_scale":False},
            
            {"name":"beta",
             "type":"range",
             "bounds":beta,
             "value_type":"float",
             "log_scale":False},
            
            {"name":"downsample",
             "type":"range",
             "bounds":downsample,
             "value_type":"int",
             "log_scale":False},
         ]
    

    
    def _evaluate_trial(self, p):
        """
        
        """
        # build a trainer for this step of the experiment
        logdir = os.path.join(self.logdir, str(len(list(os.listdir(self.logdir)))))
        # initialize a new perturbation
        perturbation = 0.5*np.ones((self.C, int(self.H/p["downsample"]),
                                    int(self.W/p["downsample"])))
        perturbation = torch.Tensor(perturbation)
        
        trainer = BlackBoxPatchTrainer(*self.inputs, logdir,
                                       num_augments=p["num_augments"],
                                       q=p["q"],
                                       beta=p["beta"],
                                       aug_params=self.aug_params,
                                       eval_augments=self.eval_augments,
                                       perturbation=perturbation,
                                       extra_params={"downsample":p["downsample"]},
                                       mlflow_uri=self.mlflow_uri,
                                       experiment_name=self.experiment_name)
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
        """
        for i in tqdm(range(n)):
            parameters, trial_index = self.client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            self.client.complete_trial(trial_index=trial_index,
                                       raw_data=self._evaluate_trial(parameters))