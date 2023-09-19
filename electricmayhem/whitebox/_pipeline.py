import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard
import yaml
import mlflow
from tqdm import tqdm
import logging
import os
import json
import inspect

from ._util import _dict_of_tensors_to_dict_of_arrays, _concat_dicts_of_arrays
from ._util import _flatten_dict, _mlflow_description, _bootstrap_std
from ._opt import _create_ax_client

class PipelineBase(torch.nn.Module):
    """
    Base class for pipeline stages. When subclassing this, specify:
        -a forward() method that should have an input x, a boolean 'control' kwarg, 
            and accept arbitrary other kwargs
        -a get_last_sample_as_dict() method that returns any stochastic parameters as a
            JSON-serializable dict
    """
    name = "PipelineBase"
    
    def __init__(self, **kwargs):
        super(PipelineBase, self).__init__()
        self.params = kwargs
        
    def to_yaml(self):
        return yaml.dump(self.params, default_flow_style=False)
    
    def log_params_to_mlflow(self):
        # string-ified loss function can be too long for MLFlow so don't log it
        p = {p:self.params[p] for p in self.params if p != "loss"}
        mlflow.log_params(_flatten_dict(p))
        
    def forward(self, x, control=False, evaluate=False, **kwargs):
        return x
        
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {"foobar":"looks like some chucklehead forgot to define this function when they subclassed PipelineBase"}
        
    def __add__(self, y):
        # check to see if it's an electricmayhem object. if not assume it's
        # a pytorch model
        if not issubclass(type(y), PipelineBase):
            y = ModelWrapper(y)
            
        return Pipeline(self,y)
    
    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.name}**"

        
    
    
class ModelWrapper(PipelineBase):
    """
    Lightweight wrapper class for torch models
    """
    name = "ModelWrapper"
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.params = {}
        if model.training:
            logging.warn("model appears to be set to train mode. was this intentional?")
        
    def forward(self, x, control=False, evaluate=False, **kwargs):
        return self.model(x)
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}
    
    
class Pipeline(PipelineBase):
    """
    Class to manage a sequence of pipeline steps
    """
    name = "Pipeline"
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.steps = torch.nn.ModuleList() #[]
        for a in args:
            _ = self.__add__(a)
        self._defaults = {}
        self.global_step = 0
        self._logging_to_mlflow = False
        self._profiling = False
        
        
        self.params = {}
        for e,s in enumerate(self.steps):
            self.params[f"{e}_{s.name}"] = s.params
            
    def forward(self, x, control=False, evaluate=False, **kwargs):
        for a in self.steps:
            x = a(x, control=control, evaluate=False, **kwargs)
        return x
    
    def __add__(self, y):
        # check to see if it's an electricmayhem object. if not assume it's
        # a pytorch model
        if not issubclass(type(y), PipelineBase):
            y = ModelWrapper(y)
        self.steps.append(y)
        return self
    
    def save_yaml(self):
        """
        Save self.params to a YAML file
        """
        yamltext = yaml.dump(self.params, default_flow_style=False)
        if hasattr(self, "logdir"):
            open(os.path.join(self.logdir, "config.yml"),
                 "w").write(yamltext)
        return 
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for e, s in enumerate(self.steps):
            sampdict = s.get_last_sample_as_dict()
            for k in sampdict:
                outdict[f"{e}_{s.name}_{k}"] = sampdict[k]
                
        return outdict
    
    def set_logging(self, logdir=None, mlflow_uri=None, experiment_name=None, 
                    description=None, tags={}, extra_params={}):
        """
        Configure TensorBoard and MLFlow for logging results
        
        :logdir: string; path to directory for saving tensorboard logs
        :mlflow_uri: string; URI of MLFlow server
        :experiment_name: string; name to use for MLflow experiment
        :description: string; markdown-formatted description for MLFlow run
        :tags: MLFlow tags
        :extra_params: dict containing exogenous parameters
        """
        self.params["extra"] = extra_params
        if logdir is not None:
            self.logdir = logdir
            self.writer = torch.utils.tensorboard.SummaryWriter(logdir)

        if (mlflow_uri is not None)&(experiment_name is not None):
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            
            alltags = {"logdir":logdir}
            for k in tags:
                alltags[k] = tags[k]
            if description is None:
                description = _mlflow_description(self)
            self.activerun = mlflow.start_run(description=description,
                                              tags=alltags)
            if len(extra_params) > 0:
                mlflow.log_params(extra_params)
            self._logging_to_mlflow = True
        elif (mlflow_uri is not None)&(experiment_name is not None):
            logging.warning("both a server URI and experiment name are required for MLFlow")
            
    def _get_profiler(self, wait=1, warmup=1, active=3, repeat=1):
        self._profiling = True
        self._stop_profiling_on = (wait+warmup+active)*repeat
        prof =  torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, 
                                             active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        prof.start()
        return prof
            
    def initialize_patch_params(self, patch_shape=None, patch=None, single_patch=True, device=None):
        """
        Generate parameters for an untrained patch uniformly on the unit interval.
        
        :patch_shape: tuple; dimensions of patch parameters. will be sampled on the
            unit interval
        :patch: torch.Tensor; instead of passing parameters you can just pass your
            own initialized patch
        :single_patch: bool; True if this is a single patch param and False if it's
            a batch of them.
        :device: which device to initialize to
        
        Saves to self.patch_params
        """
        # figure out which device the model is currently on
        if device is None:
            device = "cpu"
            for n in self.parameters():
                device = n.device
                break
            #device = next(self.parameters()).device
        if (patch_shape is not None)&(patch is None):
            patch = torch.zeros(patch_shape, dtype=torch.float32).uniform_(0,1)

        self._defaults["patch_param_shape"] = patch.shape    
        self.patch_params = patch.to(device)
        self._single_patch = single_patch
        
    def _get_learning_rate(self):
        p = self.params["training"]
        if p.get("lr_decay", None) == "cosine":
            lr = p["learning_rate"]*np.cos(self.global_step*np.pi/(2*p["num_steps"]))
        else:
            lr = p["learning_rate"]
        
        self._log_scalars(learning_rate=lr)
        return lr
        
     
    def _log_images(self, **kwargs):
        """
        log images to tensorboard
        """
        if hasattr(self, "writer"):
            for k in kwargs:
                self.writer.add_image(k, kwargs[k], global_step=self.global_step)
        
    def _log_image_to_mlflow(self, img, filename):
        if self._logging_to_mlflow:
            if len(img.shape) == 3:
                # convert from channel-first to channel-last
                img = img.permute(1,2,0).detach().cpu().numpy()
                mlflow.log_image(img, filename)

    def _log_scalars(self, mlflow_metric=False, **kwargs):
        """
        log scalars
        """
        if hasattr(self, "writer"):
            for k in kwargs:
                self.writer.add_scalar(k, kwargs[k], global_step=self.global_step)
        if mlflow_metric&self._logging_to_mlflow:
            mlflow.log_metrics(kwargs, step=self.global_step)
            
    def _log_histograms(self, **kwargs):
        """
        log scalars
        """
        if hasattr(self, "writer"):
            for k in kwargs:
                self.writer.add_histogram(k, kwargs[k], global_step=self.global_step)
        
    def set_loss(self, lossfunc, test_patch_shape=(2,3,64,64)):
        """
        Set a loss function for training a patch. lossfunc should input:
            :outputs: model outputs
            :patchparam: the parameterization of the batch of patches
            
        lossfunc should output a dictionary containing one key for each term in
        the loss function (or other metrics you want to compute) and each value
        should be elementwise values of that measure.
        
        When using loss functions from torch.nn, make sure to set
        "reduce=False" so that it returns elementwise loss.
        
        """
        self.loss = lossfunc
        
        if test_patch_shape is not None:
            test_patch = torch.ones(test_patch_shape, dtype=torch.float32).uniform_(0,1)
            model_output = self(test_patch)
            lossdict = lossfunc(model_output, test_patch)
            assert isinstance(lossdict, dict), "this loss function doesn't appear to generate a dictionary"
            for k in lossdict:
                assert isinstance(lossdict[k], torch.Tensor), f"loss function output {k} doesn't appear to be a Tensor"
                assert lossdict[k].shape == (test_patch_shape[0],), f"loss function output {k} doesn't appear to return the correct shape"
            # record loss dictionary keys
            self._lossdictkeys = list(lossdict.keys())
            
        self.params["loss"] = inspect.getsource(self.loss)
        
    def evaluate(self, batch_size, num_eval_steps):
        """
        Run a set of evaluation batches and log results.
        """
        #p = self.params["training"]
        patch_params = self.patch_params
        #if batch_size is None:
        #    batch_size = p["batch_size"]
        #if num_eval_steps is None:
        #    num_eval_steps = p["num_eval_steps"]
            
        # stack into a batch of patches
        if self._single_patch:
            patchbatch = torch.stack([patch_params for _ in range(batch_size)], 0)
        else:
            patchbatch = patch_params
            
        # store loss function outputs for each eval batch
        results = []
        # store sampled parameters for each eval batch
        samples = []
        
        # for each eval step
        for _ in range(num_eval_steps):
            stepdict = {}
            # run a batch through with the patch included
            result_patch = _dict_of_tensors_to_dict_of_arrays(self.loss(self(patchbatch, evaluate=True),
                                                                        patchbatch))
            # then a control batch; no patch but same parameters
            result_control = _dict_of_tensors_to_dict_of_arrays(self.loss(self(patchbatch, control=True,
                                                                               evaluate=True),
                                                                              patchbatch))
            for k in result_patch:
                stepdict[f"{k}_patch"] = result_patch[k]
                stepdict[f"{k}_control"] = result_control[k]
                stepdict[f"{k}_delta"] = result_patch[k] - result_control[k]
            results.append(stepdict)
            samples.append(self.get_last_sample_as_dict())
        
        # concatenate list of dicts
        results = _concat_dicts_of_arrays(*results)
        samples = _concat_dicts_of_arrays(*samples)
        self.results = results
        # combine results and sampled parameters into a dataframe and save.
        for r in results:
            samples[r] = results[r]
        self.df = pd.DataFrame(samples)
        self.df.to_csv(os.path.join(self.logdir, "eval_results.csv"), index=False)
        # record distributions
        self._log_histograms(**{f"eval_{k}_distribution":results[k] for k in results if "_control" not in k})
        # record averages
        meanresults = {f"eval_{k}":np.mean(results[k]) for k in results if "_control" not in k}
        self._log_scalars(**meanresults)
        
        # record metrics to mlflow
        if self._logging_to_mlflow:
            mlflow.log_metrics(meanresults, step=self.global_step)
        
        # if patch_params has the shape of an image we should just log it as an image
        if len(patch_params.shape) == 3:
            if patch_params.shape[0] == 3:
                self._log_images(patch_params=patch_params)
            elif patch_params.shape[0] == 1:
                self._log_images(patch_params=torch.concat([patch_params for _ in range(3)], 0))
                
        
    def train(self, batch_size, num_steps, learning_rate=1e-2, eval_every=1000, num_eval_steps=10, 
              accumulate=1, lr_decay='cosine', profile=0, **kwargs):
        """
        Patch training loop. Expects that you've already called initialize_patch_params() and
        set_loss().
        
        :batch_size: number of implantation/composition parameters to run at a time
        
        :num_steps: int; number of attack steps to take
        :learning_rate:
        :eval_every: int; how many steps before running self.evaluate()
        :num_eval_steps: int; number of evaluation batches to run
        :accumulate: int; how many batches to accumulate gradients across before updating patch
        :lr_decay: None or "cosine"
        :profile: int; if above zero, run pytorch profiler this many steps.
        :kwargs: additional training parameters to save. at least one of the terms in your
            loss function should have a weight here.
        """
        # warn the user if they didn't pass any keys from the loss dict
        if hasattr(self, "_lossdictkeys"):
            if len(set(self._lossdictkeys)&set(kwargs.keys())) == 0:
                logging.warning("no weights given for any terms in your loss dictionary")
        # record the training parameters
        trainparams = {"batch_size":batch_size,
                    "learning_rate":learning_rate, "num_steps":num_steps,
                    "eval_every":eval_every, "num_eval_steps":num_eval_steps,
                    "accumulate":accumulate, "lr_decay":lr_decay}
        for k in kwargs:
            trainparams[k] = kwargs[k]
        
        self.params["training"] = trainparams
        # dump experiment as YAML to log directory
        self.save_yaml()
        self.log_params_to_mlflow()
            
        if profile > 0:
            prof = self._get_profiler(active=profile)
        
        # copy patch and turn on gradients
        patch_params = self.patch_params.clone().detach().requires_grad_(True)
        # make a tensor to hold the gradients
        gradient = torch.zeros_like(patch_params)
        
        for i in tqdm(range(num_steps)):
            # stack into a batch of patches
            if self._single_patch:
                patchbatch = torch.stack([patch_params for _ in range(batch_size)], 0)
            else:
                patchbatch = patch_params
            
            # run through the pipeline
            outputs = self(patchbatch)
            
            # compute loss
            lossdict = self.loss(outputs, patchbatch)
            loss = 0
            record = {}
            for k in lossdict:
                meanloss = torch.mean(lossdict[k])
                record[k] = meanloss
                if k in kwargs:
                    loss += kwargs[k]*meanloss
            # save metrics to tensorboard
            self._log_scalars(mlflow_metric=False, **record)
            
            # estimate gradients
            gradient += torch.autograd.grad(loss, patch_params)[0]/accumulate
            self._log_scalars(gradient_norm = torch.mean(gradient**2))
            
            # if this is an update step- update patch, clamp to unit interval
            if (i+1)%accumulate == 0:
                patch_params = patch_params.detach() - self._get_learning_rate()*gradient.sign()
                patch_params = patch_params.clamp(0,1).detach().requires_grad_(True)
                gradient = torch.zeros_like(patch_params)
                self.patch_params = patch_params.clone().detach()
                
            
            # if this is an evaluate step- run evaluation
            if ((i+1)%eval_every == 0)&(eval_every > 0):
                self.evaluate(batch_size, num_eval_steps)
                
            # if we're profiling, update the profiler
            if self._profiling:
                prof.step()
                if self.global_step > self._stop_profiling_on+1:
                    prof.stop()
                    self._profiling = False
                    self.prof = prof
                
            self.global_step += 1
                
        # finished training- save a copy of the patch tensor
        self.patch_params = patch_params.clone().detach()
        # wrap up mlflow logging
        self._log_image_to_mlflow(patch_params, "patch.png")
        return self.patch_params
    
    def visualize_eval_results(self):
        """
        Use ipywidgets to visualize eval results within a notebook
        """
        import ipywidgets
        def scatter(x, y, c):
            plt.cla()
            plt.scatter(self.df[x].values, self.df[y].values, 
                        c=self.df[c].values, alpha=0.5)
            plt.colorbar()
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(c)
        columns = list(self.df.columns)
        colors = list(self.results.keys())
        ipywidgets.interact(scatter, x=columns, y=columns, c=colors)
        
    def __del__(self):
        if self._logging_to_mlflow: mlflow.end_run()
    
    def optimize(self, objective, logdir, patch_shape, N, num_steps, 
                 batch_size, num_eval_steps=10, mlflow_uri=None, experiment_name=None, 
                      extra_params={}, minimize=True, lr_decay="cosine",
                      **params):
        """
        foo
        """
        self.client = _create_ax_client(objective, minimize=minimize, **params)
        
        def _evaluate_trial(p):
            self.train(batch_size, num_steps, eval_every=-1, lr_decay=lr_decay, **p)
            self.evaluate(batch_size, num_eval_steps)
            result_mean = np.mean(self.results[objective])
            sem = _bootstrap_std(self.results[objective])
            return {objective:(result_mean, sem)}
        
        # for each trial
        for i in tqdm(range(N)):
            self.global_step = 0
            # point the logger to a new subdirectory
            ld = os.path.join(logdir, str(i))
            self.set_logging(logdir=ld, mlflow_uri=mlflow_uri,
                             experiment_name=experiment_name,
                             extra_params=extra_params)
            # create a new patch
            self.initialize_patch_params(patch_shape=patch_shape)
            # run the trial
            parameters, trial_index = self.client.get_next_trial()
            self.client.complete_trial(trial_index=trial_index,
                                       raw_data=_evaluate_trial(parameters))
            j = self.client.to_json_snapshot()
            json.dump(j, open(os.path.join(logdir, "log.json"), "w"))
            if self._logging_to_mlflow: mlflow.end_run()
            
        return False
        
    