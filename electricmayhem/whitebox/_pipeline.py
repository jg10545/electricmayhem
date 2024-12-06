import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard
import torch.multiprocessing as mp
import yaml
import mlflow
from tqdm import tqdm
import logging
import os
import json
import inspect
import copy
import multiprocessing

from electricmayhem import __version__
from ._util import _dict_of_tensors_to_dict_of_arrays, _concat_dicts_of_arrays
from ._util import _flatten_dict, _mlflow_description, _bootstrap_std
from ._opt import _create_ax_client
from ._multi import PatchWrapper, _run_worker_training_loop
from .optim import _get_optimizer_and_scheduler

import dill


class PipelineBase(torch.nn.Module):
    """
    Base class for pipeline stages. When subclassing this, specify:
        -a forward() method that should have an input x, a boolean 'control' kwarg,
            and accept arbitrary other kwargs
        -a get_last_sample_as_dict() method that returns any stochastic parameters as a
            JSON-serializable dict
    """
    def __init__(self, **kwargs):
        super(PipelineBase, self).__init__()
        self.params = kwargs
        self._logviz = True

    def to_yaml(self):
        return yaml.dump(self.params, default_flow_style=False)

    def log_params_to_mlflow(self):
        """
        When a Pipeline object saves its parameters to MLflow, it'll automatically pull
        together the self.params dictionaries for each stage in the pipeline.

        Is there anything OUTSIDE OF THE PARAMS DICT that this stage should record for all
        posterity when that happens? If so, put it here.
        """
        pass

    def _forward_single(self, x, control=False, evaluate=False, key=None, **kwargs):
        return x, kwargs


    def forward(self, x, control=False, evaluate=False, params={}, **kwargs):
        """
        If not overwritten- this function will call self._forward_single(), either
        routing a single image batch through or mapping across a ditionary of batches.
        """
        # single patch/image case
        if isinstance(x, torch.Tensor):
            return self._forward_single(x, control=control, evaluate=evaluate, params=params, 
                                        **kwargs)
        # multi patch/image case
        else:
            outdict = {}
            # decide which elements of the dictionary to operate on
            keys = self.params.get("keys", list(x.keys()))
            for k in x:
                if k in keys:
                    output, kwargs = self._forward_single(x[k], control=control, evaluate=evaluate,
                                                          params=params, key=k, **kwargs)
                    outdict[k] = output
                else:
                    outdict[k] = x[k]

            return outdict, kwargs

    def _apply_forward_to_dict(
        self, x, control=False, evaluate=False, params={}, **kwargs
    ):
        """
        DEPRECATE
        For pipeline stages that might input and output a dictionary of tensors (for
        example, applying some transformation to several patches before implanting
        them)
        """
        assert False, "DEPRECATE MEEEE"
        outdict = {}
        outkwargs = kwargs
        # are we using all the
        keys = list(x.keys())
        if hasattr(self, "keys"):
            if self.keys is not None:
                keys = self.keys
        # run all the tensors we're updating through the pipeline stage
        for k in keys:
            outdict[k], kw = self.forward(
                x[k], control=control, evaluate=evaluate, **kwargs
            )
            # add any new kwargs to the dictionary, adding the key for this tensor to
            # the label
            for j in kw:
                if j not in outkwargs:
                    outkwargs[f"{k}_{j}"] = kw[j]
        # any other tensors in the input dict that we're not operating on just get copied
        # into the new dict
        for k in x:
            if k not in keys:
                outdict[k] = x[k]

        return outdict, outkwargs

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {
            "foobar": "looks like some chucklehead forgot to define this function when they subclassed PipelineBase"
        }

    def __add__(self, y):
        # check to see if it's an electricmayhem object. if not assume it's
        # a pytorch model
        if not issubclass(type(y), PipelineBase):
            y = ModelWrapper(y)

        return Pipeline(self, y)

    def _log_image_to_mlflow(self, img, filename):
        if isinstance(img, dict):
            for k in img:
                self._log_image_to_mlflow(img[k], f"{k}_{filename}")
        if len(img.shape) == 3:
            # convert from channel-first to channel-last
            img = img.permute(1, 2, 0).detach().cpu().numpy()
            mlflow.log_image(img, filename)

    def get_description(self):
        """
        Return a markdown-formatted one-line string describing the pipeline step. Used for
        auto-populating a description for MLFlow.
        """
        return f"**{self.__class__.__name__}**"

    def log_vizualizations(self, x, x_control, writer, step, logging_to_mlflow=False):
        """ """
        pass

    def copy(self):
        """
        Perform a deepcopy AFTER MOVING TO THE CPU
        """
        self.cpu()
        return copy.deepcopy(self)

    def validate(self, x):
        """
        Overwrite this function to run any validation checks for this step
        of the pipeline, to catch common errors. Return True if it passes the
        checks and False otherwise. Use logging to explain what went wrong.
        """
        logging.info(f"no validation checks for {self.__class__.__name__}")
        return True


class ModelWrapper(PipelineBase):
    """
    Lightweight wrapper class for torch models
    """
    def __init__(self, model, eval_model=None):
        """
        :model: pytorch model or list/dict of models, in eval mode
        :eval_model: optional model or list/dict of models to use in eval steps
        """
        super().__init__()
        # wrap the model container if necessary
        self.model, self.wraptype = self._wrap(model)
        # wrap eval model container if necessary
        if eval_model is not None:
            self.eval_model, self.eval_wraptype = self._wrap(eval_model)
        else:
            self.eval_model = self.model
            self.eval_wraptype = self.wraptype

        self.params = {}

    def _wrap(self, x):
        """
        Wrap a container object if necessary
        """
        wraptype = "model"
        if isinstance(x, list):
            x = torch.nn.ModuleList(x)
            wraptype = "list"
        elif isinstance(x, dict):
            x = torch.nn.ModuleDict(x)
            wraptype = "dict"
        return x, wraptype

    def _call_wrapped(self, model, x):
        if isinstance(model, torch.nn.ModuleList):
            return [m(x) for m in model]
        elif isinstance(model, torch.nn.ModuleDict):
            return {m: model[m](x) for m in model}
        else:
            return model(x)

    def forward(self, x, control=False, evaluate=False, **kwargs):
        if evaluate:
            model = self.eval_model
        else:
            model = self.model

        return self._call_wrapped(model, x), kwargs

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}


def _update_patch_gradients(
    pipeline, batch_size, lossweights, accumulate=1, rho=0, clamp_to=(0, 1)
):
    """
    Input a Pipeline object and update the weights of this patch. Designed to be called from
    within the training loop. This function is a somewhat inelegant solution and the API may change
    in the future!

    The main reason to break this function out is to capture the code we'd need to do sharpness-aware
    minimization (SAM) during patch training. This option follows equation 4 of RETHINKING MODEL ENSEMBLE IN
    TRANSFER-BASED ADVERSARIAL ATTACKS by Chen et all instead of the original SAM paper (normalizing the
    gradient in the adversarial step by taking the sign instead of an L2 norm).

    The adversarial and forward step of optimization here is making two passes through the pipeline, so
    any implantation and augmentation parameters will have different values- this may add too much noise
    to the training loop; we'll see.

    I expect that this will NOT work with multi-GPU training.

    :pipeline: a Pipeline object with loss function and patch already initialized
    :batch_size: batch size to use for training
    :lossweights: dictionary of loss function weights
    :accumulate: gradient accumulation hyperparameter
    :rho: learning rate for adversarial gradient-ascent step
    :clamp_to: tuple or None; values to clamp patch to after updating

    Returns dictionary of disaggregated loss values and total loss tensor
    """

    def _getloss(outputs, extra_kwargs):
        lossdict = pipeline.loss(outputs, **extra_kwargs)
        loss = 0
        record = {}
        for k in lossdict:
            meanloss = torch.mean(lossdict[k])
            record[k] = meanloss
            # add this term to the loss function if a weight was included
            if k in lossweights:
                loss += lossweights[k] * meanloss / accumulate
        return lossdict, loss

    # patch_params is a PatchWrapper object; the batch_size arg will
    # return a stack of patches
    patchbatch = pipeline.patch_params(batch_size)

    # NORMAL CASE
    if rho == 0:
        # run through the pipeline
        outputs, k = pipeline(patchbatch)
        lossdict, loss = _getloss(outputs, k)

        # estimate gradients
        loss.backward()
    # SHARPNESS-AWARE CASE
    else:
        # make a copy of the batch
        batch_copy = patchbatch.clone().detach().requires_grad_(True)  # (N, C, H, W)

        # run it through pipeline and get the loss
        lossdict, loss = _getloss(*pipeline(batch_copy))

        # compute gradients and average across batch
        loss.backward()
        mean_grad = torch.mean(batch_copy.grad, dim=[0])  # (C,H,W)

        # normalize gradients
        grad_norm = torch.sqrt(torch.sum(mean_grad**2)) + 1e-8  # numerical stability

        # compute adversarially-perturbed x_adv
        if clamp_to is not None:
            x_adv = (
                torch.clamp(
                    pipeline.patch_params.patch + rho * mean_grad.sign(),
                    clamp_to[0],
                    clamp_to[1],
                )
                .detach()
                .requires_grad_(True)
            )
        else:
            x_adv = (
                (pipeline.patch_params.patch + rho * mean_grad.sign())
                .detach()
                .requires_grad_(True)
            )
        # now run x_adv through the pipeline and get loss
        patchbatch_adv = torch.stack([x_adv for _ in range(batch_size)], 0)
        # get sampled parameters from the pipeline to run through under the same conditions
        sampdict = pipeline.get_last_sample_as_dict()
        outputs_adv, k = pipeline(patchbatch_adv, **sampdict)
        lossdict_adv, loss = _getloss(outputs_adv, k)
        # compute gradients
        loss.backward()

        with torch.no_grad():
            # add gradient to single_patch.grad
            if pipeline.patch_params.patch.grad is None:
                pipeline.patch_params.patch.grad = x_adv.grad
            else:
                pipeline.patch_params.patch.grad += x_adv.grad

    return lossdict, loss


class Pipeline(PipelineBase):
    """
    Class to manage a sequence of pipeline steps
    """
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.steps = torch.nn.ModuleList()  # []
        for a in args:
            _ = self.__add__(a)
        # self._defaults = {} # TODO remove this?
        self.global_step = 0
        self._logging_to_mlflow = False
        self._profiling = False

        self.params = {"version": __version__}
        for e, s in enumerate(self.steps):
            self.params[f"{e}_{s.__class__.__name__}"] = s.params

        # These attributes will only be used when we're training in distributed
        # mode over multiple GPUs. In that case they'll be overwritten.
        self.rank = 0

    def forward(self, x, control=False, evaluate=False, **kwargs):
        # initialize a dictionary to propagate additional useful information through the pipeline
        extra_kwargs = {"input": x}
        # run through each pipeline stage sequentially
        for e, s in enumerate(self.steps):
            # pull out a dictionary of keyword arguments for this
            # stage of the pipeline
            key = f"{e}_{s.__class__.__name__}_"
            step_kwargs = {
                k.split(key)[-1]: kwargs[k] for k in kwargs if k.startswith(key)
            }
            x, extra_kwargs = s(
                x, control=control, evaluate=evaluate, **step_kwargs, **extra_kwargs
            )
        return x, extra_kwargs

    def __add__(self, y):
        # check to see if it's an electricmayhem object. if not assume it's
        # a pytorch model
        if not issubclass(type(y), PipelineBase):
            y = ModelWrapper(y)
        self.steps.append(y)
        # update parameter dict
        self.params = {"version": __version__}
        for e, s in enumerate(self.steps):
            self.params[f"{e}_{s.__class__.__name__}"] = s.params
        return self

    def save_yaml(self):
        """
        Save self.params to a YAML file
        """
        if self.rank == 0:
            yamltext = yaml.dump(self.params, default_flow_style=False)
            if hasattr(self, "logdir"):
                open(os.path.join(self.logdir, "config.yml"), "w").write(yamltext)
        return

    def log_params_to_mlflow(self):
        p = _flatten_dict(self.params)
        # mlflow has a limit of 500 characters for params.
        p = {k: p[k] for k in p if len(str(p[k])) < 500}
        mlflow.log_params(p)
        for s in self.steps:
            s.log_params_to_mlflow()

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for e, s in enumerate(self.steps):
            sampdict = s.get_last_sample_as_dict()
            for k in sampdict:
                outdict[f"{e}_{s.__class__.__name__}_{k}"] = sampdict[k]

        return outdict

    def set_logging(
        self,
        logdir=None,
        mlflow_uri=None,
        experiment_name=None,
        description=None,
        tags={},
        extra_params={},
    ):
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

        if (mlflow_uri is not None) & (experiment_name is not None):
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)

            alltags = {"logdir": logdir}
            for k in tags:
                alltags[k] = tags[k]
            if description is None:
                description = _mlflow_description(self)
            self.activerun = mlflow.start_run(description=description, tags=alltags)
            if len(extra_params) > 0:
                mlflow.log_params(extra_params)
            self._logging_to_mlflow = True
        elif (mlflow_uri is not None) & (experiment_name is not None):
            logging.warning(
                "both a server URI and experiment name are required for MLFlow"
            )

    def _get_profiler(self, wait=1, warmup=1, active=3, repeat=1):
        self._profiling = True
        self._stop_profiling_on = (wait + warmup + active) * repeat
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        return prof

    def initialize_patch_params(
        self, patch_shape=None, patch=None, single_patch=True, device=None
    ):
        """
        Generate parameters for an untrained patch uniformly on the unit interval.

        :patch_shape: dimensions of the patch or patches, WITHOUT batch dimension. when training a
            single patch this should be a tuple in channel-first format (e.g. (C,H,W)); when training
            multiple patches this should be a dictionary of tuples. patches will be sampled uniformly
            on the unit interval
        :patch: torch.Tensor; instead of passing parameters you can just pass your
            own initialized patch or dictionary of patches. will supercede patch_shape.
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
        if (patch_shape is not None) & (patch is None):
            # multi-patch case
            if isinstance(patch_shape, dict):
                patch = {
                    k: torch.zeros(patch_shape[k], dtype=torch.float32).uniform_(0, 1)
                    for k in patch_shape
                }
            # single patch case
            else:
                patch = torch.zeros(patch_shape, dtype=torch.float32).uniform_(0, 1)

        # self._defaults["patch_param_shape"] = patch.shape   # TODO remove this?
        # now wrap the patch in a torch.nn.Module subclass so that we can
        # distribute it if we need to
        if not isinstance(patch, PatchWrapper):
            # multi patch case
            if isinstance(patch, dict):
                patch = {
                    k: patch[k].clone().detach().requires_grad_(True) for k in patch
                }
            # single patch case
            else:
                patch = patch.clone().detach().requires_grad_(True)
            patch_wrapped = PatchWrapper(patch, single_patch=single_patch)

        self.patch_params = patch_wrapped.to(device)
        self._single_patch = single_patch

    def _get_learning_rate(self):
        p = self.params["training"]
        if p.get("lr_decay", None) == "cosine":
            lr = p["learning_rate"] * np.cos(
                self.global_step * np.pi / (2 * p["num_steps"])
            )
        else:
            lr = p["learning_rate"]

        self._log_scalars(learning_rate=lr)
        return lr

    def _log_images(self, **kwargs):
        """
        log images to tensorboard
        """
        if self.rank == 0:
            if hasattr(self, "writer"):
                for k in kwargs:
                    self.writer.add_image(k, kwargs[k], global_step=self.global_step)

    def validate(self, patch=None):
        """
        Run a test patch through the pipeline and check the validate() method of each
        stage. Returns True if no problems are raised.

        :patch: torch.Tensor containing a patch. If None, will look for a patch created by
            self.initialize_patch_params()
        """
        passed_all = True
        # get the patch and add a batch dimension
        if patch is None:
            assert hasattr(self, "patch_params"), "help i can't find a patch"
            patchbatch = self.patch_params(1)
        # multi patch case
        elif isinstance(patch, dict):
            patchbatch = {k:patch[k].unsqueeze(0) for k in patch}
        # single patch case
        else:
            patchbatch = patch.unsqueeze(0)

        x = patchbatch
        with torch.no_grad():
            # for each step in the pipeline
            for s in self.steps:
                # see if it passed that step's validation
                passed_step = s.validate(x)
                if not passed_step:
                    passed_all = False
                # run x through the step to prepare
                # for the next step
                x, _ = s(x)
        return passed_all

    def _log_scalars(self, mlflow_metric=False, **kwargs):
        """
        log scalars
        """
        if self.rank == 0:
            if hasattr(self, "writer"):
                for k in kwargs:
                    self.writer.add_scalar(k, kwargs[k], global_step=self.global_step)
            if mlflow_metric & self._logging_to_mlflow:
                mlflow.log_metrics(kwargs, step=self.global_step)

    def _log_histograms(self, **kwargs):
        """
        log scalars
        """
        if self.rank == 0:
            if hasattr(self, "writer"):
                for k in kwargs:
                    self.writer.add_histogram(
                        k, kwargs[k], global_step=self.global_step
                    )

    def set_loss(self, lossfunc, test_patch_shape=(2, 3, 64, 64)):
        """
        Set a loss function for training a patch and run a test batch through.

        :lossfunc: python function to compute loss function and evaluation metric terms. should
            input the pipeline outputs and have a **kwargs input for any additional data the
            pipeline creates. It should output a dictionary of 1D tensors (of length batchsize)
            giving the UNAGGREGATED values of each metric or loss term.
        :test_patch_shape: tuple (single patch) or dictionary of tuples (multi patch)
            giving the shape of a BATCH of patches to test the loss function with. Will
            generally be a 4-tuple (batchsize, C, H, W)

        When using loss functions from torch.nn, make sure to set
        "reduce=False" so that it returns elementwise loss.
        """
        self.loss = lossfunc

        if test_patch_shape is not None:
            # multi-patch case
            if isinstance(test_patch_shape, dict):
                test_patch = {
                    k: torch.ones(test_patch_shape[k], dtype=torch.float32).uniform_(
                        0, 1
                    )
                    for k in test_patch_shape
                }
                batch_size = test_patch_shape[list(test_patch_shape.keys())[0]][0]
            # single patch case
            else:
                test_patch = torch.ones(test_patch_shape, dtype=torch.float32).uniform_(
                    0, 1
                )
                batch_size = test_patch_shape[0]
            model_output, kwargs = self(test_patch)
            lossdict = lossfunc(model_output, **kwargs)
            assert isinstance(
                lossdict, dict
            ), "this loss function doesn't appear to generate a dictionary"
            for k in lossdict:
                assert isinstance(
                    lossdict[k], torch.Tensor
                ), f"loss function output {k} doesn't appear to be a Tensor"
                assert lossdict[k].shape == (
                    batch_size,
                ), f"loss function output {k} doesn't appear to return the correct shape; returned {lossdict[k].shape}"
            # record loss dictionary keys
            self._lossdictkeys = list(lossdict.keys())

        self.params["loss"] = inspect.getsource(self.loss)

    def log_vizualizations(self, patchbatch, *args, **kwargs):
        """
        Wraps the log_vizualizations method in each of the pipeline
        stages.
        """
        if self.rank > 0:
            return
        with torch.no_grad():
            x = patchbatch
            # multi patch case
            if isinstance(patchbatch, dict):
                x_control = {k: patchbatch[k].clone() for k in patchbatch}
            # single patch case
            else:
                x_control = patchbatch.clone()
            # run through each stage, running diagnostics on the
            # interim steps
            for s in self.steps:
                s.log_vizualizations(
                    x,
                    x_control,
                    self.writer,
                    self.global_step,
                    logging_to_mlflow=self._logging_to_mlflow,
                )
                x, _ = s(x, evaluate=True)
                x_control, _ = s(x_control, control=True, evaluate=True)

    def evaluate(self, batch_size, num_eval_steps, patchbatch=None):
        """
        Run a set of evaluation batches and log results.
        """
        if self.rank > 0:
            return
        if patchbatch is None:
            patch_params = self.patch_params
            # patch_params is a PatchWrapper object that will return a stack
            # of patches when called
            patchbatch = self.patch_params(batch_size)

        # store loss function outputs for each eval batch
        results = []
        # store sampled parameters for each eval batch
        samples = []

        # for each eval step
        for _ in range(num_eval_steps):
            stepdict = {}
            # run a batch through with the patch included
            output, kwargs = self(patchbatch, evaluate=True)
            result_patch = _dict_of_tensors_to_dict_of_arrays(
                self.loss(output, **kwargs)
            )
            # then a control batch; no patch but same parameters
            output, kwargs = self(patchbatch, evaluate=True, control=True)
            result_control = _dict_of_tensors_to_dict_of_arrays(
                self.loss(output, **kwargs)
            )

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
        saveto = os.path.join(self.logdir, "eval_results.csv")
        self.df.to_csv(saveto, index=False)

        # record distributions
        self._log_histograms(
            **{
                f"eval_{k}_distribution": results[k]
                for k in results
                if "_control" not in k
            }
        )
        # record averages
        meanresults = {
            f"eval_{k}": np.mean(results[k]) for k in results if "_control" not in k
        }
        self._log_scalars(**meanresults)

        # record metrics to mlflow
        if self._logging_to_mlflow:
            mlflow.log_metrics(meanresults, step=self.global_step)
            mlflow.log_artifact(saveto)

        self.log_vizualizations(patchbatch)

    def train_patch(
        self,
        batch_size,
        num_steps,
        learning_rate=1e-2,
        eval_every=1000,
        num_eval_steps=10,
        accumulate=1,
        lr_decay="cosine",
        optimizer="adam",
        profile=0,
        progressbar=True,
        clamp_to=(0, 1),
        rho=0,
        **kwargs,
    ):
        """
        Patch training loop. Expects that you've already called initialize_patch_params() and
        set_loss().

        :batch_size: number of implantation/composition parameters to run at a time

        :num_steps: int; number of attack steps to take
        :learning_rate:
        :eval_every: int; how many steps before running self.evaluate()
        :num_eval_steps: int; number of evaluation batches to run
        :accumulate: int; how many batches to accumulate gradients across before updating patch
        :lr_decay: "none", "cosine", "exponential", or "plateau"
        :profile: int; if above zero, run pytorch profiler this many steps.
        :progressbar: bool; if True, use tqdm to monitor progress
        :clamp_to: int or None; range to clip patch parameters to
        :rho: float; gradient ascent step size for sharpness-aware minization. 0 to disable.
        :kwargs: additional training parameters to save. at least one of the terms in your
            loss function should have a weight here.
        """
        # warn the user if they didn't pass any keys from the loss dict
        if hasattr(self, "_lossdictkeys"):
            if len(set(self._lossdictkeys) & set(kwargs.keys())) == 0:
                logging.error("no weights given for any terms in your loss dictionary")
            # warn the user if they passed a keyword argument that doesn't
            # match anything in params or the lossdict. i wasted a bunch
            # of time once when i was doing hyperparameter optimization
            # and fat-fingered one of the loss terms.
            for k in kwargs:
                if k not in self._lossdictkeys:
                    logging.warning(
                        f"param '{k}' not in loss dict keys; was that on purpose?"
                    )
        # record the training parameters
        trainparams = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
            "eval_every": eval_every,
            "num_eval_steps": num_eval_steps,
            "accumulate": accumulate,
            "lr_decay": lr_decay,
            "optimizer": optimizer,
            "rho": rho,
        }
        for k in kwargs:
            trainparams[k] = kwargs[k]

        self.params["training"] = trainparams
        # dump experiment as YAML to log directory
        self.save_yaml()
        if self._logging_to_mlflow:
            self.log_params_to_mlflow()

        if profile > 0:
            prof = self._get_profiler(active=profile)

        assert hasattr(
            self, "patch_params"
        ), "patch_params attribute not found- did you call Pipeline.initialize_patch_params()?"
        patch_params = self.patch_params
        # initialize optimizer and scheduler
        optimizer, scheduler = _get_optimizer_and_scheduler(
            optimizer,
            patch_params.parameters(),
            learning_rate,
            decay=lr_decay,
            steps=int(num_steps / accumulate),
        )

        # construct the iterator separately so we have an option to
        # disable progress bar
        loop = range(num_steps)
        if progressbar:
            loop = tqdm(loop)

        for i in loop:
            lossdict, loss = _update_patch_gradients(
                self,
                batch_size,
                kwargs,
                accumulate=accumulate,
                rho=rho,
                clamp_to=clamp_to,
            )

            # save metrics to tensorboard
            record = {k: torch.mean(lossdict[k]) for k in lossdict}
            self._log_scalars(mlflow_metric=False, **record)

            # if this is an update step- update patch, clamp to unit interval
            if (i + 1) % accumulate == 0:
                optimizer.step()
                if lr_decay == "plateau":
                    scheduler.step(loss)
                else:
                    scheduler.step()

                optimizer.zero_grad()

                self._log_scalars(learning_rate=optimizer.param_groups[0]["lr"])
                if clamp_to is not None:
                    with torch.no_grad():
                        if isinstance(self.patch_params, PatchWrapper):
                            self.patch_params.clamp(clamp_to[0], clamp_to[1])
                        else:
                            # distributed case: type should be
                            # torch.nn.parallel.distributed.DistributedDataParallel
                            self.patch_params.module.patch.clamp_(
                                clamp_to[0], clamp_to[1]
                            )

            # if this is an evaluate step- run evaluation and save params
            if ((i + 1) % eval_every == 0) & (eval_every > 0):
                self.evaluate(batch_size, num_eval_steps)
                self.save_patch_params()

            # if we're profiling, update the profiler
            if self._profiling:
                prof.step()
                if self.global_step > self._stop_profiling_on + 1:
                    prof.stop()
                    self._profiling = False
                    self.prof = prof

            self.global_step += 1

        # wrap up mlflow logging
        if self._logging_to_mlflow:
            p = patch_params.patch
            # single patch case
            if isinstance(p, torch.Tensor):
                self._log_image_to_mlflow(p, "patch.png")
            else:
                for k in p:
                    self._log_image_to_mlflow(p[k], f"patch_{k}.png")

        if self._single_patch:
            p = self.patch_params(1)
            if isinstance(p, dict):
                return {k: p[k].squeeze(0) for k in p}
            else:
                return p.squeeze(0)
        else:
            return self.patch_params()

    def __del__(self):
        if self._logging_to_mlflow:
            try:
                mlflow.end_run()
            except:
                pass

    def __getitem__(self, i):
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def save_patch_params(self, path=None):
        """
        Use torch.save to record self.patch_params to file
        """
        if self.rank == 0:
            if path is None:
                path = os.path.join(self.logdir, "patch_params.pt")
            # SINGLE GPU CASE
            if isinstance(self.patch_params, PatchWrapper):
                torch.save(self.patch_params.patch, path)
            # MULTI GPU CASE
            else:
                torch.save(self.patch_params.module.patch, path)
            if self._logging_to_mlflow:
                mlflow.log_artifact(path)

    def optimize(
        self,
        objective,
        logdir,
        patch_shape,
        N,
        num_steps,
        batch_size,
        num_eval_steps=10,
        mlflow_uri=None,
        experiment_name=None,
        extra_params={},
        minimize=True,
        clamp_to=(0, 1),
        **params,
    ):
        """
        Use a Gaussian Process to optimize hyperparameters for self.train().

        :objective: string; name of objective to use for black-box optimization.
            Should be one of the keys from your loss dictionary.
        :logdir: string; top-level directory trials will be saved under
        :patch_shape: tuple or dict of tuples; shape of patch parameter(s) to initialize.
            should look like what you pass to initialize_patch_params().
        :N: int; number of trials to run.
        :num_steps: int; budget in number of steps per trial
        :batch_size: int; batch size for training and eval
        :num_eval_steps: int; number of evaluation batches to run after each trial
        :mlflow_uri: string; location of MLFlow server
        :experiment_name: string; MLFlow experiment
        :extra_params: dict; exogenous parameters to log to MLFlow
        :minimize: bool; whether we're minimizing or maximizing the objective.
        :lr_decay: string what learning rate decay schedule type to use. "none",
            "cosine", "exponential", or "plateau"
        :optimizer: string; 'bim', 'adam', 'sgd', or 'mifgsm'
        :clamp_to: tuple or None; limits to clamp patch params to
        :params: dictionary of parameters you'd pass to train; including learning_rate,
            accumulate, and loss function weights. Specify each value in one of four
            ways:
                -scalar value: the optimizer will leave this value fixed
                -tuple (low, high): optimizer will vary this value on a linear scale
                -tuple (low, high, "log"): optimizer will vary this value on
                    a log scale
                -tuple (low, high, "int"): optimizer will vary this value but
                    only choose integers
                -for categorical options- a string or list of strings

            for example,
            params = {
                    "learning_rate":(1e-5,1e-1,"log"),
                    "accumulate":(1,5,"int"),
                    "lossdict_thingy_1":1.0,
                    "lossdict_thingy_2":(0,1),
                    "optimizer":["bim", "mifgsm"],
                    "lr_decay":["cosine", "exponential", "none"]
                }

        """
        ob = objective + "_delta"
        self.client = _create_ax_client(ob, minimize=minimize, **params)

        def _evaluate_trial(p):
            self.train_patch(
                batch_size,
                num_steps,
                eval_every=-1,
                progressbar=False,
                clamp_to=clamp_to,
                **p,
            )
            self.evaluate(batch_size, num_eval_steps)
            result_mean = np.mean(self.results[ob])
            sem = _bootstrap_std(self.results[ob])
            return {ob: (result_mean, sem)}

        # for each trial
        for i in tqdm(range(N)):
            self.global_step = 0
            # point the logger to a new subdirectory
            ld = os.path.join(logdir, str(i))
            self.set_logging(
                logdir=ld,
                mlflow_uri=mlflow_uri,
                experiment_name=experiment_name,
                extra_params=extra_params,
            )
            # create a new patch
            self.initialize_patch_params(patch_shape=patch_shape)
            # run the trial
            parameters, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index, raw_data=_evaluate_trial(parameters)
            )
            j = self.client.to_json_snapshot()
            json.dump(j, open(os.path.join(logdir, "log.json"), "w"))
            if self._logging_to_mlflow:
                mlflow.end_run()

        return False

    def distributed_train_patch(self, devices, batch_size, num_steps, **kwargs):
        """
        EXPERIMENTAL!!! NOT FULLY TESTED YET. Ye be warned.

        Parallelize training over several devices. Only the first device will
        be used for logging and evaluation steps.

        You may need to add import statements within your loss function for
        libraries like torch.

        :devices: list of torch device objects; the devices to be parallelized over
        :batch_size: int; batch size per device
        :num_steps: int; number of training steps
        :**kwargs: training keyword arguments to be passed to self.train_patch()
        """
        world_size = len(devices)
        if hasattr(self, "writer"):
            delattr(self, "writer")

        queue = mp.Queue()
        # for pytorch to retrieve a Tensor from a Queue, the subprocess that
        # added the Tensor to the Queue needs to still be alive.
        evt = mp.Event()

        pipestring = dill.dumps(self)

        ctx = mp.spawn(
            _run_worker_training_loop,
            args=(
                world_size,
                devices,
                pipestring,
                queue,
                evt,
                batch_size,
                num_steps,
                kwargs,
            ),
            nprocs=world_size,
            join=False,
        )

        for j in range(world_size):
            patch = queue.get(block=True)

        # trigger the event so the workers can end their processes
        evt.set()
        queue.close()
        queue.join_thread()
        return patch
