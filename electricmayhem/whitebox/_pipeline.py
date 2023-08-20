import numpy as np
import torch
import torch.utils.tensorboard
import yaml
import mlflow


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
        mlflow.log_params(self.params)
        
    def forward(self, x, control=False, **kwargs):
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
        
    def forward(self, x, control=False, **kwargs):
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
        super(Pipeline, self).__init__(**kwargs)
        self.steps = []
        for a in args:
            _ = self.__add__(a)
        self.params = {}
        self.global_step = 0
        self.logging_to_mlflow = False
            
    def forward(self, x, control=False, **kwargs):
        for a in self.steps:
            x = a(x, control=control)
        return x
    
    def __add__(self, y):
        # check to see if it's an electricmayhem object. if not assume it's
        # a pytorch model
        if not issubclass(type(y), PipelineBase):
            print(y, type(y))
            y = ModelWrapper(y)
        self.steps.append(y)
        return self
    
    def to_yaml(self):
        params = {s.name:s.params for s in self.steps}
        params["Pipeline"] = self.params
        return yaml.dump(params, default_flow_style=False)
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for s in self.steps:
            sampdict = s.get_last_sample_as_dict()
            for k in sampdict:
                outdict[f"{s.name}_{k}"] = sampdict[k]
                
        return outdict
    
    def set_logging(self, logdir=None, mlflow_uri=None, experiment_name=None):
        """
        Configure TensorBoard and MLFlow for logging results
        
        :logdir: string; path to directory for saving tensorboard logs
        :mlflow_uri: string; URI of MLFlow server
        :experiment_name: string; name to use for MLflow experiment
        """
        if logdir is not None:
            self.logdir = logdir
            self.writer = torch.utils.tensorboard.SummaryWriter(logdir)

        if (mlflow_uri is not None)&(experiment_name is not None):
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            self._logging_to_mlflow = True
        elif (mlflow_uri is not None)&(experiment_name is not None):
            logging.warning("both a server URI and experiment name are required for MLFlow")
            
    def initialize_patch(self, patch_shape=None, patch=None):
        """
        Generate an untrained patch uniformly on the unit interval.
        
        Saves to self.patch
        """
        if (patch_shape is not None)&(patch is None):
            patch = torch.zeros(patch_shape, dtype=torch.float32).uniform_(0,1)
            
        if len(patch.shape) != 3:
            logging.error("initialize_patch() expects a patch shape of length 3; (C,H,W)")
        elif patch.shape[0] not in [1,3]:
            logging.error("initialize_patch() only works with 1 or 3 channels")
            
        self.patch = patch
        
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
        
    
        
    