import numpy as np
import torch
import torch.utils.tensorboard
import yaml
import mlflow
import kornia.augmentation


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

        
class KorniaAugmentationPipeline(PipelineBase):
    """
    Wrapper to manage augmentations from the kornia API.
    
    Use check_reproducibility() to make sure the augmentations chosen
    are repeatable.
    """
    name = "KorniaAugmentationPipeline"
    
    def __init__(self, augmentations, ordering=None):
        """
        :augmentations: dict mapping augmentation names (as they appear in the 
            kornia API) to dictionaries of keyword arguments for that augmentation
        :ordering: list of augmentation names, specifying the order in which they
            should be applied.
        """
        super(KorniaAugmentationPipeline, self).__init__()
        # initialize the kornia augmentations
        augs = []
        if ordering is None:
            ordering = list(augmentations.keys())
        for o in ordering:
            evalstring = f"kornia.augmentation.{o}(**{augmentations[o]})"
            augs.append(eval(evalstring))
        
        self.aug = kornia.augmentation.container.AugmentationSequential(*augs)
        # and record parameters
        self.params = augmentations
        self.params["ordering"] = ordering
        
        
    def forward(self, x, control=False, params=None):
        """
        apply augmentations to image
        
        :x: torch.Tensor batch of images in channelfirst format
        :control: boolean; if True use augmentation values from previous batch
        """
        if control:
            params = self.lastsample
        if params is None:
            y = self.aug(x)
        else:
            y = self.aug(x, params=params)
        self.lastsample = self.aug._params
        return y
    
    def check_reproducibility(self, x=None, N=100, epsilon=1e-6):
        """
        I've found at least one case where following the kornia instructions for reproducing
        an augmentation didn't work perfectly. This function does a quick check to make sure
        the same batch.
        
        So far RandomPlasmaShadow seems to have trouble reproducing.
        
        :x: image tensor batch in channel-first format to test on
        :N: int; number of random checks to run
        :epsilon: threshold for average difference between augmentations
        """
        if x is None:
            x = torch.tensor(np.random.uniform(0, 1, size=(3,128,128)).astype(np.float32))
        failures = 0
        for _ in range(100):
            y1 = self(x)#self.apply(x)
            y2 = self(x, control=True)#self.apply(x, control=True)
            if ((y1-y2)**2).numpy().mean() > epsilon:
                failures += 1
        if failures > 0:
            logging.warning(f"reproducibility check failed {failures} out of {N} times")
        return failures
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for p in self.lastsample:
            for k in p.data:
                key = f"{p.name}_{k}"
                # some info not necessary to record
                if k in ["forward_input_shape"]:
                    continue
                # flatten multidimensional params
                elif len(p.data[k].shape) > 1:
                    for i in range(p.data[k].shape[1]):
                        outdict[f"{p.name}_{k}_{i}"] = [float(x[i]) for x in p.data[k]]
                else:
                    outdict[key] = [float(x) for x in p.data[k].cpu().detach().numpy()]
        return outdict
        
    
    
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