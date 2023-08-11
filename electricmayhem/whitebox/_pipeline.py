import torch
import yaml
import mlflow
import kornia.augmentation


class PipelineBase(torch.nn.Module):
    name = "PipelineBase"
    
    def __init__(self, **kwargs):
        super(PipelineBase, self).__init__()
        self.params = kwargs
        
    def to_yaml(self):
        return yaml.dump(self.params, default_flow_style=False)
    
    def log_params_to_mlflow(self):
        mlflow.log_params(self.params)
        
    def apply(self, x, **kwargs):
        return x
        
    def __call__(self, x, **kwargs):
        # check and see if we're missing a batch dimension
        missingbatch = isinstance(x, torch.Tensor)&(len(x.shape) == 3)
        if missingbatch:
            x = x.unsqueeze(0)
            
        y = self.apply(x, **kwargs)
        if missingbatch:
            y = y.squeeze(0)
            
        return y
            
        
class KorniaAugmentationPipeline(PipelineBase):
    """
    
    """
    name = "KorniaAugmentationPipeline"
    
    def __init__(self, augmentations, ordering=None):
        """
        
        """
        super(KorniaAugmentationPipeline, self).__init__()
        # initialize the kornia augmentations
        augs = []
        if ordering is None:
            ordering = augmentations.keys()
        for o in ordering:
            evalstring = f"kornia.augmentation.{o}(**{augmentations[o]})"
            augs.append(eval(evalstring))
        
        self.aug = kornia.augmentation.container.AugmentationSequential(*augs)
        # and record parameters
        self.params = augmentations
        self.params["ordering"] = ordering
        
        
    def apply(self, x, params=None):
        if params is None:
            y = self.aug(x)
        else:
            y = self.aug(x, params=params)
        self.lastsample = self.aug._params
        return y