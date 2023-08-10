import torch
import yaml
import mlflow


class PipelineBase(torch.nn.Module):
    def __init__(self, **kwargs):
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
            