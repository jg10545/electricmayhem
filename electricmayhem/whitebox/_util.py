import numpy as np
from PIL import Image
import torch

try:
    # kornia >= 0.7
    from kornia.augmentation.container.params import ParamItem
except:
    # kornia < 0.7
    from kornia.augmentation.container.base import ParamItem

def _img_to_tensor(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    if len(x.shape) == 2:
        x = np.stack([x,x,x], -1)
    
    x = torch.tensor(x.astype(np.float32)/255)[:,:,:3].permute(2,0,1)
    return x

def _dict_of_tensors_to_dict_of_arrays(d):
    """
    Turn a dictionary of tensors into a dictionary of arrays.
    """
    outdict = {}
    for k in d:
        if isinstance(d[k], torch.Tensor):
            outdict[k] = d[k].cpu().detach().numpy()
        else:
            outdict[k] = d[k]
    return outdict

def _concat_dicts_of_arrays(*d):
    """
    Combine multiple dicts of arrays into one dict
    of concatenated arrays. Assumes all dicts have the
    same keys, they're all numpy arrays, and they all
    have shapes we can concatenate along axis 0.
    """
    outdict = {}
    for k in d[0]:
        outdict[k] = np.concatenate([x[k] for x in d], 0)
    return outdict

def _bootstrap_std(measure, num_samples=100):
    """
    Estimate the standard deviation of a 1D torch.Tensor using bootstrap sampling
    """
    N = measure.shape[0]
    return np.std([torch.mean(measure[np.random.choice(np.arange(N), size=N, replace=True)]).item()
           for _ in range(num_samples)])


def from_paramitem(x):
    """
    Convert a paramitem struct to something that should be JSON-serializable
    """
    if isinstance(x, list):
        return [from_paramitem(y) for y in x]
    else:
        outdict = {"name":x.name}
        if isinstance(x.data, dict):
            outdict["data"] = {}
            for k in x.data:
                shape = list(x.data[k].shape)
                data = [y for y in x.data[k].detach().cpu().numpy().ravel()]
                dtype = str(x.data[k].dtype)
                outdict["data"][k] = (shape, data, dtype)
        elif isinstance(x.data, list):
            outdict["data"] = []
            for k in x.data:
                shape = list(k.shape)
                data = [y for y in k.detach().cpu().numpy().ravel()]
                dtype = str(k.dtype)
                outdict["data"].append((shape, data, dtype))
        elif x.data is None:
            outdict["data"] = None
    return outdict


def _tensor_from_tuple(shape, data, dtype):
    return torch.tensor(np.array(data).reshape(shape), dtype=eval(dtype))

def to_paramitem(x):
    """
    Turn a list or dict from from_paramitem() back into a ParamItem struct
    """
    if isinstance(x, list):
        return [to_paramitem(y) for y in x]
    elif isinstance(x["data"],dict):
        data = {}
        for k in x["data"]:
            data[k] = _tensor_from_tuple(*x["data"][k])
    elif isinstance(x["data"], list):
        # don't have an example handy to see if this will actually work
        data = []
        for d in x["data"]:
            data.append(_tensor_from_tuple(*d))
    elif x["data"] is None:
        data = None
    return ParamItem(x["name"], data)

