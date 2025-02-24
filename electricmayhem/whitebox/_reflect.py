import numpy as np
import torch
import kornia.augmentation
from ._create import PatchSaver

def _unpack_params(params, key=None):
    """
    Turn kornia.augmentation.RandomAffine parameters for PatchReflector
    into something JSON-serializable
    """
    pdict = {}
    if key is None:
        key = "patch"
    for k in ["translations", "center"]:
        if k in params:
            p = params[k].detach().cpu().numpy()
            for j in range(2):
                pdict[f"{key}_{k}_{j}"] = [float(x) for x in p[:,j]]
    return pdict

def _pack_params(pdict, device, key="patch"):
    """
    pack reflection parameters back up in kornia format
    """
    params = {}
    for k in ["translations", "center"]:
        p = [pdict[f"{key}_{k}_0"], pdict[f"{key}_{k}_1"]]
        params[k] = torch.tensor(p).type(torch.float32).reshape(-1,2).to(device)
        N = params[k].shape[0]

    params["scale"] = torch.tensor([[1.,1.] for _ in range(N)]).type(torch.float32).to(device)
    params["angle"] = torch.tensor([0. for _ in range(N)]).type(torch.float32).to(device)
    params["shear_x"] = torch.tensor([0. for _ in range(N)]).type(torch.float32).to(device)
    params["shear_y"] = torch.tensor([0. for _ in range(N)]).type(torch.float32).to(device)
    
    return params


class PatchReflector(PatchSaver):
    """
    Class for translating a patch with toroidal boundary conditions.

    Returns the unchanged patch during evaluation steps.
    """
    def __init__(self, keys=None):
        """
        :keys: None or list of strings; which patches to apply rotation to. If None apply to all patches.
        """
        super().__init__()
        self.params = {}
        if keys is not None:
            self.params["keys"] = keys
        self.lastsample = {}
        self.aug = kornia.augmentation.RandomAffine(degrees=0, translate=(1.,1.), padding_mode="reflection", p=1.)

    def _forward_single(self, x, control=False, evaluate=False, params={}, key=None, **kwargs):
        if key is None:
            key = "patch"
        if evaluate:
            return x, kwargs
        else:
            if control:
                params = _unpack_params(self.lastsample[key], key)
            if params is None:
                y = self.aug(x)
            elif len(params) == 0:
                y = self.aug(x)
            else:
                params = _pack_params(params, x.device, key=key)
                y = self.aug(x, params=params)
            self.lastsample[key] = self.aug._params
            return y, kwargs

    def get_last_sample_as_dict(self):
        outdict = {}
        # get parameters for each patch and add to a single flat dict
        for key in self.lastsample:
            patchdict = _unpack_params(self.lastsample[key], key)
            for p in patchdict:
                outdict[p] = patchdict[p]
        return outdict