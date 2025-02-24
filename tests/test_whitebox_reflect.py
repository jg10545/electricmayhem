
import numpy as np
import torch
import json

from electricmayhem.whitebox._reflect import PatchReflector


def test_patch_reflector():
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,32,32)).astype(np.float32))
    reflect = PatchReflector()
    y, _ = reflect(x)
    assert y.shape == x.shape
    # something should have changed
    assert np.mean((x.detach().numpy() - y.detach().numpy())**2) > 0.1
    # test reproducibility
    y2, _ = reflect(x, control=True)
    assert np.mean((y2.detach().numpy() - y.detach().numpy())**2) < 1e-5
    # test json output
    assert isinstance(json.dumps(reflect.get_last_sample_as_dict()), str)

def test_patch_reflector_multiple_patches():
    x = {"foo":torch.tensor(np.random.uniform(0, 1, size=(1,3,32,32)).astype(np.float32)),
         "bar":torch.tensor(np.random.uniform(0, 1, size=(1,3,35,31)).astype(np.float32)),
         "foobar":torch.tensor(np.random.uniform(0, 1, size=(1,3,18,19)).astype(np.float32))}
    
    reflect = PatchReflector(keys=["foo", "bar"])
    y, _ = reflect(x)

    for k in ["foo", "bar", "foobar"]:
        assert x[k].shape == y[k].shape
    
    # something should have changed
    assert np.mean((x["foo"].detach().numpy() - y["foo"].detach().numpy())**2) > 0.1
    assert np.mean((x["bar"].detach().numpy() - y["bar"].detach().numpy())**2) > 0.1
    assert np.mean((x["foobar"].detach().numpy() - y["foobar"].detach().numpy())**2) < 1e-5
    # test reproducibility
    y2, _ = reflect(x, control=True)
    assert np.mean((y2["foo"].detach().numpy() - y["foo"].detach().numpy())**2) < 1e-5
    # test json output
    assert isinstance(json.dumps(reflect.get_last_sample_as_dict()), str)