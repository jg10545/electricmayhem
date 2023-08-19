import numpy as np
import torch
from PIL import Image


from electricmayhem.whitebox import _util



def test_img_to_tensor_from_PIL_image():
    H,W,C = 17, 23, 4
    x = Image.fromarray(np.random.randint(0,255, size=(H,W,C)).astype(np.uint8))
    y = _util._img_to_tensor(x)
    
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,H,W)
    
def test_img_to_tensor_from_numpy_array():
    H,W,C = 17, 23, 4
    x = np.random.randint(0,255, size=(H,W,C)).astype(np.uint8)
    y = _util._img_to_tensor(x)
    
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,H,W)
    
    
def test_dict_of_tensors_to_dict_of_arrays():
    foo = np.arange(5)
    bar = np.zeros((3,7,1))
    foobar = True
    testdict = {"foo":torch.tensor(foo), 
                "bar":torch.tensor(bar),
               "foobar":foobar}
    outdict = _util._dict_of_tensors_to_dict_of_arrays(testdict)
    assert isinstance(outdict, dict)
    assert len(outdict) == 3
    assert (outdict["foo"] == foo).all()
    assert (outdict["bar"] == bar).all()
    assert outdict["foobar"] == True
    
def test_concat_dicts_of_arrays():
    test1 = {"a":np.arange(5), "b":np.zeros((1,5,7))}
    test2 = {"a":np.arange(12), "b":np.zeros((3,5,7))}
    outdict = _util._concat_dicts_of_arrays(test1, test2)
    assert isinstance(outdict, dict)
    assert len(outdict) == 2
    assert outdict["a"].shape == (17,)
    assert outdict["b"].shape == (4,5,7)