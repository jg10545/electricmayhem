import numpy as np
import torch
from PIL import Image

try:
    # kornia >= 0.7
    from kornia.augmentation.container.params import ParamItem
except:
    # kornia < 0.7
    from kornia.augmentation.container.base import ParamItem

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
    
def test_bootstrap_std():
    measure = torch.zeros((13), dtype=torch.float32).uniform_(0,1)
    std = _util._bootstrap_std(measure, 25)
    assert isinstance(std, float)
    assert std > 0
    
    
def test_from_paramitem():
    par = ParamItem(name='example_paramitem', data={'foo': torch.tensor([  1,2,3,4])})
    parsed = _util.from_paramitem(par)
    assert parsed["name"] == par.name
    assert isinstance(parsed["data"], dict)
    assert len(parsed["data"]["foo"]) == 3
    
def test_from_paramitem_with_list():
    par = [ParamItem(name='example_paramitem1', data={'foo': torch.tensor([  1,2,3,4])}),
           ParamItem(name='example_paramitem2', data={'foo': torch.tensor([  4,3,2,1])})]
    parsed = _util.from_paramitem(par)
    assert isinstance(parsed, list)
    for i in range(len(parsed)):
        assert parsed[i]["name"] == par[i].name
        assert isinstance(parsed[i]["data"], dict)
        assert len(parsed[i]["data"]["foo"]) == 3
        
def test_to_paramitem():
    par = ParamItem(name='example_paramitem', data={'foo': torch.tensor([  1,2,3,4])})
    parsed = _util.from_paramitem(par)
    unparsed = _util.to_paramitem(parsed)
    assert isinstance(unparsed, ParamItem)
    assert unparsed.name == par.name
    assert unparsed.data["foo"].shape == par.data["foo"].shape
    
    
def test_flatten_dict():
    testdict = {
        "foo":{"a":0, "b":1},
        "bar":{
            "x":0,
            "y":1,
            "z":{"x":0, "y":8}
            }
        }
    flattened = _util._flatten_dict(testdict)
    assert isinstance(flattened, dict)
    assert len(flattened) == 6
    print(list(flattened.keys()))
    for x in ["foo_a", "bar_x", "bar_z_y"]:
        assert x in flattened
        
def test_mlflow_description():
    class FauxPipe():
        def __init__(self, description):
            self.d = description
        def get_description(self):
            return self.d
        
    class FauxPipeline():
        def __init__(self):
            self.steps = [FauxPipe(str(i)+"_step") for i in range(10)]
            self._lossdictkeys = ["foo", "bar"]
            
    markdown = _util._mlflow_description(FauxPipeline())
    assert isinstance(markdown, str)
    for x in ["foo", "bar", "_step"]:
        assert x in markdown
    
    
        
    
    