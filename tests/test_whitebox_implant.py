import numpy as np
import torch

from electricmayhem.whitebox._implant import RectanglePatchImplanter


testtensor = np.random.randint(0, 255, size=(128,128,3))
colorpatch = torch.tensor(np.random.uniform(0, 1, size=(3, 50,50)))
bwpatch = torch.tensor(np.random.uniform(0, 1, size=(1, 50,50)))
box = [10, 10, 100, 100]
bigpatch = torch.tensor(np.random.uniform(0, 1, size=(1, 5000,5000)))

def test_rectanglepatchimplanter_validate_bad_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale=(0.75, 1.25))
    val = imp.validate(bigpatch)
    assert not val
    
def test_rectanglepatchimplanter_validate_good_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale=(0.75, 1.25))
    val = imp.validate(colorpatch)
    assert val
    
def test_rectanglepatchimplanter_sample():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    imp.sample(3)
    assert isinstance(imp.lastsample, dict)
    for k in ["scale", "image", "box", "offset_frac_x", "offset_frac_y"]:
        assert k in imp.lastsample
        assert isinstance(imp.lastsample[k], torch.Tensor)
        assert len(imp.lastsample[k]) == 3
        
def test_rectanglepatchimplanter_apply_color_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp.apply(colorpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_apply_bw_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp.apply(bwpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_call_color_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp(colorpatch)
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_call_bw_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp(bwpatch)
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_call_color_patch_batch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_call_bw_patch_batch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp(torch.stack([bwpatch,bwpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape
    
    
def test_rectanglepatchimplanter_apply_dont_implant():
    # make sure nothing happens if we tell it not to implant
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    unimplanted = imp.apply(bwpatch.unsqueeze(0), dont_implant=True)
    unimplanted = unimplanted.squeeze(0)
    assert isinstance(unimplanted, torch.Tensor)
    assert ((unimplanted.numpy() - imp.images[0].numpy())**2).mean() < 1e-6