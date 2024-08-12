import numpy as np
import torch
import matplotlib
import json

from electricmayhem.whitebox._implant import RectanglePatchImplanter
from electricmayhem.whitebox._implant import FixedRatioRectanglePatchImplanter
from electricmayhem.whitebox._implant import ScaleToBoxRectanglePatchImplanter


testtensor = np.random.randint(0, 255, size=(128,128,3))
testtensor2 = np.random.randint(0, 255, size=(128,128,3))
colorpatch = torch.tensor(np.random.uniform(0, 1, size=(3, 50,50)))
bwpatch = torch.tensor(np.random.uniform(0, 1, size=(1, 50,50)))
mask = torch.tensor(np.random.choice([0, 1], size=(25,25)))
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
    
def test_rectanglepatchimplanter_train_and_eval_images():
    imp = RectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]}, 
                                  scale=(0.75, 1.25))
    val = imp.validate(colorpatch)
    assert val
    # run a training image through
    implanted, _ = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    
def test_rectanglepatchimplanter_eval_mode_without_separate_eval_set():
    imp = RectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  scale=(0.75, 1.25))
    val = imp.validate(colorpatch)
    assert val
    # run a training image through
    implanted, _ = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    # run an eval image through
    implanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()

def test_rectanglepatchimplanter_sample():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    imp.sample(3)
    assert isinstance(imp.lastsample, dict)
    for k in ["scale", "image", "box", "offset_frac_x", "offset_frac_y"]:
        assert k in imp.lastsample
        assert isinstance(imp.lastsample[k], torch.Tensor)
        assert len(imp.lastsample[k]) == 3
        
def test_rectanglepatchimplanter_sample_with_fixed_offset():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
                                  offset_frac_x=0.5, offset_frac_y=0.25)
    imp.sample(3)
    
    for i in range(3):
        assert imp.lastsample["offset_frac_x"][i] == 0.5
        assert imp.lastsample["offset_frac_y"][i] == 0.25
    
        
def test_rectanglepatchimplanter_apply_color_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted, _ = imp(colorpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_apply_color_patch_with_brightness_scaling():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale_brightness=True)
    implanted, _ = imp(colorpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_get_min_dimensions():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    mindims = imp.get_min_dimensions()
    assert isinstance(mindims, dict)
    assert mindims["minheight"] == 90
    assert mindims["minwidth"] == 90
    
def test_rectanglepatchimplanter_apply_bw_patch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted, _ = imp(bwpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape
    

def test_rectanglepatchimplanter_apply_bw_patch_no_scaling():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
                                  scale=(1.,1.))
    implanted, _ = imp(bwpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_call_color_patch_batch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape
    
def test_rectanglepatchimplanter_call_color_patch_batch_scalar_mask():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=0.5)
    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape


def test_rectanglepatchimplanter_call_color_patch_batch_2D_mask():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=mask)
    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape


def test_rectanglepatchimplanter_call_color_patch_batch_3D_single_channel_mask():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=mask.unsqueeze(0))
    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_call_color_patch_batch_3D_3_channel_mask():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=torch.stack([mask]*3,0))
    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_call_bw_patch_batch():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted, _ = imp(torch.stack([bwpatch,bwpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape
    


def test_rectanglepatchimplanter_plot_boxes():
    # make sure nothing happens if we tell it not to implant
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    fig = imp.plot_boxes()
    assert isinstance(fig, matplotlib.figure.Figure)
    
    

def test_rectanglepatchimplanter_get_last_sample_as_dict():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted = imp(torch.stack([colorpatch,colorpatch], 0))
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    
def test_rectanglepatchimplanter_get_last_sample_as_dict_with_eval_targets():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
                                  eval_imagedict={"ev_im1":testtensor2},
                                  eval_boxdict={"ev_im1":[box]})
    implanted = imp(torch.stack([colorpatch]*10, 0))
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    for s in sampdict["image"]:
        assert s == "im1"
        
def test_rectanglepatchimplanter_get_last_sample_as_dict_evaluate_mode():
    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
                                  eval_imagedict={"ev_im1":testtensor2},
                                  eval_boxdict={"ev_im1":[box]})
    implanted = imp(torch.stack([colorpatch]*10, 0), evaluate=True)
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    for s in sampdict["image"]:
        assert s == "ev_im1"
        
        

def test_fixedratiorectanglepatchimplanter_train_and_eval_images():
    imp = FixedRatioRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]}, 
                                  frac=0.5)
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    

def test_fixedratiorectanglepatchimplanter_train_and_eval_images_with_brightness_scaling():
    imp = FixedRatioRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]}, 
                                  frac=0.5, scale_brightness=True)
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    

def test_fixedratiorectanglepatchimplanter_train_and_eval_images_scale_by_height():
    imp = FixedRatioRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]}, 
                                  frac=0.5, scale_by="height")
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
      
def test_fixedratiorectanglepatchimplanter_train_and_eval_images_scale_by_width():
    imp = FixedRatioRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]}, 
                                  frac=0.5, scale_by="width")
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    

def test_fixedratiorectanglepatchimplanter_sample_with_fixed_offset():
    imp = FixedRatioRectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
                                  offset_frac_x=0.5, offset_frac_y=0.25)
    imp.sample(3)
    
    for i in range(3):
        assert imp.lastsample["offset_frac_x"][i] == 0.5
        assert imp.lastsample["offset_frac_y"][i] == 0.25
    
    
def test_scaletoboxrectanglepatchimplanter_train_and_eval_images():
    imp = ScaleToBoxRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]})
    
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    
def test_scaletoboxrectanglepatchimplanter_train_and_eval_images_with_brightness_scaling():
    imp = ScaleToBoxRectanglePatchImplanter({"im1":testtensor, "im2":testtensor}, 
                                  {"im1":[box], "im2":[box]}, 
                                  eval_imagedict={"im3":testtensor2, "im4":testtensor2},
                                  eval_boxdict={"im3":[box], "im4":[box]},
                                  scale_brightness=True)
    
    # run a training image through
    implanted = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted = imp(colorpatch.unsqueeze(0), evaluate=True)
    # do it again without the patch
    unimplanted = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    