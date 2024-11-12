import numpy as np
import pandas as pd
import torch
import matplotlib
import json

from electricmayhem.whitebox._implant import RectanglePatchImplanter
from electricmayhem.whitebox._implant import FixedRatioRectanglePatchImplanter
from electricmayhem.whitebox._implant import ScaleToBoxRectanglePatchImplanter
from electricmayhem.whitebox._implant import _unpack_rectangle_frame


testtensor = np.random.randint(0, 255, size=(128,128,3))
testtensor2 = np.random.randint(0, 255, size=(128,128,3))
colorpatch = torch.tensor(np.random.uniform(0, 1, size=(3, 50,50)))
bwpatch = torch.tensor(np.random.uniform(0, 1, size=(1, 50,50)))
mask = torch.tensor(np.random.choice([0, 1], size=(25,25)).astype(np.float32))
box = [10, 10, 100, 100]
bigpatch = torch.tensor(np.random.uniform(0, 1, size=(1, 5000,5000)))


def test_unpack_rectangle_frame(test_png_1, test_png_2):
    box1 = [5, 5, 20, 20]
    box2 = [10, 10, 20, 20]
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_2],
                   "xmin":[box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box2[3], box2[3]],
                   "patch":["foo", "bar", "bar"]})

    img_keys, images, boxes = _unpack_rectangle_frame(df)

    assert len(img_keys) == 2
    assert test_png_1 in img_keys
    assert test_png_2 in img_keys
    assert len(images) == 2
    assert isinstance(images[0], torch.Tensor)
    assert images[0].shape == (3,100,100)
    assert len(boxes) == 2
    assert len(boxes[img_keys.index(test_png_1)]) == 2
    assert len(boxes[img_keys.index(test_png_2)]) == 1



def test_rectanglepatchimplanter_validate_bad_patch(test_png_1, test_png_2):
    box1 = [5, 5, 20, 20]
    box2 = [10, 10, 20, 20]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    val = imp.validate({"foo":bigpatch, "bar":bigpatch})
    assert not val
    
def test_rectanglepatchimplanter_validate_good_patch(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale=(0.75, 1.25))
    val = imp.validate({"foo":colorpatch, "bar":colorpatch})
    assert val

  
def test_rectanglepatchimplanter_validate_good_single_patch(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    df = pd.DataFrame({"image":[test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0]],
                   "ymin":[box1[1], box1[1]],
                   "xmax":[box1[2], box1[2]],
                   "ymax":[box1[3], box1[3]],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale=(0.75, 1.25))
    val = imp.validate(colorpatch)
    assert val
    
def test_rectanglepatchimplanter_train_and_eval_images(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   "split":["train", "eval", "train", "eval"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    val = imp.validate({"foo":colorpatch, "bar":colorpatch})
    assert val
    # run a training image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()


def test_rectanglepatchimplanter_train_and_eval_images_single_patch(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    val = imp.validate(colorpatch)
    assert val
    # run a training image through
    implanted, _ = imp(colorpatch.unsqueeze(0))
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted, _ = imp(colorpatch.unsqueeze(0),  evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    
    
def test_rectanglepatchimplanter_eval_mode_without_separate_eval_set(test_png_1):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_1, test_png_1, test_png_1],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    val = imp.validate({"foo":colorpatch, "bar":colorpatch})
    assert val
    # run a training image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    # run an eval image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()

def test_rectanglepatchimplanter_sample(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    imp.sample(3)
    assert isinstance(imp.lastsample, dict)
    for k in ["scale_foo", "scale_bar", "image", "box_foo", "offset_frac_x_foo", "offset_frac_y_foo",
              "box_bar", "offset_frac_x_bar", "offset_frac_y_bar"]:
        assert k in imp.lastsample
        assert isinstance(imp.lastsample[k], torch.Tensor)
        assert len(imp.lastsample[k]) == 3
        
def test_rectanglepatchimplanter_sample_with_fixed_offset(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25),
                                  offset_frac_x=0.5, offset_frac_y=0.25)
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
    #                              )
    imp.sample(3)
    
    for i in range(3):
        assert imp.lastsample["offset_frac_x_patch"][i] == 0.5
        assert imp.lastsample["offset_frac_y_patch"][i] == 0.25
    
        
def test_rectanglepatchimplanter_apply_color_patch(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    implanted, _ = imp(colorpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == (3,100,100)

def test_rectanglepatchimplanter_apply_color_patch_with_brightness_scaling(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), scale_brightness=True)
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, scale_brightness=True)
    implanted, _ = imp(colorpatch.unsqueeze(0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.squeeze(0).shape == (3,100,100)

"""
def test_rectanglepatchimplanter_get_min_dimensions(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
    mindims = imp.get_min_dimensions()
    assert isinstance(mindims, dict)
    assert mindims["minheight"] == 90
    assert mindims["minwidth"] == 90"""
    
#def test_rectanglepatchimplanter_apply_bw_patch():
#    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]})
#    implanted, _ = imp(bwpatch.unsqueeze(0))
#    assert isinstance(implanted, torch.Tensor)
#    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape
    

#def test_rectanglepatchimplanter_apply_bw_patch_no_scaling():
#    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]},
#                                  scale=(1.,1.))
#    implanted, _ = imp(bwpatch.unsqueeze(0))
#    assert isinstance(implanted, torch.Tensor)
#    assert implanted.squeeze(0).shape == torch.tensor(testtensor).permute(2,0,1).shape


def test_rectanglepatchimplanter_call_color_patch_batch_scalar_mask(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), mask=0.5)
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=0.5)
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 1
    assert implanted.shape[1:] == (3,100,100)


def test_rectanglepatchimplanter_call_color_patch_dict_of_scalar_masks(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), mask={"foo":0.5, "bar":0.75})
    #imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=0.5)
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 1
    assert implanted.shape[1:] == (3,100,100)


def test_rectanglepatchimplanter_call_color_patch_batch_2D_mask(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), mask=mask)
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 1
    assert implanted.shape[1:] == (3,100,100)


def test_rectanglepatchimplanter_call_color_patch_batch_dict_of_masks(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), mask={"foo":mask, "bar":mask})
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 1
    assert implanted.shape[1:] == (3,100,100)


#def test_rectanglepatchimplanter_call_color_patch_batch_3D_single_channel_mask():
#    imp = RectanglePatchImplanter({"im1":testtensor}, {"im1":[box]}, mask=mask.unsqueeze(0))
#    implanted, _ = imp(torch.stack([colorpatch,colorpatch], 0))
#    assert isinstance(implanted, torch.Tensor)
#    assert implanted.shape[0] == 2
#    assert implanted.shape[1:] == torch.tensor(testtensor).permute(2,0,1).shape

def test_rectanglepatchimplanter_call_color_patch_batch_3D_3_channel_mask(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25), mask=torch.stack([mask]*3,0))
    implanted, _ = imp({"foo":torch.stack([colorpatch,colorpatch], 0),
                                   "bar":torch.stack([colorpatch,colorpatch], 0)})
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == (3, 100, 100)

def test_rectanglepatchimplanter_call_bw_patch_batch(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df)
    implanted, _ = imp(torch.stack([bwpatch,bwpatch], 0))
    assert isinstance(implanted, torch.Tensor)
    assert implanted.shape[0] == 2
    assert implanted.shape[1:] == (3,100,100)
    


def test_rectanglepatchimplanter_plot_boxes(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df)
    fig = imp.plot_boxes()
    assert isinstance(fig, matplotlib.figure.Figure)
    
    

def test_rectanglepatchimplanter_get_last_sample_as_dict(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   })
    imp = RectanglePatchImplanter(df)
    implanted = imp(torch.stack([colorpatch,colorpatch], 0))
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)


def test_rectanglepatchimplanter_get_last_sample_as_dict_multiple_patches(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "patch":["foo", "foo", "bar", "bar"],
                   })
    imp = RectanglePatchImplanter(df)
    implanted = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}) 
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    
def test_rectanglepatchimplanter_get_last_sample_as_dict_with_eval_targets(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    implanted = imp(torch.stack([colorpatch]*10, 0))
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    for s in sampdict["image"]:
        #assert s == "im1"
        assert s in [test_png_1, test_png_2]
        
def test_rectanglepatchimplanter_get_last_sample_as_dict_evaluate_mode(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"]})
    imp = RectanglePatchImplanter(df, scale=(0.75, 1.25))
    implanted = imp(torch.stack([colorpatch]*10, 0), evaluate=True)
    
    sampdict = imp.get_last_sample_as_dict()
    assert isinstance(sampdict, dict)
    # check to make sure we can turn it into a json
    assert isinstance(json.dumps(sampdict), str)
    for s in sampdict["image"]:
        assert s in [test_png_1, test_png_2]
        
        

def test_fixedratiorectanglepatchimplanter_train_and_eval_images(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"]})
    imp = FixedRatioRectanglePatchImplanter(df, 0.5)
    # run a training image through
    implanted, _ = imp(colorpatch.unsqueeze(0))
    assert implanted.shape == (1, 3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    assert unimplanted.shape == (1,3,100,100)
    # run an eval image through
    implanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True)
    assert implanted.shape == (1,3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(colorpatch.unsqueeze(0), evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()


def test_fixedratiorectanglepatchimplanter_train_and_eval_images_multiple_patches(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = FixedRatioRectanglePatchImplanter(df, 0.5)
    patches = {"foo":colorpatch.unsqueeze(0), "bar":bwpatch.unsqueeze(0)}
    # run a training image through
    implanted, _ = imp(patches)
    assert implanted.shape == (1, 3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    assert unimplanted.shape == (1,3,100,100)
    # run an eval image through
    implanted, _ = imp(patches, evaluate=True)
    assert implanted.shape == (1,3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    

def test_fixedratiorectanglepatchimplanter_train_and_eval_images_with_brightness_scaling(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = FixedRatioRectanglePatchImplanter(df, 0.5, scale_brightness=True)
    patches = {"foo":colorpatch.unsqueeze(0), "bar":bwpatch.unsqueeze(0)}
    # run a training image through
    implanted, _ = imp(patches)
    assert implanted.shape == (1, 3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    assert unimplanted.shape == (1,3,100,100)
    # run an eval image through
    implanted, _ = imp(patches, evaluate=True)
    assert implanted.shape == (1,3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()


def test_fixedratiorectanglepatchimplanter_train_and_eval_images_scale_by_height(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = FixedRatioRectanglePatchImplanter(df, 0.5, scale_by="height")
    patches = {"foo":colorpatch.unsqueeze(0), "bar":bwpatch.unsqueeze(0)}
    # run a training image through
    implanted, _ = imp(patches)
    assert implanted.shape == (1, 3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    assert unimplanted.shape == (1,3,100,100)
    # run an eval image through
    implanted, _ = imp(patches, evaluate=True)
    assert implanted.shape == (1,3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
      
def test_fixedratiorectanglepatchimplanter_train_and_eval_images_scale_by_width(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = FixedRatioRectanglePatchImplanter(df, 0.5, scale_by="width")
    patches = {"foo":colorpatch.unsqueeze(0), "bar":bwpatch.unsqueeze(0)}
    # run a training image through
    implanted, _ = imp(patches)
    assert implanted.shape == (1, 3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    assert unimplanted.shape == (1,3,100,100)
    # run an eval image through
    implanted, _ = imp(patches, evaluate=True)
    assert implanted.shape == (1,3,100,100)
    # do it again without the patch
    unimplanted, _ = imp(patches, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    

def test_fixedratiorectanglepatchimplanter_sample_with_fixed_offset(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = FixedRatioRectanglePatchImplanter(df, 0.5, offset_frac_x=0.5, offset_frac_y=0.25)
    patches = {"foo":colorpatch.unsqueeze(0), "bar":bwpatch.unsqueeze(0)}
    # run a training image through
    implanted, _ = imp(patches)
    imp.sample(3)
    
    for i in range(3):
        for k in ["foo", "bar"]:
            assert imp.lastsample[f"offset_frac_x_{k}"][i] == 0.5
            assert imp.lastsample[f"offset_frac_y_{k}"][i] == 0.25
    
    
def test_scaletoboxrectanglepatchimplanter_train_and_eval_images(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = ScaleToBoxRectanglePatchImplanter(df)
    # run a training image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    
def test_scaletoboxrectanglepatchimplanter_train_and_eval_images_with_brightness_scaling(test_png_1, test_png_2):
    box1 = [5, 5, 80, 80]
    box2 = [10, 10, 90, 90]
    df = pd.DataFrame({"image":[test_png_1, test_png_2, test_png_1, test_png_2],
                   "xmin":[box1[0], box1[0], box2[0], box2[0]],
                   "ymin":[box1[1], box1[1], box2[1], box2[1]],
                   "xmax":[box1[2], box1[2], box2[2], box2[2]],
                   "ymax":[box1[3], box1[3], box2[3], box2[3]],
                   "split":["train", "eval", "train", "eval"],
                   "patch":["foo", "foo", "bar", "bar"]}
                   )
    imp = ScaleToBoxRectanglePatchImplanter(df, scale_brightness=True)
    # run a training image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)})
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, control=True)
    assert (unimplanted.squeeze(0) == imp.images[0]).all()
    assert not (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    # run an eval image through
    implanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True)
    # do it again without the patch
    unimplanted, _ = imp({"foo":colorpatch.unsqueeze(0), "bar":colorpatch.unsqueeze(0)}, evaluate=True, control=True)
    assert not (unimplanted.squeeze(0) == imp.images[0]).all()
    assert (unimplanted.squeeze(0) == imp.eval_images[0]).all()
    