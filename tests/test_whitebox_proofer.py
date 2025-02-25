import numpy as np
import torch
from PIL import ImageCms, Image

from electricmayhem.whitebox import _proofer

def test_tensor_batch_to_imgs():
    t = torch.tensor(np.random.uniform(0, 1, size=(5,3,7,10)))
    imgs = _proofer._tensor_batch_to_imgs(t)
    assert len(imgs) == t.shape[0]
    assert isinstance(imgs[0], Image.Image)

def test_imgs_to_tensor_batch():
    imgs = [Image.fromarray(np.random.randint(0, 255, size=(7,10,3)).astype(np.uint8))
            for _ in range(5)]
    t = _proofer._imgs_to_tensor_batch(imgs)
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == len(imgs)
    assert t.shape[1] == 3
    assert t.shape[2] == 7
    assert t.shape[3] == 10

def test_tensor_batch_to_imgs_imgs_to_tensor_batch_is_identity_operation():
    t = torch.tensor(np.random.uniform(0, 1, size=(5,3,7,10)))
    imgs = _proofer._tensor_batch_to_imgs(t)
    tprime = _proofer._imgs_to_tensor_batch(imgs)
    assert isinstance(tprime, torch.Tensor)
    assert t.shape == tprime.shape
    assert np.max((t.numpy() - tprime.numpy())**2) < 1e-3

def test_softproofer(tmp_path_factory):
    target_profile = ImageCms.createProfile("XYZ")
    proofer = _proofer.SoftProofer(target_profile)
    test_tensor = torch.tensor(np.random.uniform(0, 1, size=(5,3,7,13)))

    output_train, _ = proofer(test_tensor)
    output_eval, _ = proofer(test_tensor, evaluate=True)

    assert isinstance(output_train, torch.Tensor)
    assert test_tensor.shape == output_train.shape
    assert np.sum((test_tensor.numpy() - output_train.numpy())**2) < 1e-5
    assert isinstance(output_eval, torch.Tensor)
    assert test_tensor.shape == output_eval.shape
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    proofer.log_vizualizations(test_tensor, test_tensor, writer, 0)



def test_softproofer_with_multiple_patches(tmp_path_factory):
    target_profile = ImageCms.createProfile("XYZ")
    proofer = _proofer.SoftProofer(target_profile)
    test_tensor = {"foo":torch.tensor(np.random.uniform(0, 1, size=(5,3,7,13))),
                   "bar":torch.tensor(np.random.uniform(0, 1, size=(5,3,11,17)))}

    output_train, _ = proofer(test_tensor)
    output_eval, _ = proofer(test_tensor, evaluate=True)

    assert isinstance(output_train, dict)
    for x in ["foo", "bar"]:
        assert isinstance(output_train[x], torch.Tensor)
        assert test_tensor[x].shape == output_train[x].shape
        assert np.sum((test_tensor[x].numpy() - output_train[x].numpy())**2) < 1e-5
        assert isinstance(output_eval[x], torch.Tensor)
        assert test_tensor[x].shape == output_eval[x].shape
    
    fn = str(tmp_path_factory.mktemp("logs"))
    writer = torch.utils.tensorboard.SummaryWriter(fn)
    proofer.log_vizualizations(test_tensor, test_tensor, writer, 0)