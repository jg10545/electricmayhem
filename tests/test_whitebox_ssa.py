import numpy as np
import torch

from electricmayhem.whitebox._ssa import dct_2d, idct_2d, SpectrumSimulationAttack

def test_dct_and_inverse_dct_are_inverses():
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32))
    x_reconstructed = idct_2d(dct_2d(x))

    assert ((x-x_reconstructed)**2).numpy().sum() < 1e-4


def test_spectrumsimulationattack_correct_output_shape():
    ssa = SpectrumSimulationAttack()
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32))
    y, _ = ssa(x)
    assert x.shape == y.shape

def test_spectrumsimulationattack_get_description():
    ssa = SpectrumSimulationAttack()
    assert isinstance(ssa.get_description(), str)



def test_spectrumsimulationattack_multiple_patches_correct_output_shape():
    ssa = SpectrumSimulationAttack()
    x = {"foo":torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32)),
         "bar":torch.tensor(np.random.uniform(0, 1, size=(1,1,19,23)).astype(np.float32))}
    y, _ = ssa(x)

    for k in ["foo", "bar"]:
        assert y[k].shape == x[k].shape


def test_spectrumsimulationattack_multiple_patches_correct_output_shape_skip_one_patch():
    ssa = SpectrumSimulationAttack(keys=["foo"])
    x = {"foo":torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32)),
         "bar":torch.tensor(np.random.uniform(0, 1, size=(1,1,19,23)).astype(np.float32))}
    y, _ = ssa(x)

    for k in ["foo", "bar"]:
        assert y[k].shape == x[k].shape
    assert np.max(np.abs(y["bar"].numpy() - x["bar"].numpy())) < 1e-5
    
"""
def test_spectrumsimulationattack_reproducible():
    ssa = SpectrumSimulationAttack()
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32))
    y = ssa(x)
    y2 = ssa(x, **ssa.get_last_sample_as_dict())
    assert ((y-y2)**2).numpy().sum() < 1e-4
    """