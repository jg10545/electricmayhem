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
    y = ssa(x)
    assert x.shape == y.shape

def test_spectrumsimulationattack_reproducible():
    ssa = SpectrumSimulationAttack()
    x = torch.tensor(np.random.uniform(0, 1, size=(1,3,17,21)).astype(np.float32))
    y = ssa(x)
    y2 = ssa(x, **ssa.get_last_sample_as_dict())
    assert ((y-y2)**2).numpy().sum() < 1e-4