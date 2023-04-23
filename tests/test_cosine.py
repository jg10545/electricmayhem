import numpy as np

from electricmayhem._cosine import _inverse_cosine_transform

def test_inverse_cosine_transform():
    Hprime, Wprime = 13, 15
    z = np.random.normal(0, 1, size=Hprime*Wprime)
    x = _inverse_cosine_transform(z, (Hprime, Wprime))
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (1,Hprime, Wprime)
    
def test_inverse_cosine_transform_with_reshape():
    Hprime, Wprime = 13, 15
    final_shape = (21, 29)
    z = np.random.normal(0, 1, size=Hprime*Wprime)
    x = _inverse_cosine_transform(z, (Hprime, Wprime), final_shape)
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, final_shape[0], final_shape[1])
    