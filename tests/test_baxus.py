# -*- coding: utf-8 -*-
import numpy as np
import torch

from electricmayhem._baxus import embedding_matrix



def test_embedding_matrix_correct_shape():
    inpt = 7
    outpt = 5
    
    m = embedding_matrix(inpt, outpt)
    assert m.shape == (outpt, inpt)
    assert isinstance(m, torch.Tensor)
