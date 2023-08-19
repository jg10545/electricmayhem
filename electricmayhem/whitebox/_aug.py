import numpy as np
import torch
import kornia
import logging

from ._pipeline import PipelineBase



        
class KorniaAugmentationPipeline(PipelineBase):
    """
    Wrapper to manage augmentations from the kornia API.
    
    Use check_reproducibility() to make sure the augmentations chosen
    are repeatable.
    """
    name = "KorniaAugmentationPipeline"
    
    def __init__(self, augmentations, ordering=None):
        """
        :augmentations: dict mapping augmentation names (as they appear in the 
            kornia API) to dictionaries of keyword arguments for that augmentation
        :ordering: list of augmentation names, specifying the order in which they
            should be applied.
        """
        super(KorniaAugmentationPipeline, self).__init__()
        # initialize the kornia augmentations
        augs = []
        if ordering is None:
            ordering = list(augmentations.keys())
        for o in ordering:
            evalstring = f"kornia.augmentation.{o}(**{augmentations[o]})"
            augs.append(eval(evalstring))
        
        self.aug = kornia.augmentation.container.AugmentationSequential(*augs)
        # and record parameters
        self.params = augmentations
        self.params["ordering"] = ordering
        
        
    def forward(self, x, control=False, params=None):
        """
        apply augmentations to image
        
        :x: torch.Tensor batch of images in channelfirst format
        :control: boolean; if True use augmentation values from previous batch
        """
        if control:
            params = self.lastsample
        if params is None:
            y = self.aug(x)
        else:
            y = self.aug(x, params=params)
        self.lastsample = self.aug._params
        return y
    
    def check_reproducibility(self, x=None, N=100, epsilon=1e-6):
        """
        I've found at least one case where following the kornia instructions for reproducing
        an augmentation didn't work perfectly. This function does a quick check to make sure
        the same batch.
        
        So far RandomPlasmaShadow seems to have trouble reproducing.
        
        :x: image tensor batch in channel-first format to test on
        :N: int; number of random checks to run
        :epsilon: threshold for average difference between augmentations
        """
        if x is None:
            x = torch.tensor(np.random.uniform(0, 1, size=(3,128,128)).astype(np.float32))
        failures = 0
        for _ in range(100):
            y1 = self(x)#self.apply(x)
            y2 = self(x, control=True)#self.apply(x, control=True)
            if ((y1-y2)**2).numpy().mean() > epsilon:
                failures += 1
        if failures > 0:
            logging.warning(f"reproducibility check failed {failures} out of {N} times")
        return failures
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        outdict = {}
        for p in self.lastsample:
            for k in p.data:
                key = f"{p.name}_{k}"
                # some info not necessary to record
                if k in ["forward_input_shape"]:
                    continue
                # flatten multidimensional params
                elif len(p.data[k].shape) > 1:
                    for i in range(p.data[k].shape[1]):
                        outdict[f"{p.name}_{k}_{i}"] = [float(x[i]) for x in p.data[k]]
                else:
                    outdict[key] = [float(x) for x in p.data[k].cpu().detach().numpy()]
        return outdict