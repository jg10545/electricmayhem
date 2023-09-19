import numpy as np
import torch
import kornia
import logging

from ._pipeline import PipelineBase
from ._util import to_paramitem, from_paramitem



        
class KorniaAugmentationPipeline(PipelineBase):
    """
    Wrapper to manage augmentations from the kornia API.
    
    Use check_reproducibility() to make sure the augmentations chosen
    are repeatable.
    
    DO NOT USE p < 1.0 FOR ANY AUGMENTATIONS
    
    Tracking parameters is more complicated in this case and not currently
    implemented. 
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
            aug = eval(evalstring)
            if "p" in augmentations[o]:
                assert augmentations[o]["p"] == 1, "augmentations need to be applied with p=1.0"
            elif aug.p < 1:
                logging.warning(f"setting p=1 for {o}")
                aug.p = 1.
            augs.append(aug)
        
        self.aug = kornia.augmentation.container.AugmentationSequential(*augs)
        # and record parameters
        self.params = augmentations
        self.params["ordering"] = ordering
        
        
    def forward(self, x, control=False, evaluate=False, params=None, **kwargs):
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
            y = self.aug(x, params=to_paramitem(params))
        self.lastsample = from_paramitem(self.aug._params)
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
            y1 = self(x)
            y2 = self(x, control=True)
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
        # for each augmentation
        for s in self.lastsample:
            # for each parameter sampled for that augmentation
            for k in s['data']:
                # skip ordering, input shape, and batch_prob (which should be True for all if p=1.0)
                if ('order' not in k)&('input_shape' not in k)&('batch_prob' not in k):
                    # some samples are multidimensional- unravel to two dimensions
                    data = np.array(s["data"][k][1])
                    data = data.reshape(s["data"][k][0][0], -1)
                    # record each dimension separately
                    for i in range(data.shape[1]):
                        if data.shape[1] > 1:
                            outdict[f"{s['name']}_{k}_{i}"] = data[:,i]
                        else:
                            outdict[f"{s['name']}_{k}"] = data[:,i]
        return outdict
    
    def get_description(self):
        return f"**{self.name}:** {', '.join(self.params['ordering'])}"