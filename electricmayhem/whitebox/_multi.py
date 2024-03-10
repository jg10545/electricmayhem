import torch

class PatchWrapper(torch.nn.Module):
    """
    This object wraps a pytorch tensor containing patch parameters, to
    make it easier to synchronize across GPUs using torch's distributed
    data parallel tools.
    """
    def __init__(self, patch, single_patch=True):
        """
        :patch: pytorch tensor representing the patch or parameters to generate the patch
        :single_patch: bool; whether "patch" represents a single patch (that must be batched
            for training) or a batch of patches
        """
        super().__init__()
        self.patch = torch.nn.Parameter(patch)
        self.single_patch = single_patch
        
    def forward(self, N=None):
        if self.single_patch:
            return torch.stack([self.patch for _ in range(N)],0)
        else:
            return self.patch
        
    def clamp(self, low=0, high=1):
        """
        clamp patch parameters to some interval
        """
        with torch.no_grad():
            self.patch.clamp_(low, high)