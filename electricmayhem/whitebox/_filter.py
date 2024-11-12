import torch
from ._pipeline import PipelineBase


def highpass(x, limit_x, limit_y):
    """
    Simple 2D high-pass filter in pytorch
    """
    freq = torch.fft.fft2(x, s=(x.shape[2], x.shape[3]), dim=(-2, -1))
    freq_filtered = freq[:, :, :limit_y, :limit_x]
    x_filtered = torch.fft.ifft2(
        freq_filtered, s=(x.shape[2], x.shape[3]), dim=(-2, -1)
    )
    return x_filtered.real


class HighPassFilter(PipelineBase):
    """
    Apply a 2D high pass filter DURING TRAINING STEPS ONLY
    """

    name = "HighPassFilter"

    def __init__(self, limit=None, limit_x=None, limit_y=None, keys=None):
        """
        :limit: int; frequency limit in pixels
        :limit_x: int; alternate to limit- specify frequency cutoff in x-axis
        :limit_y: int; alternate to limit- specify frequency cutoff in y axis
        :keys: if using multiple patches and only applying the filter to some, pass a list of
            strings here to specify which ones
        """
        super().__init__()
        if limit is not None:
            limit_x = limit
            limit_y = limit
        self.keys = keys

        assert (limit_x is not None) & (
            limit_y is not None
        ), "need to specify frequency cutoff"

        self.params = {"limit_x": limit_x, "limit_y": limit_y}
        if keys is not None:
            self.params["keys"] = keys

    def forward(self, x, control=False, evaluate=False, params={}, **kwargs):
        """
        Run image through highpass filter; for evaluation steps do nothing
        """
        # multi-patch case
        if isinstance(x, dict):
            return self._apply_forward_to_dict(
                x, control=control, evaluate=evaluate, params=params, **kwargs
            )
        if evaluate:
            return x, kwargs
        else:
            return highpass(x, self.params["limit_x"], self.params["limit_y"]), kwargs

    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}

    def get_description(self):
        return f"**{self.name}:** cutoffs at x={self.params['limit_x']}, y={self.params['limit_y']}"
