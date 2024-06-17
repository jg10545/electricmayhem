import numpy as np
import torch
from ._pipeline import PipelineBase



def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


class SpectrumSimulationAttack(PipelineBase):
    """
    Implements the augmentation method described in "Frequency Domain Model Augmentation 
    for Adversarial Attack" by Long et al (2022) DURING TRAINING STEPS ONLY

    The method basically does a discrete cosine transform on the image, adds noise in the
    frequency domain, then does an inverse DCT to get back to the spatial domain.
    """
    name = "SpectrumSimulationAttack"
    
    def __init__(self, sigma=0.06, rho=0.5, clamp=(0,1)):
        """
        :limit:
        :sigma:
        :rho:
        """
        super().__init__()
        
        self.params = {
            "rho":rho,
            "sigma":sigma,
            "clamp":clamp
        }
        
    def forward(self, x, control=False, evaluate=False, params={}, **kwargs):
        """
        Run image through highpass filter; for evaluation steps do nothing
        """
        if evaluate:
            return x
        else:
            # generate noise
            with torch.no_grad():
                rho = self.params["rho"]
                sigma = self.params["sigma"]
                if "epsilon" in kwargs:
                    epsilon = torch.tensor(kwargs["epsilon"]).to(x.device)
                else:
                    epsilon = (torch.randn(x.shape)*sigma).to(x.device)
                if "mask" in kwargs:
                    mask = torch.tensor(kwargs["mask"]).to(x.device)
                else:
                    mask = (1 + 2*torch.rand_like(x)*rho - rho).to(x.device)
                self.epsilon = epsilon
                self.mask = mask

            # first map to frequency domain
            x_with_noise = x + epsilon
            x_freq = dct_2d(x+epsilon)
            # multiply by mask and map back to spatial domain
            x_new = idct_2d(x_freq*mask)
            if self.params["clamp"] is not None:
                x_new = torch.clamp(x_new, self.params["clamp"][0], self.params["clamp"][1])
            return x_new

    def _tensor_to_list(self, x):
        return x.detach().cpu().numpy().tolist()
    
    def get_last_sample_as_dict(self):
        """
        Return last sample as a JSON-serializable dict
        """
        return {}#{"epsilon":self._tensor_to_list(self.epsilon),
               #"mask":self._tensor_to_list(self.mask)}
    
    def get_description(self):
        return f"**{self.name}:** epsilon={self.params['epsilon']}, rho={self.params['rho']}"

