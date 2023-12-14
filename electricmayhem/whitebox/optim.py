import torch


class BIM(torch.optim.Optimizer):
    """
    Class for the Basic Iterative Method, or iterative FGSM, from "Adversarial
    examples in the physical world" by Kurakin et al.
    
    Note that this formulates the BIM optimizer for gradient descent as
    opposed to the ascent form in the paper.
    """
    def __init__(self, params, lr=1e-3):
        if lr <= 0.:
            raise ValueError(f"invalid learning rate {lr}")
        defaults={"lr":lr}
        super(BIM, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data -= group["lr"]*grad.sign()
                
                
                
class MIFGSM(torch.optim.Optimizer):
    """
    Momentum Iterative Fast Gradient Sign Method optimizer.
    
    From "Boosting Adversarial Attacks with Momentumn" by Dong et al
    
    Note that this formulates the MI-FGSM optimizer for gradient descent as
    opposed to the ascent form in the paper.
    """
    def __init__(self, params, lr=1e-3, mu=1.):
        if lr <= 0.:
            raise ValueError(f"invalid learning rate {lr}")
        defaults={"lr":lr, "mu":mu}
        super(MIFGSM, self).__init__(params, defaults)

    def _init_group(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("MIFGSM does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            

    def step(self):
        """
        Performs a single optimization step
        """
        # first compute the L1 norm of the gradients. used in line
        # 6 of algorithm 1 in MI-FGSM paper
        gradnorm = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gradnorm += torch.norm(p.grad.data, p=1)
        # now update the momentum buffers and gradients
        for group in self.param_groups:
            self._init_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # update momentum
                state["momentum_buffer"].data *= group["mu"]
                state["momentum_buffer"].data += grad/gradnorm
                # then update weight
                # note that we're doing gradient descent here rather than gradient ascent as
                # described in the paper
                p.data -= group["lr"]*state["momentum_buffer"].data.sign()
                
                
_OPTIMIZER_DICT = {
    "adam":torch.optim.Adam,
    "sgd":torch.optim.SGD,
    "bim":BIM,
    "migfsm":MIFGSM
}


def _get_optimizer_and_scheduler(optimizer, params, lr, decay="none", steps=None):
    """
    Macro to initialize an optimizer and scheduler.

    :optimizer: string; which optimizer to use. adam, sgd, bim, or mifgsm.
    :params: iterable of params
    :lr: float; initial learning rate
    :decay: string; learning rate decay type. none, cosine, or exponential
    :steps: int; number of training steps 

    Returns
    :opt: pytorch optimizer object
    :scheduler: pytorch scheduler object
    """
    opt = _OPTIMIZER_DICT[optimizer](params, lr)

    if decay == "none":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1)
    elif decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    elif decay == "exponential":
        # compute gamma such that the LR decays by 3 orders of magnitude
        gamma = (1e-3)**(1./steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    return opt, scheduler