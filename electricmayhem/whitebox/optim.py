import torch
import logging

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
    def __init__(self, params, lr=1e-3, mu=0.9):
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
        
                
class NIFGSM(torch.optim.Optimizer):
    """
    Nesterov-Accelerated Momentum Iterative Fast Gradient Sign Method optimizer.
    
    From "NESTEROV ACCELERATED GRADIENT AND SCALE INVARIANCE FOR ADVERSARIAL ATTACKS" by Lin et al (2020)
    https://arxiv.org/pdf/1908.06281.pdf

    Like the MIFGSM object this does gradient descent instead of the ascent described in the paper.
    The other major change between this implementation and the paper is that the tensors NIFGSM optimizes are
    the NESTEROV variables, not the ADVERSARIAL variables. If everything works right these two should converge so
    that shouldn't be a problem. But it allows us to compute the gradient at momentum-shifted locations while 
    maintaining consistency with the torch.optim.Optimizer API.

    This means that we're doing the NAG steps out of order- each step() call here is basically doing equations (7)
    and (8) from one step, then (6) for the following step.
    """
    def __init__(self, params, lr=1e-3, mu=0.9):
        if lr <= 0.:
            raise ValueError(f"invalid learning rate {lr}")
        defaults={"lr":lr, "mu":mu}
        super(NIFGSM, self).__init__(params, defaults)

    def _init_group(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("MIFGSM does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["x_adv"] = p.clone().detach().to(p.device)
            

    def step(self):
        """
        Performs a single optimization step
        """
        # first compute the L1 norm of the gradients. used in equation
        # (7) in the NI-FGSM paper
        gradnorm = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gradnorm += torch.norm(p.grad.data, p=1)
        # now update the momentum buffers and gradients
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            self._init_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # Momentum update: equation (7)
                state["momentum_buffer"].data *= mu
                state["momentum_buffer"].data += grad/gradnorm

                # x_adv update: equation (8), with the sign changed for minimization
                state["x_adv"].data -= lr*state["momentum_buffer"].data.sign()

                # x_nes update: equation (6) FOR THE NEXT STEP, with the sign changed for minimization
                p.data.copy_(state["x_adv"].data - lr*mu*state["momentum_buffer"])
        
                
_OPTIMIZER_DICT = {
    "adam":torch.optim.Adam,
    "sgd":torch.optim.SGD,
    "bim":BIM,
    "mifgsm":MIFGSM,
    "nifgsm":NIFGSM
}


def _get_optimizer_and_scheduler(optimizer, params, lr, decay="none", steps=None):
    """
    Macro to initialize an optimizer and scheduler.

    :optimizer: string; which optimizer to use. adam, sgd, bim, or mifgsm.
    :params: iterable of params
    :lr: float; initial learning rate
    :decay: string; learning rate decay type. none, cosine, exponential, or plateau
    :steps: int; number of training steps 

    Returns
    :opt: pytorch optimizer object
    :scheduler: pytorch scheduler object
    """
    opt = _OPTIMIZER_DICT[optimizer](params, lr)
    if opt == "nifgsm":
        logging.warning("THIS NIFGSM IMPLEMENTATION IS AN EARLY PROTOTYPE YOU PROBABLY SHOULDNT ACTUALLY USE FOR ANYTHING")

    if decay == "none":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1)
    elif decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    elif decay == "exponential":
        # compute gamma such that the LR decays by 3 orders of magnitude
        gamma = (1e-3)**(1./max(steps,1)) # max() to prevent error when steps=0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    elif decay == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                               factor=0.1,
                                                               patience=100)
    return opt, scheduler