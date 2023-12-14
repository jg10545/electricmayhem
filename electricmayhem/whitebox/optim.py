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