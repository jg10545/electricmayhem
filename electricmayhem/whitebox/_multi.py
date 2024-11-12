import torch
import numpy as np
import torch.distributed
import torch.nn.parallel
import torch.multiprocessing
import torch.utils.tensorboard
import os
import copy
import dill
import logging


class PatchWrapper(torch.nn.Module):
    """
    This object wraps a pytorch tensor containing patch parameters, to
    make it easier to synchronize across GPUs using torch's distributed
    data parallel tools.
    """

    def __init__(self, patch, single_patch=True):
        """
        :patch: pytorch tensor representing the patch or parameters to generate the patch, or a dict of tensors
        :single_patch: bool; whether "patch" represents a single patch (that must be batched
            for training) or a batch of patches
        """
        super().__init__()
        if isinstance(patch, dict):
            self.patch = torch.nn.ParameterDict(patch)
        else:
            self.patch = torch.nn.Parameter(patch)
        self.single_patch = single_patch

    def forward(self, N=None):
        if self.single_patch:
            if isinstance(self.patch, torch.nn.ParameterDict):
                return {
                    x: torch.stack([self.patch[x] for _ in range(N)], 0)
                    for x in self.patch
                }
            else:
                return torch.stack([self.patch for _ in range(N)], 0)
        else:
            return self.patch

    def clamp(self, low=0, high=1):
        """
        clamp patch parameters to some interval
        """
        with torch.no_grad():
            if isinstance(self.patch, torch.nn.ParameterDict):
                for k in self.patch:
                    self.patch[k].clamp_(low, high)
            else:
                self.patch.clamp_(low, high)


def _run_worker_training_loop(
    rank,
    world_size,
    devices,
    pipestring,
    queue,
    evt,
    batch_size,
    num_steps,
    kwargs,
    master_addr="localhost",
    master_port="12355",
):
    """
    Function to send to a subprocess for parallel training.

    :rank: int; rank of this process. Needs to be the first argument to work
        with mp.spawn().
    :world_size: int; number of workers
    :devices: list of devices each worker should map to
    :pipestring: Pipeline object serialized to a string using dill
    :queue: mp.Queue object; queues are the preferred way to pass objects
        between processes in pytorch, so we'll use this to get the patch out
        after training
    :evt: mp.Event object. We don't be able to retrieve the patch from the queue
        unless the originating process is stll active; we'll use this event to
        keep the process from exiting before we're ready.
    :batch_size: int; batch size PER REPLICA for training
    :num_steps: int; number of training steps
    :kwargs: dictionary of keyword arguments for pipeline.train_patch()
    """
    pipeline = dill.loads(pipestring)
    # checks
    assert hasattr(
        pipeline, "patch_params"
    ), "need to call initialize_patch_params() first"
    assert hasattr(pipeline, "loss"), "need to call set_loss() first"
    # recreate the SummaryWriter but only on one worker
    if hasattr(pipeline, "logdir") & (rank == 0):
        pipeline.writer = torch.utils.tensorboard.SummaryWriter(pipeline.logdir)

    # setup
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    # move pipeline to the corresponding device
    pipeline.to(devices[rank])

    # set pipeline.rank. this will make sure only one worker is writing to
    # tensorboard and mlflow
    pipeline.rank = rank

    # pipeline.initialize_patch_params() has already been called- we need to convert
    # the PatchWrapper object to a DDP object
    device = devices[rank]
    # DDP API is different if device is CPU
    if (device == "cpu") | (device == torch.device("cpu")):
        device_ids = None
    else:
        device_ids = [device]

    pipeline.patch_params = torch.nn.parallel.DistributedDataParallel(
        pipeline.patch_params, device_ids=device_ids
    )

    # let's try adding a clamp() method to the DDP object
    patch = pipeline.train_patch(batch_size, num_steps, progressbar=False, **kwargs)

    # move patch back to CPU from wherever it is
    patch.cpu()
    torch.distributed.destroy_process_group()
    # queues are the preferred way to send data between processes
    queue.put(patch.detach())
    # gotta keep the main process alive until all workers are done
    evt.wait()
