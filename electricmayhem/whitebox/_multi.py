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
            
            
def _run_worker_training_loop(rank, world_size, devices, pipestring, queue, evt,
                              batch_size, num_steps, kwargs):
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
    
    #print(f"starting _run_worker_training_loop() for worker {rank}")
    #logging.debug(f"starting _run_worker_training_loop() for worker {rank}")
    
    #print(f"worker {rank} event is {type(evt)}")
    #print(f"unpickling pipeline for worker {rank}")
    pipeline = dill.loads(pipestring)
    #print(f"logging to mlflow for worker {rank}: {pipeline._logging_to_mlflow}")
    # checks
    assert hasattr(pipeline, "patch_params"), "need to call initialize_patch_params() first"
    assert hasattr(pipeline, "loss"), "need to call set_loss() first"
    
    #print("UNPICKLING LOSS")
    #pipeline.loss = dill.loads(pipeline.loss)
    
    # recreate the SummaryWriter but only on one worker
    if hasattr(pipeline, "logdir")&(rank == 0):
        #print("adding writer back on worker 0")
        pipeline.writer = torch.utils.tensorboard.SummaryWriter(pipeline.logdir)
        
    
    # setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #print(f"initializing process group for worker {rank}")
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # move pipeline to the corresponding device
    #print(f"copying pipeline for worker {rank}")
    #pipeline = pipeline.copy()
    #print(f"moving pipeline for worker {rank} to device {devices[rank]}")
    pipeline.to(devices[rank])
    
    # set pipeline.rank. this will make sure only one worker is writing to
    # tensorboard and mlflow
    pipeline.rank = rank
    
    # pipeline.initialize_patch_params() has already been called- we need to convert
    # the PatchWrapper object to a DDP object
    device = devices[rank]
    # DDP API is different if device is CPU
    if (device == "cpu")|(device == torch.device("cpu")):
        device_ids = None
    else:
        device_ids = [device]
    #print(f"device_ids for worker {rank}: {device_ids}")
    #print(f"DDP step for worker {rank}")
    #original_params = pipeline.patch_params
    pipeline.patch_params = torch.nn.parallel.DistributedDataParallel(pipeline.patch_params,
                                                     device_ids=device_ids)
    #print(type(pipeline.patch_params))
    # let's try adding a clamp() method to the DDP object
    #pipeline.patch_params.clamp = original_params.clamp
    #print(f"starting actual training loop for worker {rank}")
    patch = pipeline.train_patch(batch_size, num_steps, progressbar=False,
                                 **kwargs)
    # move patch back to CPU from wherever it is
    patch.cpu()
    #print(f"done with training loop for worker {rank}")
    torch.distributed.destroy_process_group()
    #print(f"process group destroyed for worker {rank}")
    #print(f"worker {rank} patch of type {type(patch)} into queue")
    queue.put(patch.detach())
    
    #print(f"worker {rank} waiting for event")
    evt.wait()
    #print(f"worker {rank} closing")
    
    

    
def run_demo(demo_fn, world_size, model):
    torch.multiprocessing.spawn(demo_fn, args=(world_size, model),
                                nprocs=world_size, join=True)