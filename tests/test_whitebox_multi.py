import numpy as np
import torch
from electricmayhem.whitebox import _multi, _create, _pipeline
from tests import modelgenerators

def test_patchwrapper():
    N = 5
    patch = torch.tensor(np.random.uniform(0,1,size=(3,32,32)))
    pw = _multi.PatchWrapper(patch)

    batch = pw(N)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (N, 3, 32, 32)



def test_pipeline_distributed_training_loop_runs():
    """
    This is a pretty minimal test just to see
    if it runs without crashing for a trivial case, splitting
    over two CPU processes.

    I'm running into some issues with this unit test on a mac, which
    appear to be related to some detail of how multiprocessing spawns
    new processes on the mac
    """
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    batch_size = 2
    step_size = 1e-2
    num_steps = 5
    eval_every = 1000
    num_eval_steps = 1
    devices = [torch.device("cpu"), torch.device("cpu")]
    def myloss(outputs, patchparam):
        import torch
        outdict = {}
        outputs = outputs.reshape(outputs.shape[0], -1)
        outdict["mainloss"] = torch.mean(outputs, 1)
        return outdict    
    

    shape = (3,5,7)
    #step = _create.PatchResizer((11,13))
    #pipeline = _pipeline.Pipeline(step)
    pipeline  = _create.PatchResizer((11,13)) + modelgenerators.DummyConvNet().eval()
    pipeline.initialize_patch_params(shape)
    pipeline.set_loss(myloss)
    out = pipeline.distributed_train_patch(devices, batch_size, num_steps, 
                                           learning_rate=step_size, 
                                           eval_every=eval_every,
                                           num_eval_steps=num_eval_steps,
                                           mainloss=1)
    assert out.shape == shape 