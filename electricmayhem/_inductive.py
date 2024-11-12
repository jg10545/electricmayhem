import numpy as np
import torch
import collections
import mlflow
import kornia.geometry

from electricmayhem.blackbox import _augment

from .blackbox._graphite import BlackBoxPatchTrainer, estimate_transform_robustness
from electricmayhem import _mask


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, nin, nout, dilation=1):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            nin, nin, kernel_size=3, groups=nin, dilation=dilation, padding="same"
        )
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseSeparableConvTranspose(torch.nn.Module):
    def __init__(self, nin, nout, dilation=1):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.depthwise = torch.nn.Conv2d(
            nin, nin, kernel_size=3, groups=nin, dilation=dilation, padding="same"
        )
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.depthwise(x)
        out = self.pointwise(x)
        return out


class Block(torch.nn.Module):
    def __init__(self, inChannels, outChannels, dilation=1, separable=True):
        super().__init__()
        # store the convolution and RELU layers
        if separable:
            self.conv1 = DepthwiseSeparableConv(
                inChannels, outChannels, dilation=dilation
            )
            self.conv2 = DepthwiseSeparableConv(
                outChannels, outChannels, dilation=dilation
            )
        else:
            self.conv1 = torch.nn.Conv2d(
                inChannels, outChannels, 3, dilation=dilation, padding="same"
            )
            self.conv2 = torch.nn.Conv2d(
                outChannels, outChannels, 3, dilation=dilation, padding="same"
            )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class MiniUNet(torch.nn.Module):
    """
    Barebones segmentation network in the style of UNet
    """

    def __init__(
        self,
        inchannels=5,
        outchannels=1,
        start_filters=32,
        depth=2,
        dilation=1,
        verbose=False,
        separable=True,
    ):
        """
        :inchannels: int; number of input channels
        :outchannels: int; number of output channels
        :start_filters: int; number of filters in first hidden layer. Filters will double after every pooling layer
        :depth: int; number of spatial scales for network (counting the first)
        :dilation: int; dilation argument passed to all convolutions
        :verbose: bool; for diagnostics
        """
        super().__init__()
        self.depth = depth
        self.verbose = verbose
        # BUILD THE ENCODER
        self.pool = torch.nn.MaxPool2d(2)
        self.encoderblocks = torch.nn.ModuleList()
        for d in range(depth):
            if d == 0:
                block = Block(inchannels, start_filters, dilation, separable=separable)
            else:
                infilt = start_filters * (2 ** (d - 1))
                outfilt = start_filters * (2 ** (d))
                block = Block(infilt, outfilt, dilation)
            self.encoderblocks.append(block)

        # BUILD THE DECODER
        self.deconvolutions = torch.nn.ModuleList()
        self.decoderblocks = torch.nn.ModuleList()
        for d in range(depth - 1):
            # deconvolutions
            if separable:
                self.deconvolutions.append(
                    DepthwiseSeparableConvTranspose(
                        int(start_filters * (2 ** (d))),
                        int(start_filters * (2 ** (d - 1))),
                    )
                )
            else:
                self.deconvolutions.append(
                    torch.nn.ConvTranspose2d(
                        int(start_filters * (2 ** (d))),
                        int(start_filters * (2 ** (d - 1))),
                        2,
                        2,
                        # dilation=dilation
                    )
                )
            # convolution blocks- last one outputs our output channels
            if d == 0:
                block = Block(
                    4 * start_filters,
                    outchannels,
                    dilation=dilation,
                    separable=separable,
                )
            else:
                block = Block(
                    start_filters * (2 ** (d)),
                    start_filters * (2 ** (d - 1)),
                    dilation=dilation,
                    separable=separable,
                )
            self.decoderblocks.append(block)
        self.outputconv = torch.nn.Conv2d(
            start_filters, outchannels, 1, dilation=dilation, padding="same"
        )
        self.outputsigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # ENCODER
        if self.verbose:
            print(x.shape)
        block_outputs = []
        if self.verbose:
            print("encoding")
        for i in range(self.depth - 1):
            x = self.encoderblocks[i](x)
            if self.verbose:
                print("after block", x.shape)
            block_outputs.append(x)
            if i < self.depth - 2:
                x = self.pool(x)
                if self.verbose:
                    print("pool", x.shape)
        if self.verbose:
            print("block_outputs:", len(block_outputs))
        # DECODER
        for i in list(range(1, self.depth - 1))[::-1]:
            if self.verbose:
                print(i)
            x = self.deconvolutions[i](x)
            if self.verbose:
                print("after deconv", x.shape)
            x = torch.cat([x, block_outputs[i - 1]], dim=1)
            if self.verbose:
                print("after cat", x.shape)
            x = self.decoderblocks[i](x)
            if self.verbose:
                print("after decode block", x.shape)

        x = self.outputconv(x)
        x = self.outputsigmoid(x)
        if self.verbose:
            print("after sigmoid", x.shape)
        return x


class StateDict:
    """
    Wrapper for the state dict returned by pytorch models, with
    convenience functions to allow multiplying by scalars
    and adding
    """

    def __init__(self, sd):
        self.sd = sd

    def __getitem__(self, k):
        return self.sd[k]

    def to_dict(self):
        return self.sd

    def __call__(self):
        return self.sd

    def param_count(self):
        paramcount = 0
        for k in self.sd:
            paramcount += np.prod(self.sd[k].data.shape)
        return paramcount

    def __rmul__(self, a):
        newsd = collections.OrderedDict()
        for k in self.sd:
            newsd[k] = a * self.sd[k]
        return StateDict(newsd)

    def __add__(self, a):
        newsd = collections.OrderedDict()
        for k in self.sd:
            newsd[k] = self[k] + a[k]
        return StateDict(newsd)

    def __eq__(self, a):
        if set(self.sd.keys()) != set(a.sd.keys()):
            return False
        else:
            for k in self.sd:
                if not (self.sd[k].numpy() == a.sd[k].numpy()).all():
                    return False
        return True

    def weight_decay(self, a):
        newsd = collections.OrderedDict()
        for k in self.sd:
            if "bias" in k:
                newsd[k] = self.sd[k]
            else:
                newsd[k] = (1 - a) * self.sd[k]
        return StateDict(newsd)

    def random(self):
        """
        Generate a new StateDict with tensors of the same shapes, but
        with values drawn uniformly from the unit hypersphere
        """
        newsd = collections.OrderedDict()
        for k in self.sd:
            u = torch.randn(self.sd.shape).type(torch.FloatTensor)
            newsd[k] = u / torch.norm(u)
        return StateDict(newsd)


def estimate_gradient(
    images,
    init_masks,
    final_masks,
    augs,
    detect_func,
    tr_estimate,
    q=10,
    beta=1,
    include_error_as_positive=False,
    use_scores=False,
):
    """
    Use Randomized Gradient-Free estimation to compute a gradient.

    NOTE should this be normalized by q? it doesn't appear to be in the GRAPHITE codebase.

    :img: torch.Tensor in channel-first format, containing the original image
    :mask: torch.Tensor in channel-first format containing mask
        as 0,1 values
    :pert torch.Tensor in channel-first format containing the
        adversarial perturbation
    :augs: list of augmentation parameters
    :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
    :tr_estimate: float; estimated transform robustness of the current
        image/mask/perturbation
    :q: int; number of random vectors to sample
    :beta: float; smoothing parameter setting size of random shifts to perturbation
    :include_error_as_positive: bool; whether to count -1s from the detect function
        as a positive detection
    :use_scores:

    Returns gradient as a (C,H,W) torch Tensor
    """
    # if the perturbation is a different size- we need a copy of the mask for normalization
    C, H, W = pert.shape
    if (mask.shape[1] != H) | (mask.shape[2] != W):
        mask_resized = kornia.geometry.transform.resize(mask, (H, W))
    else:
        mask_resized = mask
    us = []
    u_trs = []
    for _ in range(q):
        u = torch.randn(pert.shape).type(torch.FloatTensor) * mask_resized[:C, :, :]
        u = u / torch.norm(u)

        u_est = estimate_transform_robustness(
            detect_func,
            augs,
            img,
            mask=mask,
            pert=pert + beta * u,
            include_error_as_positive=include_error_as_positive,
            use_scores=use_scores,
        )
        us.append(u)
        u_trs.append(u_est["tr"])

    gradient = torch.zeros_like(pert)
    for u, tr in zip(us, u_trs):
        gradient += (tr - tr_estimate) * u / (beta * q)

    return gradient


class InductivePatchTrainer(BlackBoxPatchTrainer):
    """ """

    def __init__(
        self,
        model,
        train_images,
        train_initial_masks,
        train_final_masks,
        test_images,
        test_initial_masks,
        test_final_masks,
        detect_func,
        logdir,
        num_augments=100,
        q=10,
        beta=1,
        subset_frac=0,
        aug_params={},
        tr_thresh=0.25,
        reduce_steps=10,
        eval_augments=1000,
        perturbation=None,
        mask_thresh=0.99,
        use_scores=False,
        num_boost_iters=1,
        include_error_as_positive=False,
        extra_params={},
        fixed_augs=None,
        mlflow_uri=None,
        experiment_name=None,
        eval_func=None,
    ):
        """
        :model:
        :train_images:
        :train_initial_masks:
        :train_final_masks:
        :test_images:
        :test_initial_masks:
        :test_final_masks:

        :detect_func: function; inputs an image and returns 1, 0, or -1 depending on whether the black-box algorithm correctly detected, missed, or threw an error
        :logdir: string; location to save tensorboard logs in
        :num_augments: int; number of augmentations to sample for each mask reduction, RGF, and line search step
        :q: int; number of random vectors to use for RGF
        :beta: float; RGF smoothing parameter
        :subset_frac: float;
        :aug_params: dict; any non-default options to pass to
            _augment.generate_aug_params()
        :tr_thresh:  float; transform robustness threshold to aim for
            during mask reudction step
        :reduce_steps: int; number of steps to take during mask reduction
        :eval_augments: int or list of aug params. Augmentations to use at the end of every epoch to evaluate performance
        :perturbation: torch.Tensor in (C,H,W) format. Optional initial perturbation
        :mask_thresh: float; when mask reduction hits this threshold swap
            over to the final_mask
        :use_scores: incorporate scores instead of hard labels (training only)
        :num_boost_iters: int; number of boosts (RGF/line search) steps to
            run per epoch. GRAPHITE used 5.
        :include_error_as_positive: bool; whether to count -1s from the detect function as a positive detection ONLY for boosting, not for mask reduction
        :extra_params: dictionary of other parameters you'd like recorded
        :fixed_augs: fixed augmentation parameters to sample from instead of
            generating new ones each step.
        :reduce_mask: whether to include GRAPHITE's mask reduction step
        :mlflow_uri: string; URI for MLFlow server or directory
        :experiment_name: string; name of MLFlow experiment to log
        :eval_func: function containing any additional evalution metrics. run
            inside self.evaluate()

        """
        self.query_counter = 0

        self.tr = 0
        self.mask_thresh = 0.99
        self.fixed_augs = fixed_augs
        self.eval_func = eval_func

        self.model = model
        self.train_images = train_images
        self.train_initial_masks = train_initial_masks
        self.train_final_masks = train_final_masks
        self.test_images = test_images
        self.test_initial_masks = test_initial_masks
        self.test_final_masks = test_final_masks

        # self.img = img
        # self.initial_mask = initial_mask
        # self.final_mask = final_mask
        # self.priority_mask = electricmayhem.mask.generate_priority_mask(initial_mask, final_mask)
        # self.detect_func = detect_func

        if isinstance(eval_augments, int):
            eval_augments = [
                _augment.generate_aug_params(**aug_params) for _ in range(eval_augments)
            ]
        self.eval_augments = eval_augments
        # if perturbation is None:
        #    perturbation = torch.Tensor(np.random.uniform(0, 1, size=img.shape))
        # self.perturbation = perturbation

        self.logdir = logdir
        self.writer = torch.utils.tensorboard.SummaryWriter(logdir)

        self.aug_params = aug_params
        self.params = {
            "num_augments": num_augments,
            "q": q,
            "beta": beta,
            "tr_thresh": tr_thresh,
            "reduce_steps": reduce_steps,
            "num_boost_iters": num_boost_iters,
            "include_error_as_positive": include_error_as_positive,
            "reduce_mask": reduce_mask,
            "subset_frac": subset_frac,
            "use_scores": use_scores,
        }
        self.extra_params = extra_params
        self._configure_mlflow(mlflow_uri, experiment_name)
        # record hyperparams for all posterity
        yaml.dump(
            {
                "params": self.params,
                "aug_params": self.aug_params,
                "extra_params": self.extra_params,
            },
            open(os.path.join(logdir, "config.yml"), "w"),
        )

    def _configure_mlflow(self, uri, expt):
        # set up connection to server, experiment, and start run
        if (uri is not None) & (expt is not None):
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(expt)
            mlflow.start_run()
            self._logging_to_mlflow = True

            # now log parameters
            mlflow.log_params(self.aug_params)
            mlflow.log_params(self.params)
            mlflow.log_params(self.extra_params)
        else:
            self._logging_to_mlflow = False

    def log_metrics_to_mlflow(self, metdict):
        if self._logging_to_mlflow:
            mlflow.log_metrics(metdict, step=self.query_counter)

    def __del__(self):
        mlflow.end_run()

    def _sample_augmentations(self, num_augments=None):
        """
        Randomly sample a list of augmentation parameters
        """
        if num_augments is None:
            num_augments = self.params["num_augments"]

        # if fixed augmentations were passed, sample from them
        # without replacement
        if self.fixed_augs is not None:
            return list(
                np.random.choice(self.fixed_augs, size=num_augments, replace=False)
            )
        # otherwise generate new ones
        else:
            return [
                _augment.generate_aug_params(**self.aug_params)
                for _ in range(num_augments)
            ]

    def _get_mask(self):
        """
        Return a binary mask using the current value of a
        """
        if self.a >= self.mask_thresh:
            return self.final_mask
        else:
            return (self.priority_mask > self.a).float()

    def _get_img_with_perturbation(self):
        """
        Return the current version of the image + masked perturbation
        glued on. Does not use composition noise
        """
        mask = self._get_mask()
        return _augment.compose(self.img, mask, self.perturbation, 0, 0)

    def _reduce_mask(self):
        """
        GRAPHITE REDUCE_MASK step.
        """
        augments = self._sample_augmentations()
        a, results = reduce_mask(
            self.img,
            self.priority_mask,
            self.perturbation,
            self.detect_func,
            augments,
            n=self.params["reduce_steps"],
            tr_threshold=self.params["tr_thresh"],
            minval=self.a,
            use_scores=self.params["use_scores"],
        )
        self.a = a
        self.tr = results[-1]["tr"]
        # for every mask threshold step record stats in tensorboard
        for r in results:
            self.query_counter += self.params["num_augments"]
            self.writer.add_scalar(
                "reduce_mask_transform_robustness",
                r["tr"],
                global_step=self.query_counter,
            )
            self.writer.add_scalar(
                "reduce_mask_crash_frac",
                r["crash_frac"],
                global_step=self.query_counter,
            )
            self.writer.add_scalar("a", r["a"], global_step=self.query_counter)

    def _estimate_gradient(self):
        """
        Estimate gradient with RGF
        """
        augments = self._sample_augmentations()
        mask = self._get_mask()
        # check to see if we need to take a random subset of the mask
        subset_frac = self.params["subset_frac"]
        if subset_frac > 0:
            mask = electricmayhem.mask.random_subset_mask(mask, subset_frac)
            self._subset_mask = mask
            self.writer.add_image("subset_mask", mask, global_step=self.query_counter)
        self.tr = estimate_transform_robustness(
            self.detect_func,
            augments,
            self.img,
            mask=mask,
            pert=self.perturbation,
            include_error_as_positive=self.params["include_error_as_positive"],
            use_scores=self.params["use_scores"],
        )["tr"]
        self.query_counter += self.params["num_augments"]
        self.writer.add_scalar("tr", self.tr, global_step=self.query_counter)

        gradient = estimate_gradient(
            self.img,
            mask,
            self.perturbation,
            augments,
            self.detect_func,
            self.tr,
            q=self.params["q"],
            beta=self.params["beta"],
            include_error_as_positive=self.params["include_error_as_positive"],
            use_scores=self.params["use_scores"],
        )
        self.query_counter += self.params["num_augments"] * self.params["q"]
        self.writer.add_scalar(
            "gradient_l2_norm",
            np.sqrt(np.sum(gradient.numpy() ** 2)),
            global_step=self.query_counter,
        )
        return gradient

    def _update_perturbation(self, gradient, lrs=None):
        """
        Pick a step size and update the perturbation
        """
        augments = self._sample_augmentations()
        # check to see whether we need to load a random subset mask
        if hasattr(self, "_subset_mask"):
            mask = self._subset_mask
        else:
            mask = self._get_mask()
        self.perturbation, resultdict = update_perturbation(
            self.img,
            mask,
            self.perturbation,
            augments,
            self.detect_func,
            gradient,
            lrs=lrs,
            include_error_as_positive=self.params["include_error_as_positive"],
            use_scores=self.params["use_scores"],
        )
        self.query_counter += 8 * self.params["num_augments"]
        self.writer.add_scalar(
            "learning_rate", resultdict["lr"], global_step=self.query_counter
        )

    def evaluate(self):
        """
        Run a suite of evaluation tests on the test augmentations.
        """
        tr_dict, outcomes, raw = estimate_transform_robustness(
            self.detect_func,
            self.eval_augments,
            self.img,
            self._get_mask(),
            self.perturbation,
            return_outcomes=True,
            include_error_as_positive=self.params["include_error_as_positive"],
        )

        self.writer.add_scalar(
            "eval_transform_robustness", tr_dict["tr"], global_step=self.query_counter
        )
        self.writer.add_scalar(
            "eval_crash_frac", tr_dict["crash_frac"], global_step=self.query_counter
        )
        # only log TR to mlflow if we got rid of the mask, otherwise you
        # could trivially get TR=1
        if self.a >= self.mask_thresh:
            self.log_metrics_to_mlflow({"eval_transform_robustness": tr_dict["tr"]})
            # store results in memory too
            self.tr_dict = tr_dict

        # visual check for correlations in transform robustness across augmentation params
        coldict = {-1: "k", 1: "b", 0: "r"}
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(
            [a["scale"] for a in self.eval_augments],
            [a["gamma"] for a in self.eval_augments],
            s=[2 + 2 * a["blur"] for a in self.eval_augments],
            c=[coldict[o] for o in outcomes],
            alpha=0.5,
        )
        ax.set_xlabel("scale", fontsize=14)
        ax.set_ylabel("gamma", fontsize=14)

        self.writer.add_figure(
            "evaluation_augmentations", fig, global_step=self.query_counter
        )

        if self.eval_func is not None:
            self.eval_func(
                self.writer,
                self.query_counter,
                img=self.img,
                mask=self._get_mask(),
                perturbation=self.perturbation,
                augs=self.eval_augments,
                tr_dict=tr_dict,
                outcomes=outcomes,
                raw=raw,
                include_error_as_positive=self.params["include_error_as_positive"],
            )

    def _log_image(self):
        """
        log image to tensorboard
        """
        self.writer.add_image(
            "img_with_mask_and_perturbation",
            self._get_img_with_perturbation(),
            global_step=self.query_counter,
        )

    def _save_perturbation(self):
        filepath = os.path.join(self.logdir, f"perturbation_{self.query_counter}.npy")
        np.save(filepath, self.perturbation.numpy())

    def _run_one_epoch(self, lrs=None, budget=None):
        if self.a < self.mask_thresh:
            self._reduce_mask()
        if budget is not None:
            if self.query_counter >= budget:
                return
        for _ in range(self.params["num_boost_iters"]):
            gradient = self._estimate_gradient()
            if budget is not None:
                if self.query_counter >= budget:
                    return

            self._update_perturbation(gradient, lrs=lrs)
            if budget is not None:
                if self.query_counter >= budget:
                    return

    def fit(self, epochs=None, budget=None, lrs=None, eval_every=1):
        """
        Fit the model.

        :epochs: int; train for this many perturbation update steps
        :budget: int; train for this many queries to the black-box model
        :lrs: list; learning rates to test for GRAPHITE
        :eval_every: int; wait this many epochs between evaluations
        """
        self._log_image()

        if epochs is not None:
            for e in tqdm(range(epochs)):
                self._run_one_epoch(lrs=lrs)
                if (eval_every > 0) & ((e + 1) % eval_every == 0):
                    self.evaluate()
                    self._save_perturbation()
                    self._log_image()

        elif budget is not None:
            i = 1
            progress = tqdm(total=budget)
            while self.query_counter < budget:
                old_qc = self.query_counter
                self._run_one_epoch(lrs=lrs)
                progress.update(n=self.query_counter - old_qc)
                if (eval_every > 0) & (i % eval_every == 0):
                    self.evaluate()
                    self._save_perturbation()
                    self._log_image()
                i += 1

            progress.close()
        else:
            print("WHAT DO YOU WANT FROM ME?")
