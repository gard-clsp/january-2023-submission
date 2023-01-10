"""
Sliding Joint Energy-based Model for speech commands classification.
Author: Martin Sustek
"""
import copy
import json
from os.path import dirname, abspath
from types import SimpleNamespace
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from art.estimators.classification import PyTorchClassifier
import random
from tqdm import tqdm
from .jem_model.model_and_data import get_dataloader, get_model_and_sampler
from .jem_model.utils import load_from_checkpoint, checkpoint


class FeatLoader:
    def __init__(self):
        from dataclasses import dataclass, asdict
        @dataclass
        class FbankConfig:
            # Spectogram-related part
            dither: float = 0.0
            window_type: str = "povey"
            frame_length: float = 25.
            frame_shift: float = 10.
            remove_dc_offset: bool = True
            round_to_power_of_two: bool = True
            energy_floor: float = 1e-10
            min_duration: float = 0.0
            preemphasis_coefficient: float = 0.97
            raw_energy: bool = True

            # Fbank-related part
            low_freq: float = 20.0
            high_freq: float = -400.0
            num_mel_bins: int = 40
            use_energy: bool = False
            vtln_low: float = 100.0
            vtln_high: float = -500.0
            vtln_warp: float = 1.0

        params = asdict(FbankConfig())
        params.update({"sample_frequency": 16000, "snip_edges": False, "num_mel_bins": 80})
        self.params = params
        self.recordings = None
        self.supervision_set = None

    def get_features(self, x):
        sq = x.ndim == 3
        unsq = x.ndim == 1
        if sq:
            x = x.squeeze(0)
        if unsq:
            x = x.unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(x, **self.params)
        if sq:
            feat = feat.unsqueeze(0)
        return feat


def get_args(load_path, args=None):
    with open("{}/variant.json".format(dirname(abspath(load_path)))) as json_file:
        configs = json.load(json_file)
    loaded = SimpleNamespace(**configs)
    if args is not None:
        overwrite = copy.deepcopy(args)
        overwrite = vars(overwrite)
        overwrite.update(vars(loaded))
        overwrite = SimpleNamespace(**overwrite)
    else:
        overwrite = loaded
        if not hasattr(overwrite, "lm_stats_dir"):
            overwrite.lm_stats_dir = None

    if args is not None:
        overwrite.exp_prefix = "{}-{}".format(args.id, overwrite.exp_prefix)
        overwrite.seed = args.seed
        overwrite.log_dir = f"{args.log_dir}/{overwrite.exp_prefix}"
        overwrite.load_paths = args.load_paths
        overwrite.workers = args.workers
        overwrite.id = args.id
        overwrite.sigma = args.sigma
        overwrite.batch_size = args.batch_size
    args = overwrite
    if hasattr(args, "n_steps"):  # backward compatibility
        args.sgld_n_steps = args.n_steps
        del args.n_steps
    return args


class JEMClassifier(PyTorchClassifier):
    def __init__(self, config_args, already_trained=False, sgld_sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_f = FeatLoader().get_features
        self.args = config_args
        self.already_trained = already_trained
        self.sgld_sampler = sgld_sampler

    def get_features_mean_and_std(self, xs, ys, device):
        # For mean and std calc
        x_sum = torch.zeros(80, dtype=torch.float64, device=device)
        x_sq_sum = torch.zeros(80, dtype=torch.float64, device=device)
        num = torch.zeros(1, dtype=torch.int64, device=device)

        x_feats = []
        #print("xs", xs.shape)
        for x in xs:
            #print("x", x.shape)
            x = self.feat_f(x)
            x = x.reshape(-1, x.shape[-1])

            x_feats.append(x.to("cpu"))
            #y_feats.append(y.repeat(len(x)))

            num += x.size(0)
            x_sum += x.sum(0)
            x_sq_sum += torch.square(x).sum(0)

        x_feats = torch.stack(x_feats)
        y_feats = torch.argmax(ys, dim=-1).to("cpu")
        mean = x_sum / num
        std = torch.sqrt(x_sq_sum / num - torch.square(mean))
        # feats on cpu; mean and std on the device
        return x_feats, y_feats, mean.to(torch.float), std.to(torch.float)

    def fit(  # pylint: disable=W0221
                self,
                x: np.ndarray,
                y: np.ndarray,
                batch_size: int = 128,
                nb_epochs: int = 10,
                training_mode: bool = True,
                drop_last: bool = False,
                scheduler: Optional[Any] = None,
                **kwargs,
        ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.
        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        if self.already_trained:
            return

        self._model.train(mode=training_mode)

        args = self.args
        f = self._model._model
        sgld_sampler = self.sgld_sampler
        device = self._device
        #print(y.shape, y[:10])
        #print(x.shape, x[:10])
        #1/0
        #args.n_epochs = 2

        with torch.no_grad():
            x_feat, y_feat, mean, std = self.get_features_mean_and_std(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), device)

        f.set_mean_and_std(mean, std)
        #print(x_feat.shape, y_feat.shape)
        kwa = {"device": device, "preloaded_xs": x_feat, "preloaded_ys": y_feat}
        if args.process_whole_recording:
            kwa["make_frames"] = False
        dload_train = get_dataloader(args, "train", **kwa)
        del x
        del y
        del x_feat
        del y_feat

        params = f.parameters()
        if args.optimizer == "adam":
            optim = torch.optim.Adam(params, lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)
            # optim = torch.optim.Adam(params, lr=args.lr, betas=[0.0, 0.9], weight_decay=args.weight_decay)
        else:
            optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        cur_iter = 0
        xent = nn.CrossEntropyLoss()

        for ep in range(args.start_epoch, args.n_epochs):
            if not args.sgld_fix_steps:
                sgld_sampler.n_steps += 1
            # Decay lr
            if ep in args.decay_epochs:
                for param_group in optim.param_groups:
                    new_lr = param_group["lr"] * args.decay_rate
                    param_group["lr"] = new_lr

            for i, (batch_x, batch_y) in tqdm(enumerate(dload_train)):
                # Uncomment for trial run ---- 
                #if i >= 5:
                #    print("Limiting epoch to have max 5 iteration.")
                #    break
                #print(batch_x.size(), batch_y.size())

                if not args.process_whole_recording and batch_y.size(0) < args.batch_size:
                    print("Skipping batch because it is too short.", batch_y.size(0))
                    continue
                assert len(batch_y) == len(
                    batch_x), f"Batch size of input {batch_x.shape} must be the same as batch size of labels {batch_y.shape}"
                assert len(
                    batch_y) == args.batch_size, f"In current setup, each batch should have BS size, can be removed."
                batch_x, batch_y = f.normalize_input(batch_x.to(device)), batch_y.to(device)

                if args.grad_penalty > 0:
                    batch_x.requires_grad = True

                # Warmup
                if cur_iter <= args.warmup_iters:
                    lr = args.lr * cur_iter / float(args.warmup_iters)
                    for param_group in optim.param_groups:
                        param_group["lr"] = lr

                L = 0.0
                l_pyxce, l_pxyce = 0, 0
                logits = f.get_logits(batch_x)  # For now treat as batch
                # print("LOGITS_SH", logits.shape, batch_y.shape)
                if args.process_whole_recording:
                    batch_y = batch_y.reshape(-1)
                    if not args.classify_whole_recording:
                        logits = logits.reshape(-1, logits.shape[-1])
                        args.batch_size = logits.size(0)  # virtually this is batch_size
                # print("Logits shape ", logits.shape)
                # Supporting something that is not using here -> logits_reshape = logits here
                logits_reshape = logits.reshape(-1, logits.shape[-1])
                logits_orig_shape = logits.shape[0]
                # print(batch_y.shape, batch_y.type(), logits_reshape.shape, logits_reshape.type())
                if args.pyxce > 0:
                    l_pyxce = xent(logits_reshape, batch_y.repeat(logits_reshape.size(0) // batch_y.size(0)))
                if args.pxyce > 0:
                    y_one_hot = nn.functional.one_hot(batch_y, num_classes=args.n_classes)
                    batch_y_c = torch.sum(y_one_hot, 0)
                    if args.sgld_batch_size > 0:
                        sample_from_px = random.random() < args.pxy_sgld_p_x
                        if sample_from_px:
                            _, fake_log = sgld_sampler.sample_px_during_training(f, args.sgld_batch_size)
                        else:
                            # sample from p(x,y=c) or p(x|y=c) where p(y=c) corresponds to p(y=c) in the current batch
                            y_q = torch.multinomial(batch_y_c.float(), args.sgld_batch_size, replacement=True).to(
                                device)
                            # For now treat as batch
                            _, fake_log = sgld_sampler.sample_px_giv_y_during_training(f, y_q)
                        if args.process_whole_recording:
                            fake_log = fake_log.reshape(-1, fake_log.shape[-1])
                        logits = torch.cat((logits, fake_log), -2)

                    u_log_py = logits.logsumexp(-2)
                    pxyce_lse = torch.sum(batch_y_c * u_log_py)
                    l_pxyce = (pxyce_lse - torch.sum(f.forward_logits(logits, batch_y))) / logits_reshape.shape[0]

                L += args.pyxce * l_pyxce + args.pxyce * l_pxyce

                if args.grad_penalty > 0:
                    real_data_lse = logits[..., :args.batch_size, :].logsumexp(-1)

                    grad_inp = torch.autograd.grad(real_data_lse.sum(), batch_x, create_graph=True)[0].flatten(
                        start_dim=1).norm(2, 1)
                    grad_pen = (0.5 * (grad_inp ** 2.)).mean() * args.grad_penalty
                    L += grad_pen

                optim.zero_grad()
                L.backward()
                try:
                    optim.step()
                except AssertionError:  # Loading model - pytorch versions
                    optim.param_groups[0]["capturable"] = True
                    optim.step()
                cur_iter += 1
        try:
            checkpoint(f, optim, sgld_sampler, f"ckpt_{args.n_epochs}.pt", args, device)
        except:
            print("Couldn't save checkpoint.")

    # Keeping here just because it was here - TODO: might be removed?
    def _apply_preprocessing(self, x, y, fit: bool = False, no_grad=True):
        device = self.preprocessing_operations[0].device
        for operation in self.preprocessing_operations:
            operation._device = 'cpu'
        x = torch.tensor(x, device = 'cpu')
        if y is not None:
            y = torch.tensor(y, device = 'cpu')
        x, y = super()._apply_preprocessing(x, y, fit, no_grad)
        for operation in self.preprocessing_operations:
            operation._device = device
        x = x.detach().numpy()
        if y is not None:
            y = y.detach().numpy()
        return x, y

    def predict(  # pylint: disable=W0221
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import torch
        device = self._device

        f = self._model._model
        f_orig_train_state = f.training
        f = f.to(device)
        f.eval()
        f_orig_batch_time_together = f.f.batch_time_together
        f.f.batch_time_together = False

        with torch.no_grad():
            x = torch.from_numpy(x).to(device)
            # IDK how to use features in parallel
            x_feat = []
            for x_rec in x:
                x_feat.append(self.feat_f(x_rec).unsqueeze(0))
            x = f.normalize_input(torch.stack(x_feat))
            logits = f.get_logits(x).logsumexp(1)

        f.train(f_orig_train_state)
        f.f.batch_time_together = f_orig_batch_time_together
        return logits.detach().cpu().numpy()


def get_new_model(model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None, device="cuda"):
    with open(model_kwargs["variant_path"]) as json_file:
        args = SimpleNamespace(**json.load(json_file))
    # TODO: workers?
    #args.process_whole_recording = True
    args.num_workers = 0  # keeping everything in memory
    f, sgld_sampler = get_model_and_sampler(args, device)

    return JEMClassifier(
        model=f,
        sgld_sampler=sgld_sampler,
        config_args=args,
        already_trained=False,
        loss=nn.CrossEntropyLoss(),
        input_shape=(16000,),
        nb_classes=12,
        **wrapper_kwargs,
    )


def get_trained_model(model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None, device="cuda"):
    model_path = model_kwargs["model_path"]
    args = get_args(model_path)
    #args.process_whole_recording = True
    model, sgld_sampler = get_model_and_sampler(args, device, sampler=False)
    load_from_checkpoint(model_path, model, device, sgld_sampler=sgld_sampler)
    model = model.to(device)
    model.eval()

    return JEMClassifier(
        model=model,
        already_trained=True,
        config_args=args,
        sgld_sampler=sgld_sampler,
        loss=nn.CrossEntropyLoss(),
        input_shape=(16000,),
        nb_classes=12,
        **wrapper_kwargs,
    )


def get_model(model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None):
    # SGE safety (we use gpu when available)
    if torch.cuda.is_available():
        device = "cuda"
        ttmp = torch.zeros([1, 1]).to(device)
    else:
        device = "cpu"
    wrapper_kwargs['device_type'] = device
    print(device)
    if 'no_train' in model_kwargs and model_kwargs["no_train"]:
        print("Loading from checkpoint...")
        return get_trained_model(model_kwargs, wrapper_kwargs, weights_path, device)
    else:
        print("Training new model...")
        return get_new_model(model_kwargs, wrapper_kwargs, weights_path, device)
