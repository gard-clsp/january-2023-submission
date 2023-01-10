import torch.nn as nn
from torch.utils import data

import os
import random
import torch

from .utils import framing
from .sgld import SgldSampler


def get_f(args, kw_resnet=None):
    if kw_resnet is None:
        kw_resnet = {"depth": 28, "widen_factor": 10, "norm": None, "dropout_rate": 0.0, "input_channels": 1,
                     "input_size": (2 * args.context_len + 1, args.input_dim)}
    if hasattr(args, "process_whole_recording") and args.process_whole_recording:
        kw_resnet["batch_time_together"] = False
    if kw_resnet["input_size"][0] >= 51:  # can be larger but not smaller due to reduction
        print("Creating no pad, no stride WRN where all convs have kernel_size 3.")
        return WideResNetTimeNoPadNoStride(conv1_kernel_size=3, **kw_resnet)
    elif kw_resnet["input_size"][0] >= 27:
        print("Creating no pad, no stride WRN where some convs have kernel_size 1.")
        return WideResNetTimeNoPadNoStride(conv1_kernel_size=1, **kw_resnet)


class Model(nn.Module):
    def __init__(self, args, device, mean=0, std=1, f=None, last_dim_f=lambda f: f.last_dim, kw_resnet=None):
        super(Model, self).__init__()
        self.mean = mean
        self.std = std
        self.f = f if f is not None else get_f(args, kw_resnet)
        self.class_output = nn.Linear(last_dim_f(self.f), args.n_classes)
        self.classify_whole_rec = hasattr(args, 'classify_whole_recording') and args.classify_whole_recording

    def normalize_input(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def add_noise(self, x):
        return x + self.noise * torch.randn_like(x)

    def denormalize_input(self, x):
        return (x * self.std.to(x.device)) + self.mean.to(x.device)

    def pred_from_logits(self, logits):
        return logits

    def forward_logits(self, logits, y=None):
        if y is None:
            return logits.logsumexp(-1)
        else:
            return self.pick_indices(logits, y)

    def forward(self, x, y=None, normalize_input=False, noise=0):
        return self.forward_logits(self.get_logits(x, normalize_input, noise), y)

    def pick_indices(self, logits, y):
        if logits.ndim - 1 > y.ndim:
            if len(y) == len(logits):  # labels are given per recording - repeat
                y = y[:, None].repeat(1, logits.size(-2)).reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))
        return torch.gather(logits, -1, y[:, None])

    def get_logits(self, x, normalize_input=False, noise=0, eval_time=False):  # eval_time just for compatibility
        if normalize_input:
            x = self.normalize_input(x)
        if noise != 0:
            x = self.add_noise(x, noise)
        logits = self.class_output(self.f(x))
        #print(logits.shape)
        if self.classify_whole_rec:  # logsumexp over time - potential place for dropout
            logits = logits.logsumexp(1)
            #print("L2", logits.shape)
        return logits

    def get_posterior(self, x=None, y=None, normalize_input=False, noise=0, logits=None, log_domain=False):
        logits = logits if logits is not None else self.get_logits(x, normalize_input, noise)
        if log_domain:
            out = nn.functional.log_softmax(logits, dim=-1)
        else:
            out = nn.functional.softmax(logits, dim=-1)
        if y is not None:
            out = self.pick_indices(out, y)
        return out

    def get_mean_and_std_dict(self):
        return {"mean": self.mean, "std": self.std}

    def set_mean_and_std(self, mean, std):
        print(f"Setting mean: {mean}")
        print(f"Setting std: {std}")
        self.mean = mean
        self.std = std


def get_dataloader(args, dset_part, shift_and_shuffle=None, data_dir=None, evaluation=False, make_frames=True,
                   inc_poison=False, device="cpu", preloaded_xs=None, preloaded_ys=None):
    tr_shift_shuffle = True if shift_and_shuffle is None else shift_and_shuffle
    te_val_shift_shuffle = False if shift_and_shuffle is None else shift_and_shuffle

    data_dir = data_dir if data_dir is not None else args.data_dir
    dset_args = {"context_len": args.context_len, "ignore_silence": args.ignore_silence, "make_frames": make_frames}

    if preloaded_xs is not None:
        dset_args["preloaded_xs"] = preloaded_xs
    if preloaded_ys is not None:
        dset_args["preloaded_ys"] = preloaded_ys
        return get_dataloader_poison(args, dset_part, data_dir, evaluation, dset_args, tr_shift_shuffle,
                                     te_val_shift_shuffle, poison_ind=inc_poison, device=device)


def get_model_and_sampler(args, device, sampler=True):
    mean = torch.tensor(0, device=device) if (args.mean_path is None) else torch.load(args.mean_path).float().to(device)
    std = torch.tensor(1, device=device) if (args.std_path is None) else torch.load(args.std_path).float().to(device)

    f = Model(args, device, mean, std)
    f = f.to(device)
    sgld_sampler = None
    if sampler:
        assert 1.0 >= args.pxy_sgld_p_x >= 0.0
        rep_buf_px_len = int(args.pxy_sgld_p_x * args.buffer_size)
        rep_buf_px_giv_y_len = int((1 - args.pxy_sgld_p_x) * args.buffer_size)

        if hasattr(args, "process_whole_recording_sgld") and args.process_whole_recording_sgld:
            shape = (1, 100, args.input_dim) # TODO: set size by parameter, for now meant for poisoned data
        else:
            shape = (1, (2 * args.context_len) + 1, args.input_dim)

        #args.sgld_n_steps
        if not hasattr(args, 'sgld_update_per_iters'):  # Old compatibility, useless after training
            args.sgld_update_per_iters = 99

        sgld_sampler = SgldSampler(args.n_classes, rep_buf_px_len, rep_buf_px_giv_y_len, args.sgld_lr, args.sgld_noise_decrease,
                                   args.sgld_lr_decay, args.sgld_momentum, args.reinit_freq, args.sgld_n_steps, shape, device,
                                   args.sgld_adaptive_lr_mult, args.sgld_adaptive_noise_decrease_mult, args.sgld_adaptive_lr_decay_mult,
                                   args.sgld_adaptive_momentum_mult, args.sgld_update_per_iters, not hasattr(args, 'dataset') or args.dataset == "speech")

    return f, sgld_sampler


def conv3x3(in_planes, out_planes, stride=1, time_padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=(time_padding, 1), bias=True)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, time_stride=1, freq_stride=1, norm=None, leak=0.2, time_padding=1, conv1_kernel_size=3):
        super(WideBasic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.time_stride = time_stride
        self.freq_stride = freq_stride
        self.conv2_time_padding = time_padding

        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_time_padding = time_padding if self.conv1_kernel_size == 3 else 0
        assert conv1_kernel_size == 1 or conv1_kernel_size == 3
        assert time_padding == 1 or time_padding == 0
        # TODO: 1Dconv
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(conv1_kernel_size, 3), padding=(self.conv1_time_padding, 1), bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=(time_stride, freq_stride), padding=(self.conv2_time_padding, 1), bias=True)

        self.shortcut = nn.Sequential()
        if time_stride != 1 or freq_stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=(time_stride, freq_stride), bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        cut_off = (1 - self.conv2_time_padding) * (self.conv1_kernel_size + 1) // 2
        out += self.shortcut(x[..., cut_off:-cut_off, :])
        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class WideResNetTimeNoPadNoStride(nn.Module):
    def __init__(self, depth, widen_factor, input_channels=3, norm=None, leak=0.2, dropout_rate=0.0, input_size=(32, 32), conv1_kernel_size=3, batch_time_together=True):
        super(WideResNetTimeNoPadNoStride, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.input_size = input_size
        self.batch_time_together = batch_time_together

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]
        time_padding = 0
        time_stride = 1

        self.conv1 = conv3x3(input_channels, nStages[0], time_padding=time_padding)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, time_stride=1, freq_stride=1, time_padding=time_padding, conv1_kernel_size=conv1_kernel_size)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, time_stride=time_stride, freq_stride=2, time_padding=time_padding, conv1_kernel_size=conv1_kernel_size)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, time_stride=time_stride, freq_stride=2, time_padding=time_padding, conv1_kernel_size=conv1_kernel_size)
        self.bn1 = get_norm(nStages[3], self.norm)
        reduced_size_in_time = self.input_size[0] - 2 - 3 * len(nStages) * (conv1_kernel_size + 1)
        #self.pool = lambda out: F.avg_pool2d(out, ((self.input_size[0] + 3) // 4, out.size(-1)))
        self.pool = nn.Conv2d(nStages[3], nStages[3], groups=nStages[3], kernel_size=(reduced_size_in_time, self.input_size[1]//4), bias=True)
        self.last_dim = nStages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, time_stride, freq_stride, time_padding, conv1_kernel_size=3):
        strides = [(time_stride, freq_stride)] + [[1, 1]] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, *stride, norm=self.norm, time_padding=time_padding, conv1_kernel_size=conv1_kernel_size))
            self.in_planes = planes

        return nn.Sequential(*layers)
        #return InputAndContextSequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        out = self.pool(out).squeeze(-1)  # BS, FEAT, TIME
        out = torch.swapaxes(out, -2, -1)  # BS, TIME, FEAT
        if self.batch_time_together:
            out = out.reshape(-1, out.size(-1))  # BS_TIME, FEAT
        return out


# Just for compatibility with K2 dataloader
def wrap_train_dl(dloader):
    dloader.before_epoch = lambda _: None
    return dloader


def collate_frames(batch):
#    for i in batch:
#        print(i)
    al = list(map(torch.cat, list(zip(*batch))))
    x, rest = al[0], al[1:]
    return x[..., None, :, :], *rest


def collate_recordings(batch):
    al = list(map(torch.stack, list(zip(*batch))))
    x, rest = al[0], al[1:]
    #print(x.shape, y.shape)
    return x[..., None, :, :], *rest


class Sequence2Frames:
    silence_label = 0

    def __init__(self, context_len=10, frame_shift=1, train_shift=False, ignore_silence=True, make_frames=True):
        self.context_len = context_len
        self.frame_shift = frame_shift
        self.window_size = self.context_len * 2 + 1
        self.ignore_silence = ignore_silence
        self.id2silence = dict()
        self.train_shift = train_shift
        self.make_frames = make_frames

    def get_start_and_end_of_silence(self, index, labels):
        if not self.ignore_silence:
            return 0, None
        if index in self.id2silence:
            return self.id2silence[index]
        first = self.context_len
        last = len(labels) - 1 - self.context_len
        while first <= last and labels[first] == Sequence2Frames.silence_label:
            first += 1
        while first < last and labels[last] == Sequence2Frames.silence_label:
            last -= 1
        self.id2silence[index] = (first, last)
        #print(self.id2silence[index][0], len(labels) - self.id2silence[index][1])
        return self.id2silence[index]

    def make_frames_from_sequence(self, x, y, index=None):
        # Input x shape [T-time,C-features], y shape [T]
        start_shift, end_shift = self.get_start_and_end_of_silence(index, y)
        x = x[start_shift:end_shift, :]
        y = y[start_shift:end_shift]
        if not self.make_frames:
            #print(x.shape)
            return x, y[self.context_len:-self.context_len]

        if self.train_shift:
            curr_fr_shift = random.randrange(self.frame_shift)
            x = x[curr_fr_shift:, :]
            y = y[curr_fr_shift:]

        assert len(x) == len(y), f"X len is {len(x)}, while Y len is {len(y)}"
        return framing(x, self.window_size, self.frame_shift), y[self.context_len:-self.context_len:self.frame_shift]


class MyDatasetPoison(data.Dataset, Sequence2Frames):
    def __init__(self, path, context_len=10, frame_shift=1, train_shift=False, ignore_silence=False, make_frames=True, cl_wh_rec=False, poison_ind=False, preloaded_xs=None, preloaded_ys=None):
        assert not ignore_silence, "Cannot ignore silence for Poison data"
        Sequence2Frames.__init__(self, context_len, frame_shift, train_shift, False, make_frames)
        data.Dataset.__init__(self)
        self.path = path
        self.preloaded_xs = preloaded_xs
        if preloaded_ys is not None:
            self.labels = preloaded_ys
        else:
            self.labels = torch.load(os.path.join(path, f'y.pt'))
        self.cl_wh_rec = cl_wh_rec
        self.include_poison_indicator = poison_ind
        if self.include_poison_indicator:
            self.turn_poisoned_y_on()
            #print("I", self.pois_ind.sum())

        #if path.endswith("et_simu_8ch"):
        #    self.labels = {k.replace("8ch", "1ch"): v for k, v in self.labels.items()}

    def __len__(self):
        return len(self.labels)

    def turn_poisoned_y_on(self):
        was_set = self.include_poison_indicator
        self.include_poison_indicator = True
        if not hasattr(self, "pois_ind"):
            self.pois_ind = torch.load(os.path.join(self.path, f'y_pois.pt')).to(torch.bool)
        return was_set

    def turn_poisoned_y_off(self):
        self.include_poison_indicator = False

    def is_poisoned_y_set(self):
        return self.include_poison_indicator

    def get_item(self, index):
        if self.preloaded_xs is not None:
            x = self.preloaded_xs[index]
        else:
            x = torch.load(os.path.join(self.path, "data", f"x{index}.pt"))
        y = self.labels[index]
        if self.cl_wh_rec:
            return x, y
        y = y.repeat(x.size(0))
        return self.make_frames_from_sequence(x, y, index)

    def __getitem__(self, index):
        if self.include_poison_indicator:
            return *self.get_item(index), self.pois_ind[index]
        return self.get_item(index)


def get_dataloader_poison(args, dset_part, data_dir, evaluation, dset_args, tr_shift_shuffle, te_val_shift_shuffle, poison_ind=False, device="cpu"):
    # TBD pin mem
    dl_args = {}
    if device != "cpu" and device != torch.device("cpu"):
        dl_args['pin_memory'] = True
    if dset_args["make_frames"]:
        dl_args["collate_fn"] = collate_frames
    else:
        dl_args["collate_fn"] = collate_recordings

    train_part_dir = "train"
    val_part_dir = "valid"
    test_part_dir = "test"

    num_workers_test = args.num_workers if evaluation else 1

    wh_rec = hasattr(args, "classify_whole_recording") and args.classify_whole_recording
    #print(os.path.join(data_dir, train_part_dir))
    if dset_part.lower() == "train":
        dset = MyDatasetPoison(os.path.join(data_dir, train_part_dir), train_shift=tr_shift_shuffle, frame_shift=args.frame_shift, cl_wh_rec=wh_rec, poison_ind=poison_ind, **dset_args)
        dload = wrap_train_dl(data.DataLoader(dset, shuffle=tr_shift_shuffle, num_workers=args.num_workers, batch_size=args.n_recordings_per_batch, **dl_args))
    elif dset_part.lower().startswith("val"):
        return None
        dset = MyDatasetPoison(os.path.join(data_dir, val_part_dir), train_shift=te_val_shift_shuffle, cl_wh_rec=wh_rec, poison_ind=poison_ind, **dset_args)
        #dload = data.DataLoader(dset, shuffle=te_val_shift_shuffle, num_workers=num_workers_test, batch_size=args.n_recordings_per_batch, **dl_args)
    elif dset_part.lower() == "test":
        dset = MyDatasetPoison(os.path.join(data_dir, test_part_dir), train_shift=te_val_shift_shuffle, cl_wh_rec=wh_rec, poison_ind=poison_ind, **dset_args)
        dload = data.DataLoader(dset, shuffle=te_val_shift_shuffle, num_workers=num_workers_test, batch_size=args.n_recordings_per_batch, **dl_args)
    else:
        raise ValueError(f"Incorrect dataset part: {dset_part}")
    return dload
