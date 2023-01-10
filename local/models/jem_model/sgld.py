import torch
import numpy as np
import random
import math

from .utils import framing


def get_grad(x_k, energy_f, mse_norm=None, return_all_acc=False):
    kwa = {}
    if mse_norm is not None:
        kwa["mse_norm"] = mse_norm
    if return_all_acc:
        energy, acc_v = energy_f(x_k, return_all_acc=True, **kwa)
    else:
        energy = energy_f(x_k, **kwa)
    grad = torch.autograd.grad(energy, [x_k], retain_graph=True)[0]
    if return_all_acc:
        return grad, energy.item(), acc_v
    return grad, energy.item()


class SgldSampler:
    def __init__(self, n_classes, rep_buf_px_len, rep_buf_px_giv_y_len, lr, noise_decrease, lr_decay, momentum, reinit_freq,
                 n_steps, instance_shape, device, adaptive_lr_mult=1.0, adaptive_noise_decrease_mult=1.0, adaptive_lr_decay_mult=1.0,
                 adaptive_momentum_mult=1.0, update_per_iters=99, init_samples_normal_d=True): #, rep_buf_px_steps=None, rep_buf_px_giv_y_steps=None):
        self.lr_px, self.lr_px_giv_y = lr, lr
        self.noise_decrease_px, self.noise_decrease_px_giv_y = noise_decrease, noise_decrease
        self.lr_decay_px, self.lr_decay_px_giv_y = lr_decay, lr_decay
        self.momentum_px, self.momentum_px_giv_y = momentum, momentum

        self.reinit_freq = reinit_freq
        self.n_steps = n_steps

        self.instance_shape = instance_shape  # (1, 21, 80)
        self.device = device
        # Whether to use normal (speech) or uniform (vision) distribution of random samples
        self.init_samples_normal_d = init_samples_normal_d

        self.rep_buf_px_len = rep_buf_px_len
        self.rep_buf_px_giv_y_len = rep_buf_px_giv_y_len
        #print("REP_BUF", rep_buf_px_len, rep_buf_px_giv_y_len)
        self.rep_buf_px_giv_y_len_per_class = self.rep_buf_px_giv_y_len // n_classes
        self.rep_buf_px = self.init_random(self.rep_buf_px_len)
        self.rep_buf_px_giv_y = self.init_random(self.rep_buf_px_giv_y_len)
        # Adjust sgld LR
        self.adaptive_lr_mult = [adaptive_lr_mult, 1.0, (1. / adaptive_lr_mult)]
        self.adaptive_noise_decrease_mult = [adaptive_noise_decrease_mult, 1.0, (1. / adaptive_noise_decrease_mult)]
        self.adaptive_lr_decay_mult = [adaptive_lr_decay_mult, 1.0, (1. / adaptive_lr_decay_mult)]
        self.adaptive_momentum_mult = [adaptive_momentum_mult, 1.0, (1. / adaptive_momentum_mult)]

        # How many iterations between updates
        self.update_per_iters = update_per_iters

        # How much iterations have been done since last LR update
        self.adj_c_px = 0  # How much iterations have been done since last LR update
        self.adj_c_px_giv_y = 0
        self.which_to_update_px = 0
        self.which_to_update_px_giv_y = 0
        self.adj_lrs_scores_px = [[], [], []]
        self.adj_lr_decay_scores_px = [[], [], []]
        self.adj_momentum_scores_px = [[], [], []]
        self.adj_noise_decrease_scores_px = [[], [], []]
        self.adj_noise_decrease_scores_px_giv_y = [[], [], []]
        self.adj_lrs_scores_px_giv_y = [[], [], []]
        self.adj_lr_decay_scores_px_giv_y = [[], [], []]
        self.adj_momentum_scores_px_giv_y = [[], [], []]

        self.verbose = True
        self.base_lr_mult = 1.5
        self.base_lr_pow = 1.05

    def get_state(self):
        state_dict = dict()
        state_dict["rep_buf_px"] = self.rep_buf_px
        state_dict["rep_buf_px_giv_y"] = self.rep_buf_px_giv_y
        state_dict["n_steps"] = self.n_steps
        state_dict["rep_buf_px_giv_y"] = self.rep_buf_px_giv_y
        state_dict["rep_buf_px"] = self.rep_buf_px
        state_dict["lr_px"] = self.lr_px
        state_dict["lr_px_giv_y"] = self.lr_px_giv_y
        state_dict["noise_decrease_px"] = self.noise_decrease_px
        state_dict["noise_decrease_px_giv_y"] = self.noise_decrease_px_giv_y
        state_dict["lr_decay_px"] = self.lr_decay_px
        state_dict["lr_decay_px_giv_y"] = self.lr_decay_px_giv_y
        state_dict["momentum_px"] = self.momentum_px
        state_dict["momentum_px_giv_y"] = self.momentum_px_giv_y
        state_dict["init_samples_normal_d"] = self.init_samples_normal_d
        # state_dict["rep_buf_px_steps"] = self.rep_buf_px_steps
        # state_dict["rep_buf_px_giv_y_steps"] = self.rep_buf_px_giv_y_steps
        return state_dict

    def load_state(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)
        if "rep_buf_px" in state_dict:
            self.rep_buf_px = self.rep_buf_px.to("cpu")
        if "rep_buf_px_giv_y" in state_dict:
            self.rep_buf_px_giv_y = self.rep_buf_px_giv_y.to("cpu")

    def init_random(self, how_many):
        # Shape: bs, n_ch, im_sz, im_sz
        #   -->: bs, 1, time (21), features (80 mel f-banks)
        if self.init_samples_normal_d:
            return torch.FloatTensor(how_many, *self.instance_shape).normal_(0, 1)
        else:
            return torch.FloatTensor(how_many, *self.instance_shape).uniform_(-1, 1)

    def get_batch_rec_f(self, shift):
        return lambda x: framing(x.squeeze(0), self.instance_shape[1], shift).unsqueeze(1)

    # This is very specific for current shape
    def init_sequence(self, how_long, shift=1, init_sample=None):
        # Shape: bs, n_ch, im_sz, im_sz
        #   -->: bs, 1, time (21), features (80 mel f-banks)
        if init_sample is None:
            sh = list(self.instance_shape)
            sh[1] = how_long  # time domain
            init_sample = torch.FloatTensor(*tuple(sh)).normal_(0, 1).to(self.device)
        batch_rec_f = self.get_batch_rec_f(shift)

        # Each part of recording is used by different number of windows -> calculate how many and normalize by that
        grad_normalizer = torch.zeros(how_long)
        for i in range(0, batch_rec_f(init_sample).size(0)):
            grad_normalizer[shift * i:shift * i + self.instance_shape[1]] += 1
        return init_sample, batch_rec_f, grad_normalizer.to(self.device)[:,None]

    def sample_px_during_training(self, f, bs):
        lr, noise_decrease, lr_decay, momentum = self.lr_px, self.noise_decrease_px, self.lr_decay_px, self.momentum_px

        ind_lr, ind_noise_decrease, ind_lr_decay, ind_momentum = 1, 1, 1, 1
        if self.which_to_update_px == 0:
            ind_lr = random.randrange(3)
        elif self.which_to_update_px == 1:
            ind_noise_decrease = random.randrange(3)
        elif self.which_to_update_px == 2:
            ind_lr_decay = random.randrange(3)
        elif self.which_to_update_px == 3:
            ind_momentum = random.randrange(3)

        lr *= self.adaptive_lr_mult[ind_lr]
        noise_decrease *= self.adaptive_noise_decrease_mult[ind_noise_decrease]
        lr_decay *= self.adaptive_lr_decay_mult[ind_lr_decay]
        momentum *= self.adaptive_momentum_mult[ind_momentum]

        samples = self.__sample_px(f, bs, lr, noise_decrease, lr_decay, momentum, self.n_steps, True)
        logits = f.get_logits(samples)
        # You can change this metric to anything else
        score = logits.logsumexp(-1).mean().item()

        if self.which_to_update_px == 0:
            self.adj_lrs_scores_px[ind_lr].append(score)
        elif self.which_to_update_px == 1:
            self.adj_noise_decrease_scores_px[ind_noise_decrease].append(score)
        elif self.which_to_update_px == 2:
            self.adj_lr_decay_scores_px[ind_lr_decay].append(score)
        elif self.which_to_update_px == 3:
            self.adj_momentum_scores_px[ind_momentum].append(score)
        self.adj_c_px += 1

        if (self.adj_c_px % self.update_per_iters) == 0:
            if self.which_to_update_px == 0:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_lrs_scores_px]))
                self.lr_px *= self.adaptive_lr_mult[max_sc_ind]
            elif self.which_to_update_px == 1:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_noise_decrease_scores_px]))
                self.noise_decrease_px *= self.adaptive_noise_decrease_mult[max_sc_ind]
            elif self.which_to_update_px == 2:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_lr_decay_scores_px]))
                self.lr_decay_px *= self.adaptive_lr_decay_mult[max_sc_ind]
                self.lr_decay_px = min(1.0, self.lr_decay_px)
            elif self.which_to_update_px == 3:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_momentum_scores_px]))
                self.momentum_px *= self.adaptive_momentum_mult[max_sc_ind]
                self.momentum_px = min(0.95, self.momentum_px)
            self.which_to_update_px = random.randrange(4)
            self.adj_lrs_scores_px = [[], [], []]
            self.adj_noise_decrease_scores_px = [[], [], []]
            self.adj_lr_decay_scores_px = [[], [], []]
            self.adj_momentum_scores_px = [[], [], []]
        return samples, logits

    def sample_px_giv_y_during_training(self, f, y):
        lr, noise_decrease, lr_decay, momentum = self.lr_px_giv_y, self.noise_decrease_px_giv_y, self.lr_decay_px_giv_y, self.momentum_px_giv_y

        ind_lr, ind_noise_decrease, ind_lr_decay, ind_momentum = 1, 1, 1, 1
        if self.which_to_update_px_giv_y == 0:
            ind_lr = random.randrange(3)
        elif self.which_to_update_px_giv_y == 1:
            ind_noise_decrease = random.randrange(3)
        elif self.which_to_update_px_giv_y == 2:
            ind_lr_decay = random.randrange(3)
        elif self.which_to_update_px_giv_y == 3:
            ind_momentum = random.randrange(3)

        lr *= self.adaptive_lr_mult[ind_lr]
        noise_decrease *= self.adaptive_noise_decrease_mult[ind_noise_decrease]
        lr_decay *= self.adaptive_lr_decay_mult[ind_lr_decay]
        momentum *= self.adaptive_momentum_mult[ind_momentum]

        samples = self.__sample_px_giv_y(f, y, lr, noise_decrease, lr_decay, momentum, self.n_steps, True)
        logits = f.get_logits(samples)
        # You can change this metric to anything else
        score = logits.logsumexp(-1).mean().item()

        if self.which_to_update_px_giv_y == 0:
            self.adj_lrs_scores_px_giv_y[ind_lr].append(score)
        elif self.which_to_update_px_giv_y == 1:
            self.adj_noise_decrease_scores_px_giv_y[ind_noise_decrease].append(score)
        elif self.which_to_update_px_giv_y == 2:
            self.adj_lr_decay_scores_px_giv_y[ind_lr_decay].append(score)
        elif self.which_to_update_px_giv_y == 3:
            self.adj_momentum_scores_px_giv_y[ind_momentum].append(score)
        self.adj_c_px_giv_y += 1

        if (self.adj_c_px_giv_y % self.update_per_iters) == 0:
            if self.which_to_update_px_giv_y == 0:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_lrs_scores_px_giv_y]))
                self.lr_px_giv_y *= self.adaptive_lr_mult[max_sc_ind]
            elif self.which_to_update_px_giv_y == 1:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_noise_decrease_scores_px_giv_y]))
                self.noise_decrease_px_giv_y *= self.adaptive_noise_decrease_mult[max_sc_ind]
            elif self.which_to_update_px_giv_y == 2:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_lr_decay_scores_px_giv_y]))
                self.lr_decay_px_giv_y *= self.adaptive_lr_decay_mult[max_sc_ind]
                self.lr_decay_px_giv_y = min(1.0, self.lr_decay_px_giv_y)
            elif self.which_to_update_px_giv_y == 3:
                max_sc_ind = np.argmax(np.array([np.array(a).mean() for a in self.adj_momentum_scores_px_giv_y]))
                self.momentum_px_giv_y *= self.adaptive_momentum_mult[max_sc_ind]
                self.momentum_px_giv_y = min(0.95, self.momentum_px_giv_y)
            self.which_to_update_px_giv_y = random.randrange(4)
            self.adj_lrs_scores_px_giv_y = [[], [], []]
            self.adj_noise_decrease_scores_px_giv_y = [[], [], []]
            self.adj_lr_decay_scores_px_giv_y = [[], [], []]
            self.adj_momentum_scores_px_giv_y = [[], [], []]
        return samples, logits

    def __sample_px(self, f, bs, lr, noise_decrease, lr_decay, momentum, n_steps, replace_in_buffer, samples=None, en_f=None):
        if en_f is None:
            en_f = lambda x, *args: f(x).sum()
        if samples is None:
            if self.rep_buf_px_len < bs:
                samples = self.init_random(bs)
            else:
                inds = torch.randperm(self.rep_buf_px_len)[:bs]
                buffer_samples = self.rep_buf_px[inds]
                samples = self.randomly_reinitialize(buffer_samples)

        sgld_f = self.apply_sgld
        samples = sgld_f(f, en_f, samples.to(self.device), lr, noise_decrease, lr_decay, momentum, n_steps)

        if self.rep_buf_px_len > 0 and replace_in_buffer:
            self.rep_buf_px[inds] = samples.cpu()

        return samples

    def __sample_px_giv_y(self, f, y, lr, noise_decrease, lr_decay, momentum, n_steps, replace_in_buffer, samples=None, en_f=None): # HERE
        if en_f is None:
            en_f = lambda x,*args: f(x, y=y).sum()
            #def en_f(x, *args):
            #    print("HHH", x.shape, y.shape, f(x, y=y).shape, f(x).shape)
            #    1/0
            #   return f(x, y=y).sum()
        bs = y.size(0)
        if samples is None:
            if self.rep_buf_px_giv_y_len_per_class == 0:
                samples = self.init_random(bs)
            else:
                inds = torch.randint(0, self.rep_buf_px_giv_y_len_per_class, (bs,))
                inds = y.cpu() * self.rep_buf_px_giv_y_len_per_class + inds
                # WARN: It can rarely produce the same indices multiple times
                buffer_samples = self.rep_buf_px_giv_y[inds]
                samples = self.randomly_reinitialize(buffer_samples)

        sgld_f = self.apply_sgld
        samples = sgld_f(f, en_f, samples.to(self.device), lr, noise_decrease, lr_decay, momentum, n_steps)

        if self.rep_buf_px_giv_y_len > 0 and replace_in_buffer:
            self.rep_buf_px_giv_y[inds] = samples.cpu()
        return samples

    def randomly_reinitialize(self, buffer_samples):
        bs = buffer_samples.size(0)
        random_samples = self.init_random(bs)
        choose_random = (torch.rand(bs) < self.reinit_freq).float().reshape(-1, *(len(self.instance_shape) * [1]))
        # Does this for any dim: choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        return choose_random * random_samples + (1 - choose_random) * buffer_samples

    def apply_sgld(self, f, energy_f, batch, lr, noise_decrease, lr_decay, momentum, n_steps, grad_norm=1, get_grad_f=get_grad, print_energy=False):
        #print(batch.shape)
        momentum = min(0.9, momentum)
        f_orig_train_state = f.training
        f.eval()
        x_k = batch.clone().detach()
        x_k.requires_grad = True
        alpha = lr
        # sgld

        # Wrapper to print energy at step k
        def pr_energy_f(k, *args):
            energy = energy_f(*args)
            if print_energy:
                print(k, energy.item())
            return energy

        for k in range(n_steps):
            x_k_grad = get_grad_f(x_k, lambda *args: pr_energy_f(k, *args))[0] / grad_norm
            #x_k.data += lr * f_prime + std * torch.randn_like(x_k)
            if k == 0:
                acc_grad = x_k_grad.clone()
            else:
                acc_grad.data *= momentum
                acc_grad.data += (1-momentum) * x_k_grad
            x_k.data += 0.5 * alpha * acc_grad + math.sqrt(alpha) / noise_decrease * torch.randn_like(x_k)
            alpha *= lr_decay
        final_samples = x_k.detach()
        # update replay buffer
        f.train(f_orig_train_state)
        return final_samples
