import os
import torch


def framing(a, window_size, shift=1, skip_between=0):
    window_size_in_orig = window_size + (window_size - 1) * skip_between
    shape = (max(0, (a.size(0) - window_size_in_orig) // shift + 1), window_size) + a.shape[1:]
    strides = (a.stride(0) * shift, a.stride(0) * (skip_between + 1)) + a.stride()[1:]
    return torch.as_strided(a, size=shape, stride=strides)


def checkpoint(f, optimizer, sgld_sampler, tag, args, device):
    f.cpu()
    ckpt_dict = {"model_state_dict": f.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 "sgld_state": sgld_sampler.get_state(), **f.get_mean_and_std_dict()}
    torch.save(ckpt_dict, os.path.join(args.log_dir, tag))
    f.to(device)


def load_from_checkpoint(load_path, f, device, optimizer=None, sgld_sampler=None):
    print(f"loading model from {load_path}")
    ckpt_dict = torch.load(load_path, map_location=device)
    ## TODO: legacy!
    if True:
        if "f.linear.weight" in ckpt_dict["model_state_dict"]:
            del ckpt_dict["model_state_dict"]["f.linear.weight"]
        if "f.linear.bias" in ckpt_dict["model_state_dict"]:
            del ckpt_dict["model_state_dict"]["f.linear.bias"]

        #ckpt_dict["model_state_dict"]["f.pool.weight"] = torch.ones(640, 1, 1, 20, device=device) / 20
        #ckpt_dict["model_state_dict"]["f.pool.bias"] = torch.zeros(640, device=device)
    f.load_state_dict(ckpt_dict["model_state_dict"])

    if "mean" not in ckpt_dict:
        #  TODO: remove - just for compatibility
        print("MEAN and STD missing, probably old model, loading expected mean and std...")
        bd = "/export/b15/msustek/speech_mi"
        mean = torch.load(os.path.join(bd, "mel_banks_mean.pt")).float()
        std = torch.load(os.path.join(bd, "mel_banks_std.pt")).float()
    else:
        mean, std = ckpt_dict["mean"], ckpt_dict["std"]
    f.set_mean_and_std(torch.tensor(mean, device=device), torch.tensor(std, device=device))

    if optimizer is not None:
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    if sgld_sampler is not None:
        sgld_sampler.load_state(ckpt_dict["sgld_state"])
