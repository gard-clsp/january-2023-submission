"""
Resnet for speech commands classification.
Originally implemented in armory, re-factored into pytorch
"""

from typing import Optional

import torch
import torchvision
import torch.nn as nn
import tensorflow as tf

from .pytorch_dump import PyTorchDumper

device = torch.device('cpu')
window = torch.tensor(tf.signal.hann_window(255).numpy()).to(device = device)

params = [
        "dump_path"
    ]

def get_spectrogram(audio):
    waveform = torch.tensor(audio, dtype = torch.float32)
    spectrogram = torch.stft(waveform, n_fft = 256, win_length = 255, hop_length = 128, return_complex = True, window = window, onesided = True, center = False)
    spectrogram = torch.abs(spectrogram)
    spectrogram = torch.unsqueeze(spectrogram, 1).to(device = device)
    return spectrogram # shape (batch_size, 124, 129, 1)


class ARTModel(nn.Module):
    def __init__(self):
        super(ARTModel, self).__init__()

        self.resnet50 = torchvision.models.resnet50(num_classes = 12)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for parameter in self.resnet50.parameters():
            if parameter.dim() >= 2:
                nn.init.xavier_uniform_(parameter)

    def forward(self, input_audio_samples):
        spectrogram = get_spectrogram(input_audio_samples)
        predictions = self.resnet50(spectrogram)
        return predictions


class ART_Classifier(PyTorchDumper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

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

def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):
    dump_path = model_kwargs['dump_path']
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ttmp = torch.zeros([1, 1]).to(torch.device("cuda"))
        wrapper_kwargs['device_type'] = 'cuda'
    else:
        device = torch.device('cpu')
        wrapper_kwargs['device_type'] = 'cpu'
    print(device)

    model = ARTModel().to(device = device)
    optimizer = torch.optim.Adam(model.parameters(), eps = 1e-7)
    loss_function = nn.CrossEntropyLoss()

    art_classifier = ART_Classifier(
        model = model,
        loss = loss_function,
        optimizer = optimizer,
        input_shape = (16000,),
        nb_classes = 12,
        dump_path = dump_path,
        **wrapper_kwargs,
    )

    return art_classifier