# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the classifier `PyTorchClassifier` for PyTorch models.
"""
# pylint: disable=C0302,R0904
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import sys, os

from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412, C0302
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDumper(PyTorchClassifier):
    """
    This class implements a classifier with the PyTorch framework.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: Optional[Union[float, str]] = "dynamic",
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        dump_path: str = None,
    ) -> None:

        super().__init__(
            model,
            loss,
            input_shape,
            nb_classes,
            optimizer,  # type: ignore
            use_amp,
            opt_level,
            loss_scale,
            channels_first,
            clip_values,
            preprocessing_defences,
            postprocessing_defences,
            preprocessing,
            device_type,
        )

        self.dump_path = dump_path
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        

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
        Dump the data
        """
        import torch
        import pickle

        # y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        with open(self.dump_path + '/x_poison_train', 'wb+') as x_poison_file:
            pickle.dump(x, x_poison_file)
        with open(self.dump_path + '/y_poison_train', 'wb+') as y_poison_file:
            pickle.dump(y, y_poison_file)

        sys.exit()
