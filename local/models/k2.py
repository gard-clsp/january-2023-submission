# MIT License
#
# Copyright (C) Yiwen Shao
# [Added defenses : Sonal Joshi]
# [Added denoiser defense : Sonal Joshi [Oct21]]
"""
This module implements the task specific estimator k2
"""
import os
import logging
from typing import List, Optional, TYPE_CHECKING, Union

import numpy as np
from art.config import ART_NUMPY_DTYPE
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from pathlib import Path
import torch

from .wave_gan import WaveGANDefender  # For wavegan
from .wave_gan_white import WaveGANDefender as WaveGANDefenderWhite

from .denoiser import DenoiserDefender
from .denoiser_white import DenoiserDefender as DenoiserDefenderWhite

from armory import paths
from armory.data.utils import maybe_download_weights_from_s3

if TYPE_CHECKING:
    import torch
    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

INT16MAX = 32767


class PyTorchK2(SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using k2.
    """

    def __init__(
            self,
            config_filepath,
            clip_values: Optional["CLIP_VALUES_TYPE"] = None,
            preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
            postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
            preprocessing: "PREPROCESSING_TYPE" = None,
            device_type: str = "cpu",
            smoothing_after_wavegan: bool = True,
            wave_gan_defender: Optional[Union[WaveGANDefender, WaveGANDefenderWhite]] = None,
            denoiser_defender: Optional[Union[DenoiserDefender, DenoiserDefenderWhite]] = None,
            smooth_sigma: float = 0,
            defense_chunk_size: int = -1,
            random_split_chunk: bool = False,
    ):
        import os
        import torch  # lgtm [py/repeated-import]
        import yaml
        import k2
        from pathlib import Path
        from snowfall.models.conformer import Conformer
        from snowfall.common import average_checkpoint, load_checkpoint
        from snowfall.training.mmi_graph import create_bigram_phone_lm
        from snowfall.training.mmi_graph import get_phone_symbols
        from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
        from snowfall.lexicon import Lexicon

        #from lhotse.features.kaldi.layers import Wav2LogFilterBank
        #from lhotse.utils import compute_num_frames
        
        # Super initialization
        super().__init__(
            model=None,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.smooth_sigma = smooth_sigma
        self.wave_gan_defender = wave_gan_defender
        self.denoiser_defender = denoiser_defender
        self.smoothing_after_wavegan = smoothing_after_wavegan
        self.defense_chunk_size = defense_chunk_size
        self.random_split_chunk = random_split_chunk

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):
                raise ValueError(
                    "This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):
                raise ValueError(
                    "This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check preprocessing and postprocessing defences
        if self.preprocessing_defences is not None:
            raise ValueError(
                "This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError(
                "This estimator does not support `postprocessing_defences`.")

        # Set cpu/gpu device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))
        

        # # Save first version of the optimizer
        # self._optimizer = optimizer
        # self._use_amp = use_amp
        maybe_download_k2_configs()
        config_filepath, saved_model_dir = self.find_filepath(config_filepath)

        # construct args
        with open(config_filepath) as file:
            k2_config = yaml.load(file, Loader=yaml.FullLoader)
        self.k2_config = k2_config

        self.prepare_model_files(saved_model_dir)

        # load symbol_table
        self.symbol_table = k2.SymbolTable.from_file(self.k2_config["lang_dir"] + '/words.txt')
        # phone_symbol_table = k2.SymbolTable.from_file(self.k2_config["lang_dir"] + '/phones.txt')
        # phone_ids = get_phone_symbols(phone_symbol_table)
        # self.P = create_bigram_phone_lm(phone_ids)

        lexicon = Lexicon(Path(self.k2_config["lang_dir"]))
        self.graph_compiler = MmiTrainingGraphCompiler(
            lexicon=lexicon,
            device=self._device,
        )
        phone_ids = lexicon.phone_symbols()
        P = create_bigram_phone_lm(phone_ids)
        P.scores = torch.zeros_like(P.scores)
        self.P = P.to(self._device)
        # L_inv = k2.Fsa.from_dict(torch.load(self.k2_config["lang_dir"] + '/Linv.pt'))
        # self.graph_compiler = MmiTrainingGraphCompiler(
        #     L_inv=L_inv,
        #     phones=phone_symbol_table,
        #     words=self.symbol_table,
        #     device=self._device,
        # )
        
        # load model
        self._model = Conformer(
            num_features=self.k2_config["num_features"],
            nhead=self.k2_config["model"]["nhead"],
            d_model=self.k2_config["model"]["attention_dim"],
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=self.k2_config["subsampling_factor"],
            num_decoder_layers=self.k2_config["model"]["num_decoder_layers"],
            vgg_frontend=self.k2_config["model"]["vgg_frontend"])

        self._model.P_scores = torch.nn.Parameter(self.P.scores.clone(), requires_grad=False)

        #self._extractor = Wav2LogFilterBank().to(self._device)

        # load checkpoint
        exp_dir = self.k2_config["checkpoint"]["dir"]
        epoch = self.k2_config["checkpoint"]["epoch"]
        avg = self.k2_config["checkpoint"]["avg"]
        if avg == 1:
            checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
            load_checkpoint(checkpoint, self._model)
        else:
            checkpoints = [os.path.join(exp_dir, 'epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                           range(epoch - avg, epoch)]
            average_checkpoint(checkpoints, self._model)

        self._model.to(self._device)
            
        # load pre-compiled HLG graph
        HLG_dict = torch.load(self.k2_config['HLG'])
        self.HLG = k2.Fsa.from_dict(HLG_dict)
        self.HLG = self.HLG.to(self._device)
        self.HLG.aux_labels = k2.ragged.remove_values_eq(self.HLG.aux_labels, 0)
        self.HLG.requires_grad_(False)

    @staticmethod
    def find_filepath(filepath):
        saved_model_dir = None
        # try absolute path
        if not os.path.exists(filepath):
            filepath_0 = filepath
            # try with docker
            paths.set_mode('docker')
            saved_model_dir = paths.runtime_paths().saved_model_dir
            filepath = os.path.join(saved_model_dir, filepath_0)
            if not os.path.exists(filepath):
                filepath_1 = filepath
                # try hostmode path
                paths.set_mode('host')
                saved_model_dir = paths.runtime_paths().saved_model_dir
                filepath = os.path.join(saved_model_dir, filepath_0)
                if not os.path.exists(filepath):
                    raise FileNotFoundError('file not found in %s / %s / %s' %
                                            (filepath_0, filepath_1, filepath))

        return filepath, saved_model_dir

    def predict(self, x, batch_size=128, transcription_output: bool = True, **kwargs):
        """
        Perform prediction for a batch of inputs.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :return: Transcription as a numpy array of characters. A possible example of a transcription return
        is `np.array(['SIXTY ONE', 'HELLO'])`.
        """

        import k2
        from snowfall.common import get_texts

        self._model.eval()
        if not transcription_output:
            raise NotImplementedError()

        batch_size = 1
        if batch_size > 1:
            raise NotImplementedError()
 
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # list of ndarray to list of tensor
        x_preprocessed = [torch.from_numpy(x_preprocessed[i]) for i in range(len(x_preprocessed))]

        # Run prediction with batch processing
        results = []
        decoded_output = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, len(x_preprocessed)),
            )
            feature, supervisions, indices = self.transform_model_input(x=x_preprocessed[begin:end])

            supervision_segments = torch.stack(
                (supervisions['sequence_idx'],
                 (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
                 (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
            
            if self.random_split_chunk:
                chunk_supervisions, n_chunks = self.split_chunks_random(supervisions,
                                                                        self.defense_chunk_size,
                                                                        self.defense_chunk_size,
                                                                        dist='beta')
            else:
                chunk_supervisions, n_chunks = self.split_chunks(supervisions,
                                                                 self.defense_chunk_size,
                                                                 self.defense_chunk_size)
            with torch.no_grad():
                feature = feature.to(self._device)
                chunk_outputs = []
                for c in range(n_chunks):
                    chunk_nnet_output, _, _ = self._model(feature, chunk_supervisions[c])
                    chunk_outputs.append(chunk_nnet_output)
                # if shift_size >= chunk_size: # no overlap
                nnet_output = torch.cat(chunk_outputs, dim=2)
            # nnet_output is [N, C, T]
            nnet_output = nnet_output.permute(0, 2,
                                              1)  # now nnet_output is [N, T, C]

            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

            assert self.HLG.device == nnet_output.device, \
                f"Check failed: LG.device ({self.HLG.device}) == nnet_output.device ({nnet_output.device})"

            lattices = k2.intersect_dense_pruned(self.HLG, dense_fsa_vec, 20.0, 7.0, 30,
                                                 10000)
            best_paths = k2.shortest_path(lattices, use_double_scores=True)
            hyps = get_texts(best_paths, indices)
            for i in range(batch_size):
                hyp_words = [self.symbol_table.get(x) for x in hyps[i]]
                hyp_sentence = " ".join(hyp_words)
                decoded_output.append(hyp_sentence)

            return decoded_output

    def split_chunks(self, supervisions, chunk_size=0, shift_size=0, start_shift=0):
        # only works for batch_size = 1. Will make changes if necessary
        total_frames = supervisions['num_frames'][0] - start_shift
        if chunk_size == 0 or chunk_size > total_frames:
            # no split
            n_chunks = 1
            return [supervisions], n_chunks
        n_chunks = (supervisions['num_frames'] - chunk_size - start_shift) // shift_size + 2
        #chunk_supervision_segments = []
        chunk_supervisions = []
        for i in range(n_chunks):
            start_frame = torch.IntTensor([i * shift_size + start_shift])
            num_frames = total_frames - start_frame if start_frame + chunk_size > total_frames else chunk_size
            num_frames = torch.IntTensor([num_frames])
            # supervision_segments = torch.stack(
            #     (supervisions['sequence_idx'],
            #      (((start_frame - 1) // 2 - 1) // 2),
            #      (((num_frames - 1) // 2 - 1) // 2)), 1).to(torch.int32)
            # chunk_supervision_segments.append(supervision_segments)
            chunk_supervision = {}
            chunk_supervision['sequence_idx'] = supervisions['sequence_idx']
            chunk_supervision['start_frame'] = start_frame
            chunk_supervision['num_frames'] = num_frames
            chunk_supervisions.append(chunk_supervision)

        return chunk_supervisions, n_chunks
        
    def split_chunks_random(self, supervisions, chunk_size=0, shift_size=0, start_shift=0, dist='unif'):
        # only works for batch_size = 1. Will make changes if necessary
        total_frames = supervisions['num_frames'][0]
        chunk_supervisions = []
        n_chunks = 0
        cur_frame = start_shift
        if dist == 'beta':
            from torch.distributions.beta import Beta
            m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        while True:
            start_frame = cur_frame
            if dist == 'unif':
                num_frames = torch.randint(200, chunk_size, (1,)).item()
            elif dist == 'beta':
                s = m.sample()
                num_frames = int(s * (chunk_size - 200) + 200)
                num_frames = torch.IntTensor([num_frames])
            if start_frame + num_frames > total_frames:
                num_frames = total_frames - start_frame
            cur_frame += num_frames
            chunk_supervision = {}
            chunk_supervision['sequence_idx'] = supervisions['sequence_idx']
            chunk_supervision['start_frame'] = torch.IntTensor([start_frame])
            chunk_supervision['num_frames'] = torch.IntTensor([num_frames])
            chunk_supervisions.append(chunk_supervision)
            n_chunks += 1
            if cur_frame == total_frames:
                break
        return chunk_supervisions, n_chunks

        
    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """

        import k2
        import math

        def get_tot_objf_and_num_frames(tot_scores: torch.Tensor,
                                        frames_per_seq: torch.Tensor
                                        ):
            ''' Figures out the total score(log-prob) over all successful supervision segments
            (i.e. those for which the total score wasn't -infinity), and the corresponding
            number of frames of neural net output
                 Args:
                    tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                               from forward-backward
                frames_per_seq: a Torch tensor of shape (num_segments,) containing the number of
                               frames for each segment
                Returns:
                     Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames)
                where ok_frames is the frames for successful (finite) segments, and
               all_frames is the frames for all segments (finite or not).
            '''
            mask = torch.ne(tot_scores, -math.inf)
            # finite_indexes is a tensor containing successful segment indexes, e.g.
            # [ 0 1 3 4 5 ]
            finite_indexes = torch.nonzero(mask).squeeze(1)
            if False:
                bad_indexes = torch.nonzero(~mask).squeeze(1)
                if bad_indexes.shape[0] > 0:
                    print("Bad indexes: ", bad_indexes, ", bad lengths: ",
                          frames_per_seq[bad_indexes], " vs. max length ",
                          torch.max(frames_per_seq), ", avg ",
                          (torch.sum(frames_per_seq) / frames_per_seq.numel()))
            # print("finite_indexes = ", finite_indexes, ", tot_scores = ", tot_scores)
            ok_frames = frames_per_seq[finite_indexes].sum()
            all_frames = frames_per_seq.sum()
            return (tot_scores[finite_indexes].sum(), ok_frames, all_frames)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x, y, fit=False)

        # list of ndarray to list of tensor
        x_preprocessed = [torch.from_numpy(x_preprocessed[i]) for i in range(len(x_preprocessed))]
        # Transform data into the model input space
        feature, supervisions, indices = self.transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=True)

        supervision_segments = torch.stack(
            (supervisions['sequence_idx'],
             (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
             (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
        supervision_segments = torch.clamp(supervision_segments, min=0)
        texts = supervisions['text']
        
        feature = feature.to(self._device)
        nnet_output, encoder_memory, memory_mask = self._model(feature, supervisions)

        nnet_output = nnet_output.permute(0, 2, 1) # [N, T, C]

        with torch.no_grad():
            num, den = self.graph_compiler.compile(texts, self.P.to(self._device))
        
        num = num.to(self._device)
        den = den.to(self._device)

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

        num = k2.intersect_dense(num, dense_fsa_vec, 10.0)
        den = k2.intersect_dense(den, dense_fsa_vec, 10.0)

        num_tot_scores = num.get_tot_scores(
            log_semiring=True,
            use_double_scores=True)
        den_tot_scores = den.get_tot_scores(
            log_semiring=True,
            use_double_scores=True)
        tot_scores = num_tot_scores - self.k2_config["den_scale"] * den_tot_scores

        (tot_score, tot_frames,
         all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                                   supervision_segments[:, 2])

        loss = (-tot_score) / len(texts)

        loss.backward()

        # Get results
        results = []
        for i in range(len(x_preprocessed)):
            results.append(x_preprocessed[i].grad.cpu().numpy().copy())

        results = [results[idx] for idx in indices]
        results = np.array(results)
        results = self._apply_preprocessing_gradient(x, results)

        return results

    def transform_model_input(
            self,
            x,
            y=None,
            compute_gradient=False,
            tensor_input=False,
            real_lengths=None
    ):
        """
        Transform the user input space into the model input space.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :param tensor_input: Indicate whether input is tensor.
        :param real_lengths: Real lengths of original sequences.
        :return: A tupe of a sorted input feature tensor, a supervision tensor,  and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        import torchaudio

        from dataclasses import dataclass, asdict
        @dataclass
        class FbankConfig:
            # Spectogram-related part
            dither: float = 0.0
            window_type: str = "povey"
            # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
            frame_length: float = 0.025
            frame_shift: float = 0.01
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
        params.update({
            "sample_frequency": self.k2_config["sampling_rate"],
            "snip_edges": False,
            "num_mel_bins": self.k2_config["num_features"]
        })
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        

        feature_list = []
        num_frames = []
        supervisions = {}

        for i in range(len(x)):
            #x[i] = x[i].astype(ART_NUMPY_DTYPE)

            # Jesus fix for error: set NaN to 0
            isnan = torch.isnan(x[i])
            nisnan=torch.sum(isnan).item()
            if nisnan > 0:
                logging.info('input isnan={}/{} {}'.format(nisnan, x[i].shape, x[i][isnan], torch.max(torch.abs(x[i]))))
                #logging.info('last_x={}'.format(self.last_x[isnan]))
                #logging.info('last_xwgan={}'.format(self.last_xwgan[isnan]))

            x[i][isnan] = 0

            if compute_gradient:
                x[i].requires_grad = True

            # ToDo : Write smarter way to order the defenses

            if self.wave_gan_defender and self.denoiser_defender: # Check if wavegan and denoiser is both enables
                raise NotImplementedError()

            if self.wave_gan_defender is not None: # Check if wavegan defense is to be used
                if self.smoothing_after_wavegan:   # If smoothing_after_wavegan is true, pass through wavegan first, then do smoothing
                    xx = self.wave_gan_defender(x[i])
                    if self.smooth_sigma > 0:      # If smooth_sigma > 0, i.e. smoothing is to be used, do smoothing pass through wavegan
                        xx = xx + self.smooth_sigma * torch.randn_like(xx) 
                else: # If smoothing_after_wavegan is false, do smoothing first, then pass through wavegan
                    if self.smooth_sigma > 0:
                        xx = x[i] + self.smooth_sigma * torch.randn_like(x[i])
                    xx = self.wave_gan_defender(xx)
            elif self.wave_gan_defender is None: # No wavegan defense is to be used, just do smoothing if  self.smooth_sigma > 0
                if self.smooth_sigma > 0:      # If smooth_sigma > 0, i.e. smoothing is to be used, do smoothing
                    xx = x[i] + self.smooth_sigma * torch.randn_like(x[i])
                else:
                    xx = x[i]
            else:
                logging.info('Error in defense calls for smoothing and WaveGAN')
                raise


            if self.denoiser_defender is not None: # Check if denoiser defense is to be used
                # Right now always do denoiser before smoothing
                # To Do - Add before or after denoising option
                xx = self.denoiser_defender(x[i])
                if self.smooth_sigma > 0:      # If smooth_sigma > 0, i.e. smoothing is to be used, do smoothing pass through denoiser
                    xx = xx + self.smooth_sigma * torch.randn_like(xx) 
            elif self.denoiser_defender is None: # No denoiser defense is to be used, just do smoothing if  self.smooth_sigma > 0
                if self.smooth_sigma > 0:      # If smooth_sigma > 0, i.e. smoothing is to be used, do smoothing
                    xx = x[i] + self.smooth_sigma * torch.randn_like(x[i])
                else:
                    xx = x[i]
            else:
                logging.info('Error in defense calls for smoothing and Denoiser')
                raise

            xx = xx.to(self._device)
            feat_i = torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **params) # [T, C]
            feat_i = feat_i.transpose(0, 1) #[C, T]
            feature_list.append(feat_i)
            num_frames.append(feat_i.shape[1])
        
        indices = sorted(range(len(feature_list)),
                         key=lambda i: feature_list[i].shape[1], reverse=True)
        indices = torch.LongTensor(indices)
        num_frames = torch.IntTensor([num_frames[idx] for idx in indices])
        start_frames = torch.zeros(len(x), dtype=torch.int)

        supervisions['sequence_idx'] = indices.int()
        supervisions['start_frame'] = start_frames
        supervisions['num_frames'] = num_frames
        if y is not None:
            supervisions['text'] = [y[idx] for idx in indices]

        feature_sorted = [feature_list[index] for index in indices]
        
        feature = torch.zeros(len(feature_sorted), feature_sorted[0].size(0), feature_sorted[0].size(1), device=self._device)

        for i in range(len(x)):
            feature[i, :, :feature_sorted[i].size(1)] = feature_sorted[i]

        # feature = self.extractor(xx)
        # if supervisions is not None:
        #     start_frames = [ compute_num_frames(sample.item() / 16000, extractor.frame_shift, 16000)
        #                      for sample in supervisions['start_sample']
        #     ]
        #     supervisions['start_frame'] = torch.LongTensor(start_frames)
            
        #     num_frames = [
        #         compute_num_frames(sample.item() / 16000, extractor.frame_shift, 16000)
        #         for sample in supervisions['num_samples']
        #     ]
        #     supervisions['num_frames'] = torch.LongTensor(num_frames)

        return feature, supervisions, indices
        

    @property
    def input_shape(self):
        """
        Return the shape of one input sample.
        :return: Shape of one input sample.
        """
        self._input_shape = None
        return self._input_shape  # type: ignore

    @property
    def compute_loss(self):
        return None
    
    @property
    def model(self):
        """
        Get current model.
        :return: Current model.
        """
        return self._model

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.
        :return: Current used device.
        """
        return self._device

    def get_activations(
            self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        raise NotImplementedError


    def prepare_model_files(self, saved_model_dir):
        """Resolves the absoute path to the model files 
           and decompress the models if necessary
        """
        if saved_model_dir is None:
            return

        self.k2_config['lang_dir'] = os.path.join(saved_model_dir,
                                                  self.k2_config['lang_dir'])
        self.k2_config['HLG'] = os.path.join(saved_model_dir,
                                             self.k2_config['HLG'])
        self.k2_config["checkpoint"]["dir"] = os.path.join(
            saved_model_dir, self.k2_config["checkpoint"]["dir"])

        import tarfile
        if not os.path.isdir(self.k2_config['lang_dir']):
            tar_file = self.k2_config['lang_dir'] + '.tar.gz'
            maybe_download_weights_from_s3(tar_file)
            f = tarfile.open(tar_file)
            f.extractall(saved_model_dir)
            f.close()

        if not os.path.isdir(self.k2_config['checkpoint']['dir']):
            tar_file = self.k2_config['checkpoint']['dir'] + '.tar.gz'
            maybe_download_weights_from_s3(tar_file)
            f = tarfile.open(tar_file)
            f.extractall(saved_model_dir)
            f.close()



def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    # Added to avoid CUDA error on GRID , Sonal 06Nov20
    if torch.cuda.is_available():
        ttmp = torch.zeros([1, 1]).to(torch.device("cuda"))
        wrapper_kwargs['device_type'] = 'cuda'
    else:
        wrapper_kwargs['device_type'] = 'cpu'

    # WaveGAN calls
    for key in ('wave_gan_root_dir', 'wave_gan_model_ckpt', 'wave_gan_white'):
        logging.info(f'WaveGAN parameter "{key}": {wrapper_kwargs[key]}')
    use_wavegan = wrapper_kwargs['wave_gan_model_ckpt'] is not None and wrapper_kwargs['wave_gan_model_ckpt'] != ""
    whitebox_wavegan = bool(wrapper_kwargs['wave_gan_white'])
    wave_gan_defender = None
    if use_wavegan:
        wrapper_kwargs['wave_gan_root_dir'] = maybe_download_denoiser(wrapper_kwargs['wave_gan_root_dir'])
        if whitebox_wavegan:
            wave_gan_defender = WaveGANDefenderWhite(
                Path(wrapper_kwargs['wave_gan_root_dir']), 
                Path(wrapper_kwargs['wave_gan_model_ckpt'])
            )
            logging.info("WHITEBOX WaveGAN is ready.")
        else:
            wave_gan_defender = WaveGANDefender(
                Path(wrapper_kwargs['wave_gan_root_dir']), 
                Path(wrapper_kwargs['wave_gan_model_ckpt']), 
                device=wrapper_kwargs['device_type']
            )
            logging.info("BLACKBOX WaveGAN is ready.")
    del wrapper_kwargs['wave_gan_white']
    del wrapper_kwargs['wave_gan_root_dir']
    del wrapper_kwargs['wave_gan_model_ckpt']

    # Denoiser calls
    for key in ('denoiser_root_dir', 'denoiser_model_ckpt', 'denoiser_white'):
        logging.info(f'Denoiser parameter "{key}": {wrapper_kwargs[key]}')
    logging.info(f"Device Type {wrapper_kwargs['device_type']}")

    use_denoiser = wrapper_kwargs['denoiser_model_ckpt'] is not None and wrapper_kwargs['denoiser_model_ckpt'] != ""
    whitebox_denoiser = bool(wrapper_kwargs['denoiser_white'])
    denoiser_defender = None
    if use_denoiser:
        wrapper_kwargs['denoiser_root_dir'] = maybe_download_denoiser(wrapper_kwargs['denoiser_root_dir'])
        print(wrapper_kwargs['denoiser_root_dir'])
        if whitebox_denoiser:
            denoiser_defender = DenoiserDefenderWhite(
                Path(wrapper_kwargs['denoiser_root_dir']), 
                Path(wrapper_kwargs['denoiser_model_ckpt']), 
                device=wrapper_kwargs['device_type']
            )
            print(wrapper_kwargs['device_type'])
            logging.info("WHITEBOX Denoiser is ready.")
        else:
            denoiser_defender = DenoiserDefender(
                Path(wrapper_kwargs['denoiser_root_dir']), 
                Path(wrapper_kwargs['denoiser_model_ckpt']), 
                device=wrapper_kwargs['device_type']
            )
            logging.info("BLACKBOX Denoiser is ready.")
    del wrapper_kwargs['denoiser_white']
    del wrapper_kwargs['denoiser_root_dir']
    del wrapper_kwargs['denoiser_model_ckpt']

    return PyTorchK2(wave_gan_defender=wave_gan_defender, denoiser_defender=denoiser_defender,**wrapper_kwargs)


def maybe_download_denoiser(root_dir):
    if os.path.exists(root_dir):
        return root_dir

    tar_file = root_dir + '.tar.gz'
    maybe_download_weights_from_s3(tar_file)

    saved_model_dir = paths.runtime_paths().saved_model_dir
    root_dir = os.path.join(saved_model_dir, root_dir)
    if os.path.isdir(root_dir):
        return root_dir
    
    tar_file = os.path.join(saved_model_dir, tar_file)
    if not os.path.exists(tar_file):
        raise FileNotFoundError('file not found in %s ' % filepath)

    import tarfile
    f = tarfile.open(tar_file)
    f.extractall(saved_model_dir)
    f.close()

    return root_dir

def maybe_download_wavegan():
    for f in [
            'JHUM_wavegan_libri_t1.4_config.yml',
            'JHUM_wavegan_libri_t1.4_stats.h5', 'JHUM_wavegan_libri_t1.4.pkl'
    ]:
        maybe_download_weights_from_s3(f)


def maybe_download_k2_configs():
    for f in [
            'JHUM_snowfall-conformer-noam-mmi-att-musan-sa-vgg-epoch20-avg5.yaml'
    ]:
        maybe_download_weights_from_s3(f)
