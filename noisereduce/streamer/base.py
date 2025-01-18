import tempfile

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from noisereduce.torchgate import TorchGate as TG


class SpectralGate:
    def __init__(
            self,
            y,
            sr,
            prop_decrease,
            chunk_size,
            padding,
            n_fft,
            win_length,
            hop_length,
            time_constant_s,
            freq_mask_smooth_hz,
            time_mask_smooth_ms,
            tmp_folder,
            use_tqdm,
            n_jobs,
    ):
        self.sr = sr
        # if this is a 1D single channel recording
        self.flat = False

        y = np.array(y)
        # reshape data to (#channels, #frames)
        if len(y.shape) == 1:
            self.y = np.expand_dims(y, 0)
            self.flat = True
        elif len(y.shape) > 2:
            raise ValueError("Waveform must be in shape (# frames, # channels)")
        else:
            self.y = y

        self._dtype = y.dtype
        # get the number of channels and frames in data
        self.n_channels, self.n_frames = self.y.shape
        self._chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs

        self.use_tqdm = use_tqdm
        # where to create a temp file for parallel
        # writing
        self._tmp_folder = tmp_folder

        ### Parameters for spectral gating
        self._n_fft = n_fft
        # set window and hop length for stft
        if win_length is None:
            self._win_length = self._n_fft
        else:
            self._win_length = win_length
        if hop_length is None:
            self._hop_length = self._win_length // 4
        else:
            self._hop_length = hop_length

        self._time_constant_s = time_constant_s

        self._prop_decrease = prop_decrease

    def _read_chunk(self, i1, i2):
        """read chunk and pad with zerros"""
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > self.n_frames:
            i2b = self.n_frames
        else:
            i2b = i2
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1: i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        """Pad and perform filtering"""
        i1 = start_frame - self.padding
        i2 = end_frame + self.padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1: end_frame - i1]

    def _get_filtered_chunk(self, ind):
        """Grabs a single chunk"""
        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        raise NotImplementedError

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        filtered_chunk0 = self._get_filtered_chunk(ich)
        filtered_chunk[:, pos: pos + end0 - start0] = filtered_chunk0[:, start0:end0]
        pos += end0 - start0

    def get_traces(self, start_frame=None, end_frame=None):
        """Grab filtered data iterating over chunks"""
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.n_frames

        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                ich1 = int(start_frame / self._chunk_size)
                ich2 = int((end_frame - 1) / self._chunk_size)

                # write output to temp memmap for parallelization
                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    # create temp file
                    filtered_chunk = np.memmap(
                        fp,
                        dtype=self._dtype,
                        shape=(self.n_channels, int(end_frame - start_frame)),
                        mode="w+",
                    )
                    pos_list = []
                    start_list = []
                    end_list = []
                    pos = 0
                    for ich in range(ich1, ich2 + 1):
                        if ich == ich1:
                            start0 = start_frame - ich * self._chunk_size
                        else:
                            start0 = 0
                        if ich == ich2:
                            end0 = end_frame - ich * self._chunk_size
                        else:
                            end0 = self._chunk_size
                        pos_list.append(pos)
                        start_list.append(start0)
                        end_list.append(end0)
                        pos += end0 - start0

                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._iterate_chunk)(
                            filtered_chunk, pos, end0, start0, ich
                        )
                        for pos, start0, end0, ich in zip(
                            tqdm(pos_list, disable=not (self.use_tqdm)),
                            start_list,
                            end_list,
                            range(ich1, ich2 + 1),
                        )
                    )
                    if self.flat:
                        return filtered_chunk.astype(self._dtype).flatten()
                    else:
                        return filtered_chunk.astype(self._dtype)

        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        if self.flat:
            return filtered_chunk.astype(self._dtype).flatten()
        else:
            return filtered_chunk.astype(self._dtype)


class StreamedTorchGate(SpectralGate):
    '''
    Run interface with noisereduce.
    '''

    def __init__(
            self,
            y,
            sr,
            stationary=False,
            y_noise=None,
            prop_decrease=1.0,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            thresh_n_mult_nonstationary=2,
            sigmoid_slope_nonstationary=10,
            n_std_thresh_stationary=1.5,
            tmp_folder=None,
            chunk_size=600000,
            padding=30000,
            n_fft=1024,
            win_length=None,
            hop_length=None,
            clip_noise_stationary=True,
            use_tqdm=False,
            n_jobs=1,
            device="cuda",
    ):
        super().__init__(
            y=y,
            sr=sr,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            tmp_folder=tmp_folder,
            prop_decrease=prop_decrease,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
        )

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # noise convert to torch if needed
        if y_noise is not None:
            if y_noise.shape[-1] > y.shape[-1] and clip_noise_stationary:
                y_noise = y_noise[: y.shape[-1]]
            y_noise = torch.from_numpy(y_noise).to(device)
            # ensure that y_noise is in shape (#channels, #frames)
            if len(y_noise.shape) == 1:
                y_noise = y_noise.unsqueeze(0)
        self.y_noise = y_noise

        # create a torch object
        self.tg = TG(
            sr=sr,
            nonstationary=not stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_thresh_nonstationary=thresh_n_mult_nonstationary,
            temp_coeff_nonstationary=1 / sigmoid_slope_nonstationary,
            n_movemean_nonstationary=int(time_constant_s / self._hop_length * sr),
            prop_decrease=prop_decrease,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
        ).to(device)

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        # convert to torch if needed
        if type(chunk) is np.ndarray:
            chunk = torch.from_numpy(chunk).to(self.device)
        chunk_filtered = self.tg(x=chunk, xn=self.y_noise)
        return chunk_filtered.cpu().detach().numpy()
