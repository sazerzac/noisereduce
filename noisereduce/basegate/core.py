import numpy as np
from scipy.signal import fftconvolve, stft, istft

from .utils import temperature_sigmoid, amp_to_db, std_mean


class BaseGate:
    """
    A class that implements the noisereduce algorithm using NumPy and SciPy.

    This method performs noise reduction by computing the short-time Fourier transform (STFT) of the input signal
    and applying a signal mask based on either stationary or non-stationary assumptions.

    Arguments:
        sr (int): Sample rate of the input signal.
        nonstationary (bool): Whether to use non-stationary or stationary masking (default: False).
        n_std_thresh_stationary (float): Threshold for stationary noise reduction (default: 1.5).
        n_thresh_nonstationary (float): Threshold for non-stationary noise reduction (default: 1.3).
        temp_coeff_nonstationary (float): Temperature coefficient for non-stationary masking (default: 0.1).
        n_movemean_nonstationary (int): Window size for moving average in non-stationary masking (default: 20).
        prop_decrease (float): Proportion to decrease signal where the mask is zero (default: 1.0).
        n_fft (int): FFT size for short-time Fourier transform (STFT) (default: 1024).
        win_length (int | None): Window length for STFT (default: None).
        hop_length (int | None): Hop length for STFT (default: None).
        freq_mask_smooth_hz (float): Frequency smoothing width for the mask in Hz (default: 500).
        time_mask_smooth_ms (float): Time smoothing width for the mask in ms (default: 50).
    """

    def __init__(
            self,
            sr: int,
            nonstationary: bool = False,
            n_std_thresh_stationary: float = 1.5,
            n_thresh_nonstationary: float = 1.3,
            temp_coeff_nonstationary: float = 0.1,
            n_movemean_nonstationary: int = 20,
            prop_decrease: float = 1.0,
            n_fft: int = 1024,
            win_length: int | None = None,
            hop_length: int | None = None,
            freq_mask_smooth_hz: float = 500,
            time_mask_smooth_ms: float = 50,
    ):
        super().__init__()

        # General Params
        self.sr = sr
        self.nonstationary = nonstationary
        assert 0.0 <= prop_decrease <= 1.0
        self.prop_decrease = prop_decrease

        # STFT Params
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else self.n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4

        # Stationary Params
        self.n_std_thresh_stationary = n_std_thresh_stationary

        # Non-Stationary Params
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary

        # Smooth Mask Params
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.smoothing_filter = self._generate_mask_smoothing_filter()

    def _generate_mask_smoothing_filter(self) -> np.ndarray | None:
        """
        Generates a smoothing filter for the mask.

        Returns:
            np.ndarray: A 2D tensor representing the smoothing filter.
        """
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None

        n_grad_freq = (
            1
            if self.freq_mask_smooth_hz is None
            else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2)))
        )

        # TODO: remove this if
        if n_grad_freq < 1:
            raise ValueError(
                f"freq_mask_smooth_hz needs to be at least {int((self.sr / (self.n_fft / 2)))} Hz"
            )

        n_grad_time = (
            1
            if self.time_mask_smooth_ms is None
            else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000))
        )
        if n_grad_time < 1:
            raise ValueError(
                f"time_mask_smooth_ms needs to be at least {int((self.hop_length / self.sr) * 1000)} ms"
            )

        if n_grad_time == 1 and n_grad_freq == 1:
            return None

        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        return smoothing_filter

    def _stationary_mask(
            self, xf_db: np.ndarray, xn: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Computes a stationary binary mask.

        This method computes a stationary binary mask by comparing the given spectrogram (or an optional audio signal)
        to a threshold derived from the mean and standard deviation along the frequency axis.

        Arguments:
            xf_db (np.ndarray): 2D array of shape (frames, freq_bins) representing the log-amplitude spectrogram of the signal.
            xn (np.ndarray | None): 1D array containing the time-domain audio signal corresponding to `xf_db`.
                                       If provided, this is used to compute the spectrogram for noise estimation.

        Returns:
            np.ndarray: A binary mask of shape (frames, freq_bins), where entries are set to 1 if the corresponding
                        spectrogram value exceeds the computed noise threshold, and 0 otherwise.
        """
        if xn is not None:
            XN = stft(
                xn,
                nfft=self.n_fft,
                noverlap=self.win_length - self.hop_length,
                nperseg=self.win_length,
                padded=False
            )[2]
            XN_db = amp_to_db(XN)
        else:
            XN_db = xf_db

        # calculate mean and standard deviation along the frequency axis
        std_freq_noise, mean_freq_noise = std_mean(XN_db, axis=-1)

        # compute noise threshold
        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary

        # create binary mask by thresholding the spectrogram
        sig_mask = xf_db > np.expand_dims(noise_thresh, axis=2)
        return sig_mask

    def _nonstationary_mask(self, xf_abs: np.ndarray) -> np.ndarray:
        """
        Computes a non-stationary binary mask.

        This method computes a non-stationary binary mask by applying a moving average smoothing filter to the
        magnitude spectrogram and computing the slowness ratio between the original and smoothed spectrogram.

        Arguments:
            xf_abs (np.ndarray): 2D array of shape (frames, freq_bins) containing the magnitude spectrogram of the signal.

        Returns:
            np.ndarray: A binary mask of shape (frames, freq_bins), where entries are set to 1 if the corresponding
                        region is identified as non-stationary (rapid changes), and 0 otherwise.
        """
        kernel = np.ones(self.n_movemean_nonstationary, dtype=xf_abs.dtype) / self.n_movemean_nonstationary
        X_smoothed = np.array([
            fftconvolve(xf_abs, kernel, mode='same')
        ])

        # Compute slowness ratio and apply temperature sigmoid
        slowness_ratio = (xf_abs - X_smoothed) / X_smoothed
        sig_mask = temperature_sigmoid(
            slowness_ratio, self.n_thresh_nonstationary, self.temp_coeff_nonstationary
        )

        return sig_mask

    def __call__(
            self, x: np.ndarray, xn: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply noise reduction to the input signal.

        Arguments:
            x (np.ndarray): The input audio signal, with shape (channels, signal_length).
            xn (np.ndarray | None): The noise signal used for stationary noise reduction. If `None`, the input
                                         signal is used as the noise signal. Default: `None`.

        Returns:
            np.ndarray: The denoised signal.
        """
        assert x.ndim == 2
        if x.shape[-1] < self.win_length * 2:
            raise Exception(f"x must be bigger than {self.win_length * 2}")

        assert xn is None or xn.ndim == 1 or xn.ndim == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2:
            raise Exception(f"xn must be bigger than {self.win_length * 2}")

        # Compute short-time Fourier transform (STFT)
        X = stft(
            x,
            nfft=self.n_fft,
            noverlap=self.win_length - self.hop_length,
            nperseg=self.win_length,
            padded=False
        )[2]

        # Compute signal mask based on stationary or nonstationary assumptions
        if self.nonstationary:
            sig_mask = self._nonstationary_mask(np.abs(X))
        else:
            sig_mask = self._stationary_mask(amp_to_db(X), xn)

        # Propagate decrease in signal power
        sig_mask = self.prop_decrease * (sig_mask * 1.0 - 1.0) + 1.0

        # Smooth signal mask with 2D convolution
        if self.smoothing_filter is not None:
            sig_mask = fftconvolve(sig_mask, self.smoothing_filter, mode="same")

        # Apply signal mask to STFT magnitude and phase components
        Y = X * sig_mask

        # Inverse STFT to obtain time-domain signal
        y = istft(
            Y,
            nfft=self.n_fft,
            noverlap=self.win_length - self.hop_length,
            nperseg=self.win_length
        )[1]

        return y
