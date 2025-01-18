import tensorflow as tf

from .utils import amp_to_db, linspace, temperature_sigmoid


class TFGate(tf.keras.Model):
    """
    A TensorFlow module that implements the noisereduce algorithm.

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
        win_length (int | None Window length for STFT (default: None).
        hop_length (int | None Hop length for STFT (default: None).
        freq_mask_smooth_hz (float): Frequency smoothing width for the mask in Hz (default: 500).
        time_mask_smooth_ms (float): Time smoothing width for the mask in ms (default: 50).
    """

    def __init__(
            self,
            sr: int,
            nonstationary: bool = False,
            n_std_thresh_stationary: float = 1.5,
            n_thresh_nonstationary: bool = 1.3,
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

    def _generate_mask_smoothing_filter(self) -> tf.Tensor | None:
        """
        Generates a smoothing filter for the mask.

        Returns:
            tf.Tensor: A 2D tensor representing the smoothing filter.
        """
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None

        n_grad_freq = (
            1
            if self.freq_mask_smooth_hz is None
            else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2)))
        )
        if n_grad_freq < 1:
            raise ValueError(
                f"freq_mask_smooth_hz needs to be at least {int(self.sr / (self.n_fft / 2))} Hz"
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

        v_f = tf.concat(
            [
                linspace(0.0, 1.0, n_grad_freq + 1, endpoint=False),
                linspace(1.0, 0.0, n_grad_freq + 2),
            ],
            axis=0,
        )[1:-1]
        v_t = tf.concat(
            [
                linspace(0.0, 1.0, n_grad_time + 1, endpoint=False),
                linspace(1.0, 0.0, n_grad_time + 2),
            ],
            axis=0,
        )[1:-1]
        smoothing_filter = tf.einsum("i,j->ij", v_t, v_f)[tf.newaxis, tf.newaxis]
        smoothing_filter /= tf.reduce_sum(smoothing_filter)
        smoothing_filter = tf.transpose(smoothing_filter, perm=[2, 3, 0, 1])
        return smoothing_filter

    def _stationary_mask(self, xf_db: tf.Tensor, xn: tf.Tensor | None = None) -> tf.Tensor:
        """
        Computes a stationary binary mask.

        This method computes a stationary binary mask by comparing the given spectrogram (or an optional audio signal)
        to a threshold derived from the mean and standard deviation along the frequency axis.

        Arguments:
            xf_db (tf.Tensor): 2D array of shape (frames, freq_bins) representing the log-amplitude spectrogram of the signal.
            xn (tf.Tensor | None): 1D array containing the time-domain audio signal corresponding to `xf_db`.
                                       If provided, this is used to compute the spectrogram for noise estimation.

        Returns:
            tf.Tensor: A binary mask of shape (frames, freq_bins), where entries are set to 1 if the corresponding
                        spectrogram value exceeds the computed noise threshold, and 0 otherwise.
        """
        if xn is not None:
            XN = tf.signal.stft(
                xn,
                frame_length=self.n_fft,
                frame_step=self.hop_length,
                pad_end=True,
                fft_length=self.n_fft,
                window_fn=tf.signal.hann_window,
            )
            XN_db = amp_to_db(XN)
        else:
            XN_db = xf_db

        std_freq_noise, mean_freq_noise = tf.math.reduce_std(
            XN_db, axis=1, keepdims=True
        ), tf.math.reduce_mean(XN_db, axis=1, keepdims=True)

        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary
        sig_mask = tf.math.greater(xf_db, noise_thresh)

        return sig_mask

    def _nonstationary_mask(self, xf_abs: tf.Tensor) -> tf.Tensor:
        """
        Computes a non-stationary binary mask.

        This method computes a non-stationary binary mask by applying a moving average smoothing filter to the
        magnitude spectrogram and computing the slowness ratio between the original and smoothed spectrogram.

        Arguments:
            xf_abs (tf.Tensor): 2D array of shape (frames, freq_bins) containing the magnitude spectrogram of the signal.

        Returns:
            tf.Tensor: A binary mask of shape (frames, freq_bins), where entries are set to 1 if the corresponding
                        region is identified as non-stationary (rapid changes), and 0 otherwise.
        """
        X_smoothed = (
                tf.nn.conv1d(
                    xf_abs,
                    filters=tf.ones(
                        shape=(self.n_movemean_nonstationary, 1, xf_abs.shape[-1]),
                        dtype=xf_abs.dtype,
                    ),
                    stride=1,
                    padding="SAME",
                )
                / self.n_movemean_nonstationary
        )

        slowness_ratio = (xf_abs - X_smoothed) / X_smoothed
        sig_mask = temperature_sigmoid(
            slowness_ratio, self.n_thresh_nonstationary, self.temp_coeff_nonstationary
        )

        return sig_mask

    def _process(self, x: tf.Tensor, xn: tf.Tensor | None) -> tf.Tensor:
        """
        Core logic for processing.

        Arguments:
            x (tf.Tensor): The input audio signal, with shape (channels, signal_length).
            xn (tf.Tensor | None) The noise signal used for stationary noise reduction. If `None`, the input
                                         signal is used as the noise signal. Default: `None`.

        Returns:
            tf.Tensor: The denoised signal.
        """
        X = tf.signal.stft(
            x,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            pad_end=True,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )

        if self.nonstationary:
            sig_mask = self._nonstationary_mask(tf.abs(X))
        else:
            sig_mask = tf.where(self._stationary_mask(amp_to_db(X), xn), 1.0, 0.0)

        sig_mask = self.prop_decrease * (sig_mask - 1.0) + 1.0
        sig_mask = sig_mask[..., tf.newaxis]

        if self.smoothing_filter is not None:
            sig_mask = tf.nn.conv2d(
                sig_mask,
                self.smoothing_filter,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )

        sig_mask = tf.complex(sig_mask, 0.0)[..., 0]
        Y = X * sig_mask

        y = tf.signal.inverse_stft(
            Y,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
        )

        return y

    def __call__(self, x: tf.Tensor, xn: tf.Tensor | None = None) -> tf.Tensor:
        """
        Apply noise reduction to the input signal.

        Arguments:
            x (tf.Tensor): The input audio signal, with shape (channels, signal_length).
            xn (tf.Tensor | None): The noise signal used for stationary noise reduction. If `None`, the input
                                         signal is used as the noise signal. Default: `None`.

        Returns:
            tf.Tensor: The denoised signal.
        """
        assert x.shape.ndims == 2
        if x.shape[-1] < self.win_length * 2:
            raise Exception(f"x must be bigger than {self.win_length * 2}")

        assert xn is None or xn.shape.ndims == 1 or xn.shape.ndims == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2:
            raise Exception(f"xn must be bigger than {self.win_length * 2}")

        return tf.stop_gradient(self._process(x, xn))
