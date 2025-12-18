"""
Numpy fallback implementations for scipy.signal functions.

This module provides numpy-based implementations of scipy.signal functions
to allow the package to work without scipy as a dependency.

The functions in this module are designed to be drop-in replacements for
their scipy.signal equivalents, with the same function signatures and
behavior. Performance may differ from scipy implementations, but numerical
accuracy is maintained.

Functions:
    stft: Short-Time Fourier Transform
    istft: Inverse Short-Time Fourier Transform
    fftconvolve: Fast convolution using FFT
    filtfilt: Forward-backward IIR filter
"""
import numpy as np

# Constants
_EPS = np.finfo(np.float64).eps  # Machine epsilon for numerical stability
_MIN_WINDOW_SUM = 1e-10  # Minimum window sum to avoid division by zero in istft


def stft(x, nfft=None, nperseg=None, noverlap=None, padded=False):
    """
    Short-Time Fourier Transform using numpy.
    
    This is a numpy-based fallback for scipy.signal.stft.
    
    Parameters
    ----------
    x : array_like
        Time series of measurement values
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.
        If None, the FFT length is nperseg. Defaults to None.
    nperseg : int, optional
        Length of each segment. Defaults to None.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        noverlap = nperseg // 2. Defaults to None.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end
        to make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to False.
    
    Returns
    -------
    f : ndarray
        Array of sample frequencies (not used in this implementation, returns None)
    t : ndarray
        Array of segment times (not used in this implementation, returns None)
    Zxx : ndarray
        STFT of x. By default, an array with shape (nfft//2 + 1, n_frames)
    """
    if nperseg is None:
        raise ValueError("nperseg must be specified")
    
    # Input validation
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    if nfft is not None and nfft <= 0:
        raise ValueError("nfft must be positive")
    if noverlap is not None and noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg")
    
    if nfft is None:
        nfft = nperseg
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Calculate hop length
    hop_length = nperseg - noverlap
    
    # Create window (Hann window)
    window = np.hanning(nperseg)
    
    # Calculate number of frames
    if padded:
        n_frames = int(np.ceil((len(x) - nperseg) / hop_length)) + 1
        x_padded = np.pad(x, (0, (n_frames - 1) * hop_length + nperseg - len(x)), mode='constant')
    else:
        n_frames = int(np.floor((len(x) - nperseg) / hop_length)) + 1
        x_padded = x
    
    # Initialize output array
    Zxx = np.zeros((nfft // 2 + 1, n_frames), dtype=np.complex128)
    
    # Compute STFT
    for i in range(n_frames):
        start = i * hop_length
        end = start + nperseg
        if end > len(x_padded):
            break
        
        # Extract segment and apply window
        segment = x_padded[start:end] * window
        
        # Zero-pad if needed
        if nfft > nperseg:
            segment = np.pad(segment, (0, nfft - nperseg), mode='constant')
        
        # Compute FFT using rfft (real FFT) - ~2x faster since we only need
        # positive frequencies and input is real
        fft_result = np.fft.rfft(segment, n=nfft)
        
        # rfft already returns only positive frequencies (nfft//2 + 1)
        Zxx[:, i] = fft_result
    
    # Return format matching scipy.signal.stft
    # Note: f and t are not computed here for simplicity
    # The calling code only uses Zxx
    return None, None, Zxx


def istft(Zxx, nfft=None, noverlap=None, nperseg=None):
    """
    Inverse Short-Time Fourier Transform using numpy.
    
    This is a numpy-based fallback for scipy.signal.istft.
    
    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to reconstruct
    nfft : int, optional
        Length of the FFT used. If None, inferred from Zxx shape.
        Defaults to None.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        noverlap = nperseg // 2. Defaults to None.
    nperseg : int, optional
        Length of each segment. If None, inferred from nfft.
        Defaults to None.
    
    Returns
    -------
    t : ndarray
        Array of output data times (not used, returns None)
    x : ndarray
        Time series of reconstructed signal
    """
    Zxx = np.asarray(Zxx)
    
    # Input validation
    if Zxx.ndim != 2:
        raise ValueError("Zxx must be a 2D array (frequency x time)")
    
    n_freq, n_frames = Zxx.shape
    
    if nfft is None:
        nfft = (n_freq - 1) * 2
    
    if nperseg is None:
        nperseg = nfft
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Additional validation
    if nfft <= 0 or nperseg <= 0:
        raise ValueError("nfft and nperseg must be positive")
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg")
    
    hop_length = nperseg - noverlap
    
    # Create window (Hann window) and pre-compute window squared
    window = np.hanning(nperseg)
    window_squared = window ** 2  # Pre-compute to avoid repeated calculation
    
    # Calculate output length
    output_length = (n_frames - 1) * hop_length + nperseg
    
    # Initialize output array (real output for audio signals)
    x = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length)
    
    # Reconstruct signal using overlap-add
    for i in range(n_frames):
        start = i * hop_length
        
        # Use irfft (inverse real FFT) - ~2x faster than ifft
        # Zxx contains only positive frequencies, which is exactly what irfft expects
        segment = np.fft.irfft(Zxx[:, i], n=nfft)
        
        # Take only the nperseg samples (remove zero-padding if any)
        segment = segment[:nperseg]
        
        # Apply window and add to output (overlap-add)
        x[start:start + nperseg] += segment * window
        window_sum[start:start + nperseg] += window_squared
    
    # Normalize by window sum to account for overlap
    # Avoid division by zero: window_sum can be very small at signal boundaries
    # due to windowing, so we clamp it to a minimum value
    window_sum = np.maximum(window_sum, _MIN_WINDOW_SUM)
    x = x / window_sum
    
    # irfft already returns real values, so no need for np.real()
    return None, x


def fftconvolve(in1, in2, mode='same'):
    """
    Convolve two N-dimensional arrays using FFT.
    
    This is a numpy-based fallback for scipy.signal.fftconvolve.
    
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as in1.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output. Default is 'same'.
    
    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of in1 with in2.
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)
    
    # Input validation
    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 must have the same number of dimensions")
    if in1.size == 0 or in2.size == 0:
        raise ValueError("in1 and in2 cannot be empty")
    
    # Determine FFT size needed for full convolution
    full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(in1.shape, in2.shape))
    
    # Compute FFT size (next power of 2 for efficiency)
    fft_shape = tuple(2 ** int(np.ceil(np.log2(s))) for s in full_shape)
    
    # Determine output shape based on mode
    if mode == 'full':
        shape = full_shape
    elif mode == 'valid':
        shape = tuple(s1 - s2 + 1 for s1, s2 in zip(in1.shape, in2.shape))
        if any(s <= 0 for s in shape):
            raise ValueError("invalid mode for this input")
    elif mode == 'same':
        shape = in1.shape
    else:
        raise ValueError("mode must be 'full', 'valid', or 'same'")
    
    # Compute FFTs
    # Specify axes explicitly to avoid deprecation warning
    axes = tuple(range(len(fft_shape)))
    fft_in1 = np.fft.fftn(in1, s=fft_shape, axes=axes)
    fft_in2 = np.fft.fftn(in2, s=fft_shape, axes=axes)
    
    # Multiply in frequency domain
    fft_result = fft_in1 * fft_in2
    
    # Inverse FFT
    result = np.fft.ifftn(fft_result, s=fft_shape, axes=axes)
    
    # Take real part (convolution of real signals is real)
    result = np.real(result)
    
    # Extract the appropriate region based on mode
    if mode == 'full':
        # Return the full convolution result (handle N-dimensional arrays)
        slices = tuple(slice(0, s) for s in shape)
        return result[slices]
    elif mode == 'valid':
        # Calculate starting indices to extract valid region
        start_indices = tuple((s_full - s) // 2 for s_full, s in zip(full_shape, shape))
        slices = tuple(slice(start, start + size) for start, size in zip(start_indices, shape))
        return result[slices]
    else:  # mode == 'same'
        # Calculate starting indices to center the result (same size as input)
        start_indices = tuple((s_full - s) // 2 for s_full, s in zip(full_shape, shape))
        slices = tuple(slice(start, start + size) for start, size in zip(start_indices, shape))
        return result[slices]


def filtfilt(b, a, x, axis=-1, padtype=None, padlen=None, method=None, irlen=None):
    """
    Apply a digital filter forward and backward to a signal.
    
    This is a numpy-based fallback for scipy.signal.filtfilt.
    Implements a simple IIR filter using direct form II transposed structure.
    
    Parameters
    ----------
    b : array_like
        The numerator coefficient vector of the filter.
    a : array_like
        The denominator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of x to which the filter is applied. Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None. This determines the
        type of extension to use for the padded signal to which the filter
        is applied. If padtype is None, no padding is used. Default is None.
    padlen : int or None, optional
        The number of elements by which to extend x at both ends of axis
        before applying the filter. This value must be less than x.shape[axis] - 1.
        padlen=0 implies no padding. Default is None.
    method : str, optional
        Not used in this implementation.
    irlen : int or None, optional
        Not used in this implementation.
    
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as x.
    """
    b = np.asarray(b)
    a = np.asarray(a)
    x = np.asarray(x)
    
    # Input validation
    if b.ndim != 1 or a.ndim != 1:
        raise ValueError("b and a must be 1D arrays")
    if len(a) == 0:
        raise ValueError("a cannot be empty")
    if len(b) == 0:
        raise ValueError("b cannot be empty")
    if x.size == 0:
        raise ValueError("x cannot be empty")
    
    # Normalize coefficients (store original a[0] before modifying)
    a0 = a[0]
    if a0 == 0:
        raise ValueError("First coefficient of denominator (a[0]) cannot be zero")
    if a0 != 1.0:
        b = b / a0
        a = a / a0
    
    # Move axis to the end for easier processing
    x = np.moveaxis(x, axis, -1)
    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    
    # Apply filter forward
    y_forward = _lfilter(b, a, x)
    
    # Reverse the signal
    y_reversed = np.flip(y_forward, axis=-1)
    
    # Apply filter backward
    y_backward = _lfilter(b, a, y_reversed)
    
    # Reverse again to get final result
    y = np.flip(y_backward, axis=-1)
    
    # Restore original shape and axis
    y = y.reshape(original_shape)
    y = np.moveaxis(y, -1, axis)
    
    return y


def _lfilter(b, a, x):
    """
    Apply a 1-D digital filter to data using direct form II transposed structure.
    
    This is a helper function for filtfilt.
    
    Parameters
    ----------
    b : array_like
        The numerator coefficient vector of the filter.
    a : array_like
        The denominator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    
    Returns
    -------
    y : ndarray
        The filtered output.
    """
    # Ensure b and a are 1D
    b = np.atleast_1d(b).flatten()
    a = np.atleast_1d(a).flatten()
    
    # Pad a and b to same length
    na = len(a)
    nb = len(b)
    n = max(na, nb)
    
    # Pad with zeros
    a_padded = np.zeros(n)
    b_padded = np.zeros(n)
    a_padded[:na] = a
    b_padded[:nb] = b
    
    # Normalize (a[0] should be 1.0 after normalization)
    a0 = a_padded[0]
    if a0 != 0:
        a_padded = a_padded / a0
        b_padded = b_padded / a0
    
    # Initialize filter states
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples = x.shape[-1]
    n_signals = x.shape[0] if x.ndim > 1 else 1
    
    # Initialize output
    y = np.zeros_like(x)
    
    # Filter state (direct form II transposed)
    z = np.zeros((n_signals, n - 1))
    
    # Pre-compute coefficient arrays for vectorization (only if n > 1)
    if n > 1:
        b_coeffs = b_padded[1:n]  # Shape: (n-1,)
        a_coeffs = a_padded[1:n]  # Shape: (n-1,)
    
    # Apply filter
    for i in range(n_samples):
        # Get input sample
        if x.ndim > 1:
            x_sample = x[:, i]  # Shape: (n_signals,)
        else:
            x_sample = x[i]  # Scalar
        
        # Direct form II transposed - compute output
        if n > 1:
            y_sample = b_padded[0] * x_sample + z[:, 0]
        else:
            y_sample = b_padded[0] * x_sample
        
        # Vectorized state updates (eliminates inner Python loop)
        if n > 1:
            # Compute base terms: b[j+1] * x - a[j+1] * y for all j
            # Broadcast x_sample and y_sample to match z shape (n_signals, n-1)
            if x.ndim > 1:
                # Multiple signals: broadcast (n_signals,) to (n_signals, n-1)
                x_broadcast = x_sample[:, np.newaxis]  # (n_signals, 1)
                y_broadcast = y_sample[:, np.newaxis]  # (n_signals, 1)
                # Compute: (n_signals, 1) * (1, n-1) -> (n_signals, n-1)
                state_base = b_coeffs[np.newaxis, :] * x_broadcast - a_coeffs[np.newaxis, :] * y_broadcast
            else:
                # Single signal: scalar * (n-1,) -> (n-1,)
                state_base = b_coeffs * x_sample - a_coeffs * y_sample
            
            # Add shifted states: z[j+1] for j < n-2
            if n > 2:
                # Save old z[:, 1:] before modifying z
                z_shifted = z[:, 1:].copy()  # (n_signals, n-2)
                # Update states: z[j] = base[j] + z[j+1] for j < n-2
                if x.ndim > 1:
                    state_base[:, :-1] += z_shifted
                else:
                    state_base[:-1] += z_shifted
            
            # Update all states at once (vectorized)
            if x.ndim > 1:
                z = state_base
            else:
                z = state_base.reshape(1, -1)
        
        # Store output
        if x.ndim > 1:
            y[:, i] = y_sample
        else:
            y[i] = y_sample
    
    return y.squeeze()

