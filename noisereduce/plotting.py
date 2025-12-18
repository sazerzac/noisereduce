try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
import numpy as np


def plot_spectrogram(signal, title):
    """Plot a spectrogram.
    
    Parameters
    ----------
    signal : array_like
        2D array representing the spectrogram (frequency x time)
    title : str
        Title for the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install noisereduce[plotting]"
        )
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.afmhot,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    """Plots basic statistics of noise reduction.
    
    Parameters
    ----------
    mean_freq_noise : array_like
        Mean power of noise across frequencies
    std_freq_noise : array_like
        Standard deviation of noise power across frequencies
    noise_thresh : array_like
        Noise threshold values by frequency
    smoothing_filter : array_like
        2D filter matrix used for smoothing the mask
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install noisereduce[plotting]"
        )
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    (plt_mean,) = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    (plt_std,) = ax[0].plot(std_freq_noise, label="Std. power of noise")
    (plt_std,) = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def plot_reduction_steps(
    noise_stft_db,
    mean_freq_noise,
    std_freq_noise,
    noise_thresh,
    smoothing_filter,
    sig_stft_db,
    sig_mask,
    recovered_spec,
):
    """Plot all steps of the noise reduction process.
    
    Parameters
    ----------
    noise_stft_db : array_like
        Spectrogram of the noise signal in dB
    mean_freq_noise : array_like
        Mean power of noise across frequencies
    std_freq_noise : array_like
        Standard deviation of noise power across frequencies
    noise_thresh : array_like
        Noise threshold values by frequency
    smoothing_filter : array_like
        2D filter matrix used for smoothing the mask
    sig_stft_db : array_like
        Spectrogram of the signal in dB
    sig_mask : array_like
        Mask applied to the signal spectrogram
    recovered_spec : array_like
        Recovered/denoised spectrogram
    """
    plot_spectrogram(noise_stft_db, title="Noise")
    plot_statistics_and_filter(
        mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
    )
    plot_spectrogram(sig_stft_db, title="Signal")
    plot_spectrogram(sig_mask, title="Mask applied")
    plot_spectrogram(recovered_spec, title="Recovered spectrogram")

