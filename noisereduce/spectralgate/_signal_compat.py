"""
Compatibility layer for scipy.signal functions.

This module provides scipy.signal functions when available,
or falls back to numpy-based implementations when scipy is not installed.
"""
try:
    from scipy.signal import stft, istft, fftconvolve, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    from ._scipy_fallbacks import stft, istft, fftconvolve, filtfilt

__all__ = ['stft', 'istft', 'fftconvolve', 'filtfilt', 'SCIPY_AVAILABLE']

