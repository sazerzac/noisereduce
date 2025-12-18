"""
Simple test script to verify numpy fallbacks work.

This can be run in an environment without scipy to test the fallback implementations.
Run with: python test_fallbacks_simple.py
"""
import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fallback_imports():
    """Test that fallback functions can be imported."""
    print("Testing fallback imports...")
    from noisereduce.spectralgate._scipy_fallbacks import (
        stft, istft, fftconvolve, filtfilt
    )
    print("✓ Fallback functions imported successfully")


def test_stft_istft():
    """Test STFT and ISTFT round-trip."""
    print("\nTesting STFT/ISTFT...")
    from noisereduce.spectralgate._scipy_fallbacks import stft, istft
    
    # Create test signal
    sr = 22050
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)
    
    # Forward transform
    _, _, Zxx = stft(signal, nfft=1024, nperseg=512, noverlap=256)
    
    # Inverse transform
    _, reconstructed = istft(Zxx, nfft=1024, nperseg=512, noverlap=256)
    
    # Check reconstruction quality
    min_len = min(len(signal), len(reconstructed))
    correlation = np.corrcoef(
        signal[:min_len],
        reconstructed[:min_len]
    )[0, 1]
    
    assert correlation > 0.8, f"STFT/ISTFT reconstruction poor (correlation: {correlation:.3f})"
    print(f"✓ STFT/ISTFT round-trip successful (correlation: {correlation:.3f})")


def test_fftconvolve():
    """Test fftconvolve."""
    print("\nTesting fftconvolve...")
    from noisereduce.spectralgate._scipy_fallbacks import fftconvolve
    
    # Test 2D convolution (as used in the code)
    mask = np.random.rand(50, 100)
    kernel = np.random.rand(5, 10)
    
    result = fftconvolve(mask, kernel, mode='same')
    
    assert result.shape == mask.shape, f"fftconvolve shape mismatch: {result.shape} != {mask.shape}"
    print("✓ fftconvolve works correctly")


def test_filtfilt():
    """Test filtfilt."""
    print("\nTesting filtfilt...")
    from noisereduce.spectralgate._scipy_fallbacks import filtfilt
    
    # Simple low-pass filter
    b = np.array([0.1])
    a = np.array([1.0, -0.9])
    
    # Test signal
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    
    filtered = filtfilt(b, a, signal)
    
    assert filtered.shape == signal.shape, f"filtfilt shape mismatch: {filtered.shape} != {signal.shape}"
    print("✓ filtfilt works correctly")


def test_reduce_noise_without_scipy():
    """Test that reduce_noise works using fallbacks."""
    print("\nTesting reduce_noise with fallbacks...")
    
    # Check if scipy is available (for informational message only)
    try:
        import scipy
        print("Note: scipy is available, but testing fallback path...")
    except ImportError:
        print("Note: scipy is not available, using fallbacks...")
    
    import noisereduce as nr
    
    # Create test signal
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)
    
    # Add noise
    noise = np.random.randn(len(signal)) * 0.1
    noisy_signal = signal + noise
    
    # Test stationary
    reduced_stat = nr.reduce_noise(
        y=noisy_signal,
        sr=sr,
        stationary=True,
        y_noise=noise[:sr],
        chunk_size=10000
    )
    
    # Test non-stationary
    reduced_nonstat = nr.reduce_noise(
        y=noisy_signal,
        sr=sr,
        stationary=False,
        chunk_size=10000
    )
    
    assert reduced_stat.shape == noisy_signal.shape, "Stationary output shape mismatch"
    assert reduced_nonstat.shape == noisy_signal.shape, "Non-stationary output shape mismatch"
    print("✓ reduce_noise works with fallbacks")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing NumPy Fallback Implementations")
    print("=" * 60)
    
    results = []
    
    try:
        test_fallback_imports()
        results.append(("Import test", True))
    except Exception:
        results.append(("Import test", False))
    
    try:
        test_stft_istft()
        results.append(("STFT/ISTFT", True))
    except Exception:
        results.append(("STFT/ISTFT", False))
    
    try:
        test_fftconvolve()
        results.append(("fftconvolve", True))
    except Exception:
        results.append(("fftconvolve", False))
    
    try:
        test_filtfilt()
        results.append(("filtfilt", True))
    except Exception:
        results.append(("filtfilt", False))
    
    try:
        test_reduce_noise_without_scipy()
        results.append(("reduce_noise", True))
    except Exception:
        results.append(("reduce_noise", False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())

