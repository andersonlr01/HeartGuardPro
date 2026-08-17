"""
Signal preprocessing filters for HeartGuard Pro.

Implements the 4th-order Butterworth bandpass filter described in the
manuscript (Section III.B.1), isolating the cardiac frequency band
0.5-8.0 Hz from raw PPG signals sampled at 50 Hz.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butterworth_bandpass(signal: np.ndarray, fs: float = 50.0,
                          low_hz: float = 0.5, high_hz: float = 8.0,
                          order: int = 4) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Raw PPG signal.
    fs : float
        Sampling frequency in Hz (paper: 50 Hz, 24-bit ADC).
    low_hz, high_hz : float
        Cardiac passband edges (paper: 0.5-8.0 Hz).
    order : int
        Filter order (paper: n = 4).

    Returns
    -------
    np.ndarray
        Zero-phase filtered signal, same shape as input.
    """
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    if not (0 < low < high < 1):
        raise ValueError(
            f"Invalid band [{low_hz}, {high_hz}] Hz for fs={fs} Hz "
            f"(Nyquist={nyq} Hz). Increase fs or lower high_hz."
        )
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0,
                  quality: float = 30.0) -> np.ndarray:
    """Notch filter for powerline interference (50/60 Hz), commonly needed
    for real hardware PPG/ECG acquisition even though not explicitly
    itemized in the manuscript's preprocessing list."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization per-window."""
    mu, sigma = np.mean(signal), np.std(signal)
    if sigma < 1e-8:
        return signal - mu
    return (signal - mu) / sigma
