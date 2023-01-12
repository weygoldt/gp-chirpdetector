from scipy.signal import butter, sosfiltfilt
import numpy as np


def bandpass_filter(
        data: np.ndarray,
        rate: float,
        lowf: float,
        highf: float,
) -> np.ndarray:
    """Bandpass filter a signal.

    Parameters
    ----------
    data : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    lowf : float
        The low cutoff frequency
    highf : float
        The high cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, (lowf, highf), "bandpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def highpass_filter(
    data: np.ndarray,
    rate: float,
    cutoff: float,
) -> np.ndarray:
    """Highpass filter a signal.

    Parameters
    ----------
    data : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    cutoff : float
        The cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, cutoff, "highpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def lowpass_filter(
    data: np.ndarray,
    rate: float,
    cutoff: float
) -> np.ndarray:
    """Lowpass filter a signal.

    Parameters
    ----------
    data : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    cutoff : float
        The cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, cutoff, "lowpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def envelope(data: np.ndarray, rate: float, freq: float) -> np.ndarray:
    """Calculate the envelope of a signal using a lowpass filter.

    Parameters
    ----------
    data : np.ndarray
        The signal to calculate the envelope of
    rate : float
        The sampling rate of the signal
    freq : float
        The cutoff frequency of the lowpass filter

    Returns
    -------
    np.ndarray
        The envelope of the signal
    """
    sos = butter(2, freq, "lowpass", fs=rate, output="sos")
    envelope = np.sqrt(2) * sosfiltfilt(sos, np.abs(data))
    return envelope
