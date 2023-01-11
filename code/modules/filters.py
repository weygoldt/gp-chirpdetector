from scipy.signal import butter, sosfiltfilt
import numpy as np


def bandpass_filter(data, rate, lowf=100, highf=1100):
    sos = butter(2, (lowf, highf), "bandpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def highpass_filter(
    data,
    rate,
    cutoff=100,
):

    sos = butter(2, cutoff, "highpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def lowpass_filter(
    data,
    rate,
    cutoff=100,
):

    sos = butter(2, cutoff, "lowpass", fs=rate, output="sos")
    fdata = sosfiltfilt(sos, data)
    return fdata


def envelope(data, rate, freq=100):
    sos = butter(2, freq, "lowpass", fs=rate, output="sos")
    envelope = np.sqrt(2) * sosfiltfilt(sos, np.abs(data))
    return envelope
