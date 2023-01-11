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


def create_chirp(
    eodf=500,
    chirpsize=100,
    chirpduration=0.015,
    ampl_reduction=0.05,
    chirptimes=[0.05, 0.2],
    kurtosis=1.0,
    duration=1.0,
    dt=0.00001,
):
    """create a fake fish eod that contains chirps at the given times. EOF is a simple sinewave. Chirps are modeled with Gaussian profiles in amplitude reduction and frequency ecxcursion.

    Args:
        eodf (int, optional): The chriping fish's EOD frequency. Defaults to 500 Hz.
        chirpsize (int, optional): the size of the chrip's frequency excursion. Defaults to 100 Hz.
        chirpwidth (float, optional): the duration of the chirp. Defaults to 0.015 s.
        ampl_reduction (float, optional): Amount of amplitude reduction during the chrips. Defaults to 0.05, i.e. 5\%
        chirptimes (list, optional): Times of chirp centers. Defaults to [0.05, 0.2].
        kurtosis (float, optional): The kurtosis of the Gaussian profiles. Defaults to 1.0
        dt (float, optional): the stepsize of the simulation. Defaults to 0.00001 s.

    Returns:
        np.ndarray: the time
        np.ndarray: the eod
        np.ndarray: the amplitude profile
        np.adarray: tha frequency profile
    """
    p = 0.0

    time = np.arange(0.0, duration, dt)
    signal = np.zeros_like(time)
    ampl = np.ones_like(time)
    freq = np.ones_like(time)

    ck = 0
    csig = 0.5 * chirpduration / np.power(2.0 * np.log(10.0), 0.5 / kurtosis)
    #csig = csig*-1
    for k, t in enumerate(time):
        a = 1.0
        f = eodf

        if ck < len(chirptimes):
            if np.abs(t - chirptimes[ck]) < 2.0 * chirpduration:
                x = t - chirptimes[ck]
                gg = np.exp(-0.5 * np.power((x / csig) ** 2, kurtosis))
                cc = chirpsize * gg

                # g = np.exp( -0.5 * (x/csig)**2 )
                f = chirpsize * gg + eodf
                a *= 1.0 - ampl_reduction * gg
            elif t > chirptimes[ck] + 2.0 * chirpduration:
                ck += 1
        freq[k] = f
        ampl[k] = a
        p += f * dt
        signal[k] = -1 * a * np.sin(2 * np.pi * p)

    return time, signal, ampl, freq

