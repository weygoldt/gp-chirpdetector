import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from thunderfish import fakefish

from modules.filters import bandpass_filter
from chirpdetection import instantaneos_frequency
from modules.simulations import create_chirp

# create chirp

time, signal, ampl, freq = create_chirp(
    chirptimes=[0.05, 0.2501, 0.38734, 0.48332, 0.73434, 0.823424], )
# filter signal with bandpass_filter
signal = bandpass_filter(signal, 1/0.00001, 495, 505)
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(np.arange(len(ampl)), ampl)

# plot instatneous frequency

baseline_freq_time, baseline_freq = instantaneos_frequency(signal, 1/0.00001)
axs[1].plot(baseline_freq_time[1:], baseline_freq[1:])
plt.close()


# trying thunderfish fakefish chirp simulation ---------------------------------
samplerate = 44100
freq, ampl = fakefish.chirps(eodf=500, chirp_contrast=0.2)
data = fakefish.wavefish_eods(fish='Alepto', frequency=freq, phase0=3)

# filter signal with bandpass_filter
data_filterd = bandpass_filter(data*ampl+1, samplerate, 0.01, 1.99)
data_freq_time, data_freq = instantaneos_frequency(data*ampl, samplerate)


fig, ax = plt.subplots(4, 1, figsize=(20 / 2.54, 12 / 2.54), sharex=True)

ax[0].plot(np.arange(len(data))/samplerate, data*ampl+1)
ax[1].plot(np.arange(len(data_filterd))/samplerate, data_filterd)
ax[2].plot(np.arange(len(freq))/samplerate, freq)
ax[3].plot(data_freq_time[1:], data_freq[1:])


plt.show()
embed()