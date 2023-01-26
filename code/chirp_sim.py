import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from thunderfish import fakefish

from modules.filters import bandpass_filter
from modules.datahandling import instantaneous_frequency
from modules.simulations import create_chirp



# trying thunderfish fakefish chirp simulation ---------------------------------
samplerate = 44100
freq, ampl = fakefish.chirps(eodf=500, chirp_contrast=0.2)
data = fakefish.wavefish_eods(fish='Alepto', frequency=freq, phase0=3, samplerate=samplerate)

# filter signal with bandpass_filter
data_filterd = bandpass_filter(data*ampl+1, samplerate, 0.01, 1.99)
embed()
data_freq_time, data_freq = instantaneous_frequency(data, samplerate, 5)


fig, ax = plt.subplots(4, 1, figsize=(20 / 2.54, 12 / 2.54), sharex=True)

ax[0].plot(np.arange(len(data))/samplerate, data*ampl)
#ax[0].scatter(true_zero, np.zeros_like(true_zero), color='red')
ax[1].plot(np.arange(len(data_filterd))/samplerate, data_filterd)
ax[2].plot(np.arange(len(freq))/samplerate, freq)
ax[3].plot(data_freq_time, data_freq)


plt.show()
embed()




