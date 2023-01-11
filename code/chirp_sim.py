from modules.filters import create_chirp, bandpass_filter
import matplotlib.pyplot as plt 
from chirpdetection import instantaneos_frequency
from IPython import embed
# create chirp

time, signal, ampl, freq = create_chirp(chirptimes=[0.05, 0.2501, 0.38734, 0.48332, 0.73434, 0.823424], )

# filter signal with bandpass_filter 

signal = bandpass_filter(signal, 1/0.00001, 495, 505)
embed()
exit()
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(time, signal)

# plot instatneous frequency

baseline_freq_time, baseline_freq = instantaneos_frequency(signal, 1/0.00001)
axs[1].plot(baseline_freq_time[1:], baseline_freq[1:])



plt.show()
