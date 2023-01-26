---
date: 2023-01-26
author: Patrick Weygoldt
type: talk
speakers:
  - name: 
    affiliation: 
  - name: 
    affiliation: 
aliases: 
tags: 
---
# Chirp detection poster script

## 10 minute presentation

introduction:
- Project goal: Develop a chirp detection algorithm
- What are chirps?
	- short frequency excursions in ms range of EOD (electric organ discharge) of weakly el. fish.
- Show plot:
		- spectrogram of the EODf of two fish (two lines)
			- frequency resolution 150Hz
			- nfft: number of windows/datapoints over which the Fourier transform is performed
		- frequency over time is shown
		- color indicates power
		- chirp = upper line, frequency increases shortly
 - Problem:
		- to resolve chirps on the time domain, frequency domain too coarse
		- if lower fish chirps it becomes even harder
		=> time-frequency uncertainty problem (general)
- Goal: Improve existing detection methods to detect and assign chirps for electric recordings with n fish

Chirp detection algorithm
Availabe data:
1. Raw electrical signal (EOD of multiple fish) over n electrodes (n = 11)
2. Tracked frequency bands on spectrogram (pre-tracked): just as in upper right plot, but with lower sampling rate (3Hz)
	- for one frequency we want the electrode on which the power of the f is the greatest
	- power of the strongest of 11 electrodes for each frequencypoint in time was used to track the frequency band
	- we cannot track freq on spectrogram's time resolution is too low for detecting chirps if you wanna distinguish between the fish
Feature extraction (in 5s rolling window):
1. Bandpass filter around the tracked frequency band for one individual (+-5Hz)
	- first subplot grey, red = envelope of filtered baseline
2. Dynamic bandpass filter above baseline (+-5Hz) = 2nd subplot, gray filtered search frequency, orange = envelope
	- dynamic search window above the current fish of interest
		- Why dynamic: if another fish has a higher frequency, we need to find a window without another fish to be able to detect the chirp
			- window above fish: look if there's another fish (array stuff), True/false thing, find longest subarray
		- chirps excursions always increase the frequency and decrease the amplitude
		- to find chirp, we need to search above the fish and look for break down in amplitude
		- no peaks in filtered above = no chirp
		- amplitude break down of baseline can have multiple reasons (e.g. fish swims away, stone)
3. Instantaneous frequency of baseline = 3rd subplot, gray filtered inst., yellow = envelope
	- calculated on filtered raw data
	- get zero crossings of each period and calculate frequency manually
	- we know that chirps are increases in frequency, here we look at the frequency feature of chirps
Peak detection:
1. Detect peaks on bandpass filtered and inverted baseline envelope (lower red line)
2. Detect peaks on bandpass filtered search frequency (lower orange line)
3. Detect peaks on absolute inst. freq (lower yellow line)
	- Peak prominence: Minimal distance from highest peak to next peak
Peak classification:
- all three features have to be present at once in a 20ms window (appr. chirp length) in strongest electrodes
- mean of peak timestamps of features is saved as chirp timestamp

Chirps in dyadic competitions
1. Competition experiment by Til Raab:
	- two fish compete for one shelter
	- 6h recording, 3h light, 3h dark
	- electrical and video recordings
	- with video recordings, behavior was tracked and assigned to an antagonistic category: Chasing (on- and offset) and physical contacts
- we did behavioral analysis with the detected chirps of our algorithm
2. Plot: Contact an chasing event timepoints, chirps of both fish, tracked frequency bands
	- from literature: Chirps assumed as submissive signal by loser fish
4. Winner Loser boxplot
	- chirps counts for winner and loser (n=22 recordings with winner and loser)
	- loser tends to chirp more (Wilcoxon not significant, but trend with 0.054)
	- white lines are paired fish for competition
5. Size difference plot
	- Literature: Larger wish usually wins. (Larger resource holding potential theory, Till with rises)
	- The smaller the size difference between fish, the more chirps are emitted taken winner and loser together
	- correlation within winners and losers are not significant
	- n = 21 because one recording with equal fish size was excluded
6. Frequency plot
	- Literature: Males are more aggressive and chirp more, males have a lower EODf
	- EODf has no effect on the competition outcome

Chirps emitted by loser fish might stop chasing events
- Chirps were centered around the timestamps of each event in a +-60s time window (for each category and each recording)
- kernel density estimation of centered chirps (gaussian kernel with 2s width and 10ms resolution)
- We show some example plots
1. First plot: No correlation case for chasing offset
	- no correlation between chirping and the offset
	- this was the case for most recordings and all events
2. Second plot: Correlation case for chirping and offset
	- For some few dyads/individuals, chirp rate increases drastically before the chasing offset
	- also slightly visible before chasing onsets
	- no correlation for physical contacts
3. Third plot: Time of chasing events in the night VS the chirps during the chasing events and during night
	- fraction of chirps during chasings is not increased relative to the fraction of chasing events overall
	- Chirps do not seem to have an increased significance for chasing events
	- only for some few dyads the chirp rate increased during chasings
- Gray/black areas:
	- bootstrapped data (n = 50)
	- all chirps for one recording during the night (because there more chirps)
	- all shuffled chirps again centered around event and convolved

Conclusion:
- First tests indicate that our algorithm is able to detect chirps in recordings of multiple fish
- Algorithm results were applied on behavioral data for further analysis


## 2 Minute presentation

Introduction:
- Project goal: Develop a chirp detection algorithm
- What are chirps?
	- Short frequency excursions in ms range of EOD (electric organ discharge) of weakly el. fish.
	-  to resolve chirps on the time domain, frequency domain too coarse, especially for multiple fish
- Goal: Improve existing detection methods

Detection algorithm
- Improved existing detection methods by extracting 3 features that change during a chirp but are not limited by the time frequency uncertainty
	1. Amplitude drop of EOD (show trough)
	2. Peaks of instantaneous frequency of EOD
	3. Peaks in the dynamically adjusted frequency band above the fish's baseline EODf (special).
- Detected and classified peaks are chirp times

Application: Chirps during competition
- Detected over 10000 chirps in real data from a competition experiment
- Analysis of the relationship of chirps and competition events
- Fish competed for a shelter
- Were able to replicate some findings from literature
	- e.g. loser fish tend to chirp more
	- Other findings are not that clear and require the consideration of more factors, e.g. sex
- We explored how the chirp rate changes during onsets and offsets of chasing events
	- For some recordings, chirping increased strongly before the offset of a chasing, for some nothing happens
	- The number of chirps during chasings is only elevated for some dyads

Conclusion
- Algorithm can be used to detect chirps
- We could replicate some literature findings and motivate further examination