import os 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from tqdm import tqdm
from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from modules.datahandling import causal_kde1d, acausal_kde1d

logger = makeLogger(__name__)
ps = PlotStyle()

class Behavior:
    """Load behavior data from csv file as class attributes
        Attributes
    ----------
    behavior: 0: chasing onset, 1: chasing offset, 2: physical contact
    behavior_type:         
    behavioral_category:   
    comment_start:         
    comment_stop:          
    dataframe: pandas dataframe with all the data            
    duration_s:             
    media_file:            
    observation_date:      
    observation_id:        
    start_s: start time of the event in seconds               
    stop_s:  stop time of the event in seconds               
    total_length:          
    """

    def __init__(self, folder_path: str) -> None:
        print(f'{folder_path}')
        LED_on_time_BORIS = np.load(os.path.join(folder_path, 'LED_on_time.npy'), allow_pickle=True)
        self.time = np.load(os.path.join(folder_path, "times.npy"), allow_pickle=True)
        csv_filename = [f for f in os.listdir(folder_path) if f.endswith('.csv')][0] # check if there are more than one csv file
        self.dataframe = read_csv(os.path.join(folder_path, csv_filename))
        self.chirps = np.load(os.path.join(folder_path, 'chirps.npy'), allow_pickle=True)
        self.chirps_ids = np.load(os.path.join(folder_path, 'chirp_ids.npy'), allow_pickle=True)

        for k, key in enumerate(self.dataframe.keys()):
            key = key.lower() 
            if ' ' in key:
                key = key.replace(' ', '_')
                if '(' in key:
                    key = key.replace('(', '')
                    key = key.replace(')', '')
            setattr(self, key, np.array(self.dataframe[self.dataframe.keys()[k]]))

        last_LED_t_BORIS = LED_on_time_BORIS[-1]
        real_time_range = self.time[-1] - self.time[0]
        factor = 1.034141
        shift = last_LED_t_BORIS - real_time_range * factor
        self.start_s = (self.start_s - shift) / factor
        self.stop_s = (self.stop_s - shift) / factor

"""
1 - chasing onset
2 - chasing offset
3 - physical contact event

temporal encpding needs to be corrected ... not exactly 25FPS.

### correspinding python code ###

    factor = 1.034141
    LED_on_time_BORIS = np.load(os.path.join(folder_path, 'LED_on_time.npy'), allow_pickle=True)
    last_LED_t_BORIS = LED_on_time_BORIS[-1]
    real_time_range = times[-1] - times[0]
    shift = last_LED_t_BORIS - real_time_range * factor

    data = pd.read_csv(os.path.join(folder_path, file[1:-7] + '.csv'))
    boris_times = data['Start (s)']
    data_times = []

    for Cevent_t in boris_times:
        Cevent_boris_times = (Cevent_t - shift) / factor
        data_times.append(Cevent_boris_times)

    data_times = np.array(data_times)
    behavior = data['Behavior']
"""


def correct_chasing_events(
    category: np.ndarray, 
    timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

    onset_ids = np.arange(
        len(category))[category == 0]
    offset_ids = np.arange(
        len(category))[category == 1]

    wrong_bh = np.arange(len(category))[category!=2][:-1][np.diff(category[category!=2])==0]
    if onset_ids[0] > offset_ids[0]:
        offset_ids = np.delete(offset_ids, 0)
        help_index = offset_ids[0]
        wrong_bh = np.append(wrong_bh[help_index])

    category = np.delete(category, wrong_bh)
    timestamps = np.delete(timestamps, wrong_bh)


    # Check whether on- or offset is longer and calculate length difference
    if len(onset_ids) > len(offset_ids):
        len_diff = len(onset_ids) - len(offset_ids)
        logger.info(f'Onsets are greater than offsets by {len_diff}')
    elif len(onset_ids) < len(offset_ids):
        len_diff = len(offset_ids) - len(onset_ids)
        logger.info(f'Offsets are greater than onsets by {len_diff}')
    elif len(onset_ids) == len(offset_ids):
        logger.info('Chasing events are equal')
    
    return category, timestamps


def event_triggered_chirps(
    event: np.ndarray, 
    chirps:np.ndarray,
    time_before_event: int,
    time_after_event: int,
    dt: float,
    width: float,
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:

    event_chirps = []   # chirps that are in specified window around event
    centered_chirps = []    # timestamps of chirps around event centered on the event timepoint

    for event_timestamp in event:
        start = event_timestamp - time_before_event
        stop = event_timestamp + time_after_event
        chirps_around_event = [c for c in chirps if (c >= start) & (c <= stop)]
        event_chirps.append(chirps_around_event)
        if len(chirps_around_event) == 0:
            continue
        else: 
            centered_chirps.append(chirps_around_event - event_timestamp)
    
    time = np.arange(-time_before_event, time_after_event, dt)
    
    # Kernel density estimation with some if's
    if len(centered_chirps) == 0:
        centered_chirps = np.array([])
        centered_chirps_convolved = np.zeros(len(time))
    else:
        centered_chirps = np.concatenate(centered_chirps, axis=0)   # convert list of arrays to one array for plotting
        centered_chirps_convolved = (acausal_kde1d(centered_chirps, time, width)) / len(event)

    return event_chirps, centered_chirps, centered_chirps_convolved


def main(datapath: str):

    foldernames = [datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath + x)]

    nrecording_chirps = []
    nrecording_chirps_fish_ids = []
    nrecording_chasing_onsets = []
    nrecording_chasing_offsets = []
    nrecording_physicals = []

    # Iterate over all recordings and save chirp- and event-timestamps
    for folder in foldernames:
        # exclude folder with empty LED_on_time.npy
        if folder == '../data/mount_data/2020-05-12-10_00/':
            continue

        bh = Behavior(folder)
        
        # Chirps are already sorted
        category = bh.behavior
        timestamps = bh.start_s
        chirps = bh.chirps
        nrecording_chirps.append(chirps)
        chirps_fish_ids = bh.chirps_ids
        nrecording_chirps_fish_ids.append(chirps_fish_ids)
        fish_ids = np.unique(chirps_fish_ids)

        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(category, timestamps)

        # Split categories
        chasing_onsets = timestamps[category == 0]
        nrecording_chasing_onsets.append(chasing_onsets)
        chasing_offsets = timestamps[category == 1]
        nrecording_chasing_offsets.append(chasing_offsets)
        physical_contacts = timestamps[category == 2]
        nrecording_physicals.append(physical_contacts)
    

    # Define time window for chirps around event analysis
    time_before_event = 30
    time_after_event = 60
    dt = 0.01
    width = 1.5   # width of kernel, currently gaussian kernel
    time = np.arange(-time_before_event, time_after_event, dt)

    ##### Chirps around events, all fish, all recordings #####
    # Centered chirps per event type
    nrecording_centered_onset_chirps = []
    nrecording_centered_offset_chirps = []
    nrecording_centered_physical_chirps = []
    # Bootstrapped chirps per recording and per event: 27[1000[n]] 27 recs, 1000 shuffles, n chirps
    nrecording_shuffled_convolved_onset_chirps = []
    nrecording_shuffled_convolved_offset_chirps = []
    nrecording_shuffled_convolved_physical_chirps = []

    nbootstrapping = 10

    for i in range(len(nrecording_chirps)):
        chirps = nrecording_chirps[i]
        chasing_onsets = nrecording_chasing_onsets[i]
        chasing_offsets = nrecording_chasing_offsets[i]
        physical_contacts = nrecording_physicals[i]

        # Chirps around chasing onsets
        _, centered_chasing_onset_chirps, _ = event_triggered_chirps(chasing_onsets, chirps, time_before_event, time_after_event, dt, width)
        # Chirps around chasing offsets
        _, centered_chasing_offset_chirps, _ = event_triggered_chirps(chasing_offsets, chirps, time_before_event, time_after_event, dt, width)
        # Chirps around physical contacts
        _, centered_physical_chirps, _ = event_triggered_chirps(physical_contacts, chirps, time_before_event, time_after_event, dt, width)

        nrecording_centered_onset_chirps.append(centered_chasing_onset_chirps)
        nrecording_centered_offset_chirps.append(centered_chasing_offset_chirps)
        nrecording_centered_physical_chirps.append(centered_physical_chirps)

        ## Shuffled chirps ##
        nshuffled_onset_chirps = []
        nshuffled_offset_chirps = []
        nshuffled_physical_chirps = []

        for i in tqdm(range(nbootstrapping)):
        # Calculate interchirp intervals; add first chirp timestamp in beginning to get equal lengths
            interchirp_intervals = np.append(np.array([chirps[0]]), np.diff(chirps))
            np.random.shuffle(interchirp_intervals)
            shuffled_chirps = np.cumsum(interchirp_intervals)
            # Shuffled chasing onset chirps
            _, _, cc_shuffled_onset_chirps = event_triggered_chirps(chasing_onsets, shuffled_chirps, time_before_event, time_after_event, dt, width)
            nshuffled_onset_chirps.append(cc_shuffled_onset_chirps)
            # Shuffled chasing offset chirps
            _, _, cc_shuffled_offset_chirps = event_triggered_chirps(chasing_offsets, shuffled_chirps, time_before_event, time_after_event, dt, width)
            nshuffled_offset_chirps.append(cc_shuffled_offset_chirps)
            # Shuffled physical contact chirps
            _, _, cc_shuffled_physical_chirps = event_triggered_chirps(physical_contacts, shuffled_chirps, time_before_event, time_after_event, dt, width)
            nshuffled_physical_chirps.append(cc_shuffled_physical_chirps)
        
        nrecording_shuffled_convolved_onset_chirps.append(nshuffled_onset_chirps)
        nrecording_shuffled_convolved_offset_chirps.append(nshuffled_offset_chirps)
        nrecording_shuffled_convolved_physical_chirps.append(nshuffled_physical_chirps)

    #  vstack um 1. Dim zu cutten
    nrecording_shuffled_convolved_onset_chirps = np.vstack(nrecording_shuffled_convolved_onset_chirps)
    nrecording_shuffled_convolved_offset_chirps = np.vstack(nrecording_shuffled_convolved_offset_chirps)
    nrecording_shuffled_convolved_physical_chirps = np.vstack(nrecording_shuffled_convolved_physical_chirps)
    
    shuffled_q5_onset, shuffled_median_onset, shuffled_q95_onset = np.percentile(
        nrecording_shuffled_convolved_onset_chirps, (5, 50, 95), axis=0)
    shuffled_q5_offset, shuffled_median_offset, shuffled_q95_offset = np.percentile(
        nrecording_shuffled_convolved_offset_chirps, (5, 50, 95), axis=0)
    shuffled_q5_physical, shuffled_median_physical, shuffled_q95_physical = np.percentile(
        nrecording_shuffled_convolved_physical_chirps, (5, 50, 95), axis=0)

    # Flatten all chirps 
    all_chirps = np.concatenate(nrecording_chirps).ravel()  # not centered

    # Flatten event timestamps
    all_onsets = np.concatenate(nrecording_chasing_onsets).ravel() # not centered
    all_offsets = np.concatenate(nrecording_chasing_offsets).ravel() # not centered
    all_physicals = np.concatenate(nrecording_physicals).ravel() # not centered

    # Flatten all chirps around events
    all_onset_chirps = np.concatenate(nrecording_centered_onset_chirps).ravel()   # centered
    all_offset_chirps = np.concatenate(nrecording_centered_offset_chirps).ravel() # centered
    all_physical_chirps = np.concatenate(nrecording_centered_physical_chirps).ravel() # centered

    # Convolute all chirps
    # Divide by total number of each event over all recordings
    all_onset_chirps_convolved = (acausal_kde1d(all_onset_chirps, time, width)) / len(all_onsets)
    all_offset_chirps_convolved = (acausal_kde1d(all_offset_chirps, time, width)) / len(all_offsets)
    all_physical_chirps_convolved = (acausal_kde1d(all_physical_chirps, time, width)) / len(all_physicals)


    # Plot all events with all shuffled
    fig, ax = plt.subplots(1, 3, figsize=(28*ps.cm, 16*ps.cm, ), constrained_layout=True, sharey='all')
    # offsets = np.arange(1,28,1)
    ax[0].set_xlabel('Time[s]')
    # Plot chasing onsets
    ax[0].set_ylabel('Chirp rate [Hz]')
    ax[0].plot(time, all_onset_chirps_convolved, color=ps.yellow, zorder=2)
    ax0 = ax[0].twinx()
    nrecording_centered_onset_chirps = np.asarray(nrecording_centered_onset_chirps, dtype=object)
    ax0.eventplot(np.array(nrecording_centered_onset_chirps), linelengths=0.5, colors=ps.gray, alpha=0.25, zorder=1)
    ax0.vlines(0, 0, 1.5, ps.black, 'dashed')
    ax[0].set_zorder(ax0.get_zorder()+1)
    ax[0].patch.set_visible(False)
    ax0.set_yticklabels([])
    ax0.set_yticks([])
    ax[0].fill_between(time, shuffled_q5_onset, shuffled_q95_onset, color=ps.gray, alpha=0.5)
    ax[0].plot(time, shuffled_median_onset, color=ps.black)
    # Plot chasing offets
    ax[1].set_xlabel('Time[s]')
    ax[1].plot(time, all_offset_chirps_convolved, color=ps.orange, zorder=2)
    ax1 = ax[1].twinx()
    nrecording_centered_offset_chirps = np.asarray(nrecording_centered_offset_chirps, dtype=object)
    ax1.eventplot(np.array(nrecording_centered_offset_chirps), linelengths=0.5, colors=ps.gray, alpha=0.25, zorder=1)
    ax1.vlines(0, 0, 1.5, ps.black, 'dashed')
    ax[1].set_zorder(ax1.get_zorder()+1)
    ax[1].patch.set_visible(False)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax[1].fill_between(time, shuffled_q5_offset, shuffled_q95_offset, color=ps.gray, alpha=0.5)
    ax[1].plot(time, shuffled_median_offset, color=ps.black)
    # Plot physical contacts
    ax[2].set_xlabel('Time[s]')
    ax[2].plot(time, all_physical_chirps_convolved, color=ps.maroon, zorder=2)
    ax2 = ax[2].twinx()
    nrecording_centered_physical_chirps = np.asarray(nrecording_centered_physical_chirps, dtype=object)
    ax2.eventplot(np.array(nrecording_centered_physical_chirps), linelengths=0.5, colors=ps.gray, alpha=0.25, zorder=1)
    ax2.vlines(0, 0, 1.5, ps.black, 'dashed')
    ax[2].set_zorder(ax2.get_zorder()+1)
    ax[2].patch.set_visible(False)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax[2].fill_between(time, shuffled_q5_physical, shuffled_q95_physical, color=ps.gray, alpha=0.5)
    ax[2].plot(time, shuffled_median_physical, ps.black)
    plt.show()
    # plt.close()

    embed()
    
    # chasing_durations = []
    # # Calculate chasing duration to evaluate a nice time window for kernel density estimation
    # for onset, offset in zip(chasing_onsets, chasing_offsets):
    #     duration = offset - onset
    #     chasing_durations.append(duration)

    # fig, ax = plt.subplots()
    # ax.boxplot(chasing_durations)
    # plt.show()
    # plt.close()


    # # Associate chirps to individual fish
    # fish1 = chirps[chirps_fish_ids == fish_ids[0]]
    # fish2 = chirps[chirps_fish_ids == fish_ids[1]]
    # fish = [len(fish1), len(fish2)]

    # Concolution over all recordings
    # Rasterplot for each recording


    # #### Chirps around events, winner VS loser, one recording ####
    # # Load file with fish ids and winner/loser info
    # meta = pd.read_csv('../data/mount_data/order_meta.csv')
    # current_recording = meta[meta.index == 43]
    # fish1 = current_recording['rec_id1'].values
    # fish2 = current_recording['rec_id2'].values
    # # Implement check if fish_ids from meta and chirp detection are the same???
    # winner = current_recording['winner'].values
    
    # if winner == fish1:
    #     loser = fish2
    # elif winner == fish2:
    #     loser = fish1

    # winner_chirps = chirps[chirps_fish_ids == winner]
    # loser_chirps = chirps[chirps_fish_ids == loser]

    # # Event triggered winner chirps
    # _, winner_centered_onset, winner_cc_onset = event_triggered_chirps(chasing_onsets, winner_chirps, time_before_event, time_after_event, dt, width)
    # _, winner_centered_offset, winner_cc_offset = event_triggered_chirps(chasing_offsets, winner_chirps, time_before_event, time_after_event, dt, width)
    # _, winner_centered_physical, winner_cc_physical = event_triggered_chirps(physical_contacts, winner_chirps, time_before_event, time_after_event, dt, width)

    # # Event triggered loser chirps
    # _, loser_centered_onset, loser_cc_onset = event_triggered_chirps(chasing_onsets, loser_chirps, time_before_event, time_after_event, dt, width)
    # _, loser_centered_offset, loser_cc_offset = event_triggered_chirps(chasing_offsets, loser_chirps, time_before_event, time_after_event, dt, width)
    # _, loser_centered_physical, loser_cc_physical = event_triggered_chirps(physical_contacts, loser_chirps, time_before_event, time_after_event, dt, width)

    # ########## Winner VS Loser plot ##########
    # fig, ax = plt.subplots(2, 3, figsize=(50 / 2.54, 15 / 2.54), constrained_layout=True, sharey='row')
    # offset = [1.35]
    # ax[1][0].set_xlabel('Time[s]')
    # ax[1][1].set_xlabel('Time[s]')
    # ax[1][2].set_xlabel('Time[s]')
    # # Plot winner chasing onsets
    # ax[0][0].set_ylabel('Chirp rate [Hz]')
    # ax[0][0].plot(time, winner_cc_onset, color='tab:blue', zorder=100)
    # ax0 = ax[0][0].twinx()
    # ax0.eventplot(np.array([winner_centered_onset]), lineoffsets=offset, linelengths=0.1, colors=['tab:green'], alpha=0.25, zorder=-100)
    # ax0.set_ylabel('Event')
    # ax0.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[0][0].set_zorder(ax0.get_zorder()+1)
    # ax[0][0].patch.set_visible(False)
    # ax0.set_yticklabels([])
    # ax0.set_yticks([])
    # # Plot winner chasing offets
    # ax[0][1].plot(time, winner_cc_offset, color='tab:blue', zorder=100)
    # ax1 = ax[0][1].twinx()
    # ax1.eventplot(np.array([winner_centered_offset]), lineoffsets=offset, linelengths=0.1, colors=['tab:purple'], alpha=0.25, zorder=-100)
    # ax1.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[0][1].set_zorder(ax1.get_zorder()+1)
    # ax[0][1].patch.set_visible(False)
    # ax1.set_yticklabels([])
    # ax1.set_yticks([])
    # # Plot winner physical contacts
    # ax[0][2].plot(time, winner_cc_physical, color='tab:blue', zorder=100)
    # ax2 = ax[0][2].twinx()
    # ax2.eventplot(np.array([winner_centered_physical]), lineoffsets=offset, linelengths=0.1, colors=['tab:red'], alpha=0.25, zorder=-100)
    # ax2.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[0][2].set_zorder(ax2.get_zorder()+1)
    # ax[0][2].patch.set_visible(False)
    # ax2.set_yticklabels([])
    # ax2.set_yticks([])
    # # Plot loser chasing onsets
    # ax[1][0].set_ylabel('Chirp rate [Hz]')
    # ax[1][0].plot(time, loser_cc_onset, color='tab:blue', zorder=100)
    # ax3 = ax[1][0].twinx()
    # ax3.eventplot(np.array([loser_centered_onset]), lineoffsets=offset, linelengths=0.1, colors=['tab:green'], alpha=0.25, zorder=-100)
    # ax3.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[1][0].set_zorder(ax3.get_zorder()+1)
    # ax[1][0].patch.set_visible(False)
    # ax3.set_yticklabels([])
    # ax3.set_yticks([])
    # # Plot loser chasing offsets
    # ax[1][1].plot(time, loser_cc_offset, color='tab:blue', zorder=100)
    # ax4 = ax[1][1].twinx()
    # ax4.eventplot(np.array([loser_centered_offset]), lineoffsets=offset, linelengths=0.1, colors=['tab:purple'], alpha=0.25, zorder=-100)
    # ax4.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[1][1].set_zorder(ax4.get_zorder()+1)
    # ax[1][1].patch.set_visible(False)
    # ax4.set_yticklabels([])
    # ax4.set_yticks([])
    # # Plot loser physical contacts
    # ax[1][2].plot(time, loser_cc_physical, color='tab:blue', zorder=100)
    # ax5 = ax[1][2].twinx()
    # ax5.eventplot(np.array([loser_centered_physical]), lineoffsets=offset, linelengths=0.1, colors=['tab:red'], alpha=0.25, zorder=-100)
    # ax5.vlines(0, 0, 1.5, 'tab:grey', 'dashed')
    # ax[1][2].set_zorder(ax5.get_zorder()+1)
    # ax[1][2].patch.set_visible(False)
    # ax5.set_yticklabels([])
    # ax5.set_yticks([])
    # plt.show()
    # plt.close()
    

    # for i in range(len(fish_ids)):
    #     fish = fish_ids[i]
    #     chirps_temp = chirps[chirps_fish_ids == fish]
    #     print(fish)

    #### Chirps around events, only losers, one recording ####



if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/'
    main(datapath)
